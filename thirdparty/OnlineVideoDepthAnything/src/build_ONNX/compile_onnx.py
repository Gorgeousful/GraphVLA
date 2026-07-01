import sys
import os

import onnxruntime as ort
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
from einops import rearrange

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.build_ONNX.compile_utils import load_dummy_data
from src.utils.loading_utils import load_frames_as_numpy, load_video_as_numpy
from src.models.video_depth import onlineVideoDepthAnything
from src.models.utils.preprocessing import VideoPreprocessor


def _init_video_depth_anything(config_path: str, device: str = 'cuda:0') -> onlineVideoDepthAnything:
    '''
    Initialise oVDA
    ---------------
    Initialise and loades oVDA. Preapares the model for inference.

    Parameters
    ----------
    config_path : str
        Path to the config file
    device : str, default 'cuda:0'
        Device to load the model on

    Returns
    -------
    oVDA : onlineVideoDepthAnything
        oVDA model (pytorch model)
    '''
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    oVDA = onlineVideoDepthAnything(**config['net'])

    ckpt = torch.load(os.path.join(config['pretrained_path']), map_location='cpu')

    oVDA.load_state_dict(ckpt)
    oVDA = oVDA.to(device)
    oVDA.eval()

    return oVDA


def test_video_depth_anything(config_path: str, frame_path: str, input_size=518, 
                              device: str = 'cuda:0') -> None:
    '''
    Test ONNX model
    -------------------
    Tests ONNX model. Only for checking if build was successful. Not meant for measuring time. This runs on the cpu.

    WARNING: This code is for testing only.

    Parameters
    ----------
    config_path : str
        Path to the config file
    frame_path : str
        Path to the input frames or input video
    input_size : int, default 518
        Input size of the model, set acording to build model.
    device : str, default 'cuda:0'
        Device to load the model on. Does not change the cexecution device
    
    Returns
    -------
    None    
    '''
    # Load model
    model = _init_video_depth_anything(config_path, device=device)

    # Load frames
    if frame_path.endswith('.mp4'):
        inputs = load_video_as_numpy(frame_path)
    else:
        inputs = load_frames_as_numpy(frame_path)

    # Run oVDA
    depth_predictions = model.infer_video_depth(inputs, device=device, preprocess_device=device, input_size=input_size, fp32=False,
                                         print_process_res=True, output_raw=False)

    depth_predictions = (depth_predictions + depth_predictions.min())
    depth_predictions = depth_predictions / depth_predictions.max()
    grid = make_grid(torch.from_numpy(depth_predictions)[0].unsqueeze(1), nrow=4).numpy()
    grid = rearrange(grid, 'c h w -> h w c')
    grid = np.mean(grid, axis=2)

    # Save with colormap
    plt.imsave('output.png', grid, cmap='Spectral')

    del model
    torch.cuda.empty_cache()


def compile_onnx(ckpt_path: str, onnx_path: str, h: int, w: int,
                 device: str = 'cuda:0') -> None:
    '''
    Compile model to ONNX
    -------------------
    Exports the oVDA Model to ONNX format. This is builds a model Tree. Can easily be visualized. 
    Does only include the model forward. Since we want to transform this into an TensorRT model 
    it has to be non dynamic, meaning the height and width is fixed from this point on and can 
    not be changed anymore. 

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file
    onnx_path : str
        Path to the output ONNX file
    h : int
        Height of the input frames
    w : int
        Width of the input frames 
    device: str, default 'cuda:0'
        device the model is loaded to. Default 'cuda:0'. Does not change the execution device.
    
    Returns
    -------
    None
    '''
    model = _init_video_depth_anything(ckpt_path, device=device)
    cache_dict = model.setup_cache(h=h, w=w, device=device)
    inputs = torch.rand((1, 1, 3, h, w)).to(device)
    mask_indices = torch.tensor(list(range(1, 16))).to(device)
    input_position = torch.tensor([0, 1]).to(device)

    cache_input_names = [f"cache_dict.{k}" for k in cache_dict.keys()]
    cache_output_names = [f"output_cache.{k}" for k in cache_dict.keys()]
    input_names = ["input_frame", *cache_input_names, "mask_indices", "input_position"]
    output_names = ["pred_depth", *cache_output_names]

    torch.onnx.export(
        model,
        args=(inputs, cache_dict, mask_indices, input_position),
        f=onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        export_params=True,
        do_constant_folding=True,
    )

    del model
    torch.cuda.empty_cache()

def test_onnx(onnx_path: str, frame_path: str, window_size: int = 16, input_size=518) -> None:
    '''
    Test builded model 
    ------------------
    Tests the builded model with the given input video. This will run on the cpu and is therefore slow. 
    If the test is successful this does not mean that the TensorRT build will be successfull, since in 
    ONNX models there can still be if statements which are not supported by TensorRT.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX file
    frame_path : str
        Path to the input
    window_size: int, default = 16
        Gives the Size of the cache and the sliding window. Must be the same as in the loaded model. 
    input_size: int, default = 518
        Input size of the model, set acording to build model. Because we use the standart preprocessing.
        The true input resolution changes depending on the aspect ratio of the video provided. 
        Process Resolution will be printed out. 
    
    Returns
    -------
    None

    '''
    session = ort.InferenceSession(onnx_path)

    # Load frames
    if frame_path.endswith('.mp4'):
        inputs = load_video_as_numpy(frame_path)
    else:
        inputs = load_frames_as_numpy(frame_path)
    t, h, w, c = inputs.shape

    frame_height, frame_width = inputs[0].shape[:2]
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    if ratio > 1.78:  # VDA recommendet to process video with ratio smaller than 16:9 due to memory limitation
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    
    pre = VideoPreprocessor(input_size=input_size, device='cpu', ensure_multiple_of=14, keep_aspect_ratio=True, resize_method='lower_bound')
    prepared_frames = pre.preprocess(inputs)

    b, t, c, h, w = prepared_frames.size()
    print(f'Inferring Video Depth at res: {h}x{w}')

    _, cache_dict, _, _ = load_dummy_data(h=h, w=w, format="numpy", cache_size=window_size)

    cache_size = 0
    mask_indices = np.arange(1, window_size, dtype=np.int64)
    input_position = np.array([0, 1], dtype=np.int64)

    # Run depth predictions
    depth_predictions = []
    for i in tqdm(range(prepared_frames.shape[1])):
        input_frame = prepared_frames[:, i, :, :, :].unsqueeze(dim=1).numpy()  # Shape: (1, 1, C, H, W)

        input_feed = {"input_frame": input_frame, "mask_indices": mask_indices, "input_position": input_position}
        for k, v in cache_dict.items():
            input_feed[f"cache_dict.{k}"] = v

        output_names = [output.name for output in session.get_outputs()]
        
        outputs = session.run(output_names, input_feed)

        pred_depth = outputs[0]
        pred_depth = pred_depth.squeeze(1).reshape((1, 1, *pred_depth.squeeze(1).shape[1:]))

        output_cache_values = outputs[1:]

        cache_dict_curr = {key: val for key, val in zip(cache_dict.keys(), output_cache_values)}

        pred_depth = np.squeeze(pred_depth, axis=(0, 1))
        depth_predictions.append(pred_depth)

        # Update cache
        if cache_size == window_size - 1:
            for key in cache_dict:
                cache_dict[key] = np.concatenate([cache_dict[key][:, :, :, 1:cache_size, :], cache_dict_curr[key], np.zeros_like(cache_dict_curr[key])], axis=3)
        else:
            for key in cache_dict:
                cache_dict[key][:, :, :, cache_size, :] = cache_dict_curr[key][:, :, :, 0, :]

        cache_size += 1
        cache_size = min(cache_size, window_size - 1)

        remaining = list(range(cache_size, window_size))
        padding = [window_size - 1] * (cache_size - 1)
        mask_indices = np.array(remaining + padding, dtype=np.int64)

        input_position = np.array([cache_size, cache_size], dtype=np.int64)

    # Stack predictions and normalize
    depth_predictions = np.stack(depth_predictions, axis=0)  # Shape: [T, H, W]
    depth_predictions -= np.min(depth_predictions)
    depth_predictions /= np.max(depth_predictions + 1e-8)

    def make_grid_np(images, nrow):
        """Creates a tiled grid of grayscale images: [T, H, W] -> [H', W']"""
        T, H, W = images.shape
        ncol = (T + nrow - 1) // nrow
        grid = np.zeros((ncol * H, nrow * W), dtype=np.float32)

        for idx, img in enumerate(images):
            row = idx // nrow
            col = idx % nrow
            grid[row * H:(row + 1) * H, col * W:(col + 1) * W] = img

        return grid

    grid = make_grid_np(depth_predictions, nrow=4)

    # Save with colormap
    plt.imsave('./outputs/ONNX_output.png', grid, cmap='Spectral')

if __name__ == '__main__':
    test_video_depth_anything("configs/oVDA_c16.yaml", "assets/example_videos/Cars_and_Gasstation.mp4", 518, device='cuda:0')
    compile_onnx("configs/oVDA_c16.yaml", "model.onnx", 518, 924)
    # test_onnx("model.onnx", "assets/example_videos/Cars_and_Gasstation.mp4", window_size=16, input_size=518)
