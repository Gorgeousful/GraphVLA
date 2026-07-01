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

# Point to CUDA Toolkit inside your conda environment
cuda_path = os.environ.get("CONDA_PREFIX")  # This is where conda installed CUDA

# Set environment variables
os.environ["CUDA_HOME"] = cuda_path
os.environ["CPATH"] = f"{cuda_path}/include"
os.environ["LIBRARY_PATH"] = f"{cuda_path}/lib:{cuda_path}/lib64"
os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib:{cuda_path}/lib64"

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This automatically initializes CUDA driver
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.loading_utils import load_frames_as_numpy, load_video_as_numpy
from src.models.utils.preprocessing import VideoPreprocessor

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, trt_file_path, use_fp16=False):
    '''
    Build TensorRT engine from OnnX model
    --------------------------------------
    Builds TensorRT engine from OnnX model in given prcision

    Parameters
    ----------
    onnx_file_path : str
        Path to the OnnX model file
    trt_file_path : str
        Path to to save the builded engine to 
    use_fp16 : bool, default False
        Whether to use FP16 precision or not

    Returns
    -------
    None
    
    '''
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 34)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed.")

    with open(trt_file_path, "wb") as f:
        f.write(engine)


def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())
    
def allocate_buffers(engine):
    inputs = {}
    outputs = {}
    stream = cuda.Stream()

    for name in engine:
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        tensor_info = {
            'host': host_mem,
            'device': device_mem,
            'shape': shape,
            'dtype': dtype
        }

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs[name] = tensor_info
        else:
            outputs[name] = tensor_info

    return inputs, outputs, stream

def prepare_static_inference(trt_path):
    '''
    Sets up TensorRT engine
    -----------------------
    Initialises prebuild TensorRT engine and prepares cuda stream. 

    Parameters
    ----------
    trt_path : str
        Path to the TensorRT engine file

    Returns
    -------
    engine : TensorRT engine
        TensorRT engine
    context : TensorRT execution context
        TensorRT execution context
    inputs : input_buffer
        Dictionary of input tensors
    outputs : output_buffer
        Dictionary of output tensors
    stream : pycuda.driver.Stream
        PyCUDA stream
    '''
    engine = load_engine(trt_path)
    inputs, outputs, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    # Set static tensor addresses once
    for name, info in inputs.items():
        context.set_tensor_address(name, int(info['device']))
    for name, info in outputs.items():
        context.set_tensor_address(name, int(info['device']))

    return engine, context, inputs, outputs, stream


def infer_static(context, inputs, outputs, stream, input_data):
    '''
    TensorRT Forward
    ----------------
    Handels forward of the TensorRT engine and cuda stream. 

    Parameters
    ----------
    context : trt.Runtime.execution_context
        TensorRT execution context
    inputs : dict
        Dictionary of input tensors
    outputs : dict
        Dictionary of output tensors
    stream : pycuda.driver.Stream
        PyCUDA stream
    input_data : dict
        Dictionary of input data

    Returns
    -------
    dict
        Dictionary of output data 
    '''
    # Copy input to device
    for name, info in inputs.items():
        np.copyto(info['host'], input_data[name].ravel())
        cuda.memcpy_htod_async(info['device'], info['host'], stream)

    # Run inference
    context.execute_async_v3(stream.handle)

    # Copy output to host
    for name, info in outputs.items():
        cuda.memcpy_dtoh_async(info['host'], info['device'], stream)

    stream.synchronize()

    return {name: np.array(info['host']).reshape(info['shape']) for name, info in outputs.items()}

def test_tensorrt(tensorrt_path: str, frame_path: str, window_size: int = 16, input_size=518) -> None:
    '''
    Test TensorRT model
    -------------------
    Tests TensorRT model build for the current hardware. Only for checking if build was successful. Not meant for measuring time. 

    WARNING: This code is for testing only. It is not efficient nor is it optimized. Only to test if build was successful and model runs as expected. 

    Parameters
    ----------
    tensorrt_path : str
        Path to the TensorRT engine file
    frame_path : str
        Path to the input frames
    window_size : int, default 16
        Size of the cache, by default 16, set acording to loaded model
    input_size : int, default 518
        Input size of the model, set acording to build model. (Because the build is not flexible, as soon as model hase been build only the 
        build resolution will work)
    
    Returns
    -------
    None
    '''
    engine, context, inputs, outputs, stream = prepare_static_inference(tensorrt_path)

    # Load frames
    if frame_path.endswith('.mp4'):
        input_d = load_video_as_numpy(frame_path)
    else:
        input_d = load_frames_as_numpy(frame_path)
    t, h, w, c = input_d.shape

    frame_height, frame_width = input_d[0].shape[:2]
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    if ratio > 1.78:  # VDA recommendet to process video with ratio smaller than 16:9 due to memory limitation
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    
    pre = VideoPreprocessor(input_size=input_size, device='cpu', ensure_multiple_of=14, keep_aspect_ratio=True, resize_method='lower_bound')
    prepared_frames = pre.preprocess(input_d)

    b, t, c, h, w = prepared_frames.size()
    print(f'Inferring Video Depth at res: {h}x{w}')

    _, cache_dict, _, _ = load_dummy_data(h=h, w=w, format="numpy", cache_size=window_size)

    cache_size = 0
    mask_indices = np.arange(1, window_size, dtype=np.int64)
    input_position = np.array([0, 1], dtype=np.int64)

    times = []
    depth_predictions = []
    for i in tqdm(range(prepared_frames.shape[1])):
        input_frame = prepared_frames[:, i, :, :, :].unsqueeze(dim=1).numpy()  # Shape: (1, 1, C, H, W)

        input_feed = {"input_frame": input_frame, "mask_indices": mask_indices, "input_position": input_position}
        for k, v in cache_dict.items():
            input_feed[f"cache_dict.{k}"] = v

        trt_outputs = infer_static(context, inputs, outputs, stream, input_feed)
        depth_pred = trt_outputs["pred_depth"]

        output_cache_values = {}
        for k in cache_dict.keys():
            output_cache_values[k] = trt_outputs[f"output_cache.{k}"]

        cache_dict_curr = {key: val for key, val in zip(cache_dict.keys(), output_cache_values.values())}

        depth_pred = np.squeeze(depth_pred, axis=(0, 1))
        depth_predictions.append(depth_pred)

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

    # Create a grid (like torchvision make_grid)
    def make_grid_np(images, nrow):
        """
        Creates a tiled grid of grayscale images: [T, H, W] -> [H', W']
        Only for easy visualization
        """
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
    plt.imsave('./outputs/TensorRT_output.png', grid, cmap='Spectral')


if __name__ == "__main__":
    onnx_path = "model.onnx"
    trt_path = "model.trt"

    #Build engine (only needs to be done once)
    #build_engine(onnx_path, trt_path, use_fp16=True)

    test_tensorrt("model.trt", "assets/example_videos/Cars_and_Gasstation.mp4", window_size=16, input_size=518)