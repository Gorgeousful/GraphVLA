import inspect
import json
import glob as glob_mod
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import yaml
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from safetensors.torch import load_file
from oVDA.models import onlineVideoDepthAnything
from oVDA.preprocessing import VideoPreprocessor
from rich.console import Console
cs = Console()


class DepthPredictor: # oVDA
    """Online video depth prediction using oVDA.

    Usage::

        predictor = DepthPredictor()
        for frame in frames:
            depth = predictor.predict(frame, anchor_frame=(i == 0))
    """

    def __init__(
        self,
        model_path="/data0/luokang/dataset/luokang/ckpts/oVDA/oVDA_c16.pth",
        config_path=None,
        device="cuda:0",
        input_size=518,
        fp32=False,
        preprocess_device="cpu",
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir, os.pardir,
                "thirdparty", "OnlineVideoDepthAnything", "configs", "oVDA_c16.yaml",
            )

        self.device = device
        self.fp32 = fp32
        self.input_size = input_size
        self.preprocess_device = preprocess_device

        # build model
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        net_cfg = {
            k: v for k, v in cfg["net"].items()
            if k in inspect.signature(onlineVideoDepthAnything.__init__).parameters and k != "self"
        }
        self.model = onlineVideoDepthAnything(**net_cfg)

        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        self.model.load_state_dict(ckpt)
        self.model.to(device).eval()

        self.use_half = (not fp32) and str(device).startswith("cuda")
        if not str(device).startswith("cuda"):
            self.model.float()

        # state — populated by reset()
        self.state = None

    def reset(self, frame_rgb):
        """Initialize cache from the first frame."""
        h, w = frame_rgb.shape[:2]
        ratio = max(h, w) / min(h, w)
        input_size = self.input_size
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        preprocessor = VideoPreprocessor(
            input_size=input_size,
            device=self.preprocess_device,
            ensure_multiple_of=14,
            keep_aspect_ratio=True,
            resize_method="lower_bound",
        )

        prepared = preprocessor.preprocess(np.expand_dims(frame_rgb, 0))
        _, _, _, proc_h, proc_w = prepared.size()

        input_cache = self.model.setup_cache(proc_h, proc_w, self.device)
        if self.use_half:
            self.model.half()
            for key in input_cache:
                input_cache[key] = input_cache[key].half()

        self.state = {
            "preprocessor": preprocessor,
            "input_cache": input_cache,
            "cache_size": 0,
            "mask_indices": torch.tensor(
                list(range(1, self.model.cache_size)), device=self.device
            ),
            "input_position": torch.tensor([0, 1], device=self.device),
            "output_size": (h, w),
        }

    def update(self, output_cache):
        """Advance the sliding-window cache after one frame inference."""
        st = self.state
        cache_size = st["cache_size"]
        input_cache = st["input_cache"]

        if cache_size == self.model.cache_size - 1:
            for key in input_cache:
                input_cache[key] = torch.cat(
                    [
                        input_cache[key][:, :, :, 1:cache_size, :],
                        output_cache[key],
                        torch.zeros_like(output_cache[key]),
                    ],
                    dim=3,
                )
        else:
            for key in input_cache:
                input_cache[key][:, :, :, cache_size, :] = output_cache[key][:, :, :, 0, :]

        cache_size = min(cache_size + 1, self.model.cache_size - 1)
        remaining = list(range(cache_size, self.model.cache_size))
        padding = [self.model.cache_size - 1] * (cache_size - 1)

        st["input_cache"] = input_cache
        st["cache_size"] = cache_size
        st["mask_indices"] = torch.tensor(remaining + padding, device=self.device)
        st["input_position"] = torch.tensor(
            [cache_size, cache_size], device=self.device
        )

    def predict(self, frame_rgb, anchor_frame=True):
        """Predict depth for a single RGB frame.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), uint8, RGB order.
            anchor_frame: If True, reset internal cache (treat as first frame).
                          If False, continue from previous cache state.

        Returns:
            np.ndarray, shape (H, W), float32, depth map.
        """
        if anchor_frame:
            self.reset(frame_rgb)

        st = self.state
        frame_input = np.expand_dims(frame_rgb.astype(np.float32) / 255.0, 0)
        prepared = st["preprocessor"].preprocess(frame_input).to(self.device)
        if self.use_half:
            prepared = prepared.half()

        depth_pred, output_cache = self.model.forward(
            prepared,
            input_cache=st["input_cache"],
            mask_indices=st["mask_indices"],
            input_position=st["input_position"],
        )

        self.update(output_cache)

        depth_pred = depth_pred.squeeze(1).unflatten(0, (1, 1)).float()
        if depth_pred.shape[-2:] != st["output_size"]:
            depth_pred = F.interpolate(
                depth_pred, size=st["output_size"],
                mode="bilinear", align_corners=True,
            )
        return depth_pred.squeeze().cpu().numpy().astype(np.float32)

    def draw_on_image(self, depths, save_path=None):
        depth_stack = np.stack(depths, axis=0)
        vmin = float(np.nanpercentile(depth_stack, 1))
        vmax = float(np.nanpercentile(depth_stack, 99))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        frames_bgr = []
        for depth in depth_stack:
            norm = (np.clip(depth, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
            depth_u8 = (norm * 255).astype(np.uint8)
            frames_bgr.append(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO))

        if save_path is not None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            h, w = frames_bgr[0].shape[:2]
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{w}x{h}",
                    "-r", str(fps),
                    "-i", "-",
                    "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
                    "-pix_fmt", "yuv420p",
                    save_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            for frame in frames_bgr:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
            cs.print(f"Vis video: {save_path} ({len(depth_stack)} frames, {fps:.1f} fps)")  
        
        return frames_bgr



def OnlineTest():
    video_path = (
        "/data0/luokang/research/GraphVLA/__test__/v1/data/"
        "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding"
        "_to_the_right_of_the_plate_ep0_image.mp4"
    )
    predictor = DepthPredictor()

    depths = []
    cap = cv2.VideoCapture(video_path)
    interval = 1
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i%interval != 0:
            i += 1
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth = predictor.predict(frame_rgb, anchor_frame=(len(depths) == 0))
        depths.append(depth)
        i += 1
        cs.print(f"frame {i}: shape={depth.shape}, min={depth.min():.3f}, max={depth.max():.3f}")
    cap.release()
    predictor.draw_on_image(depths, save_path="/data0/luokang/research/GraphVLA/__tmp__/depth_prediction_vis.mp4")


if __name__ == "__main__":
    OnlineTest()
