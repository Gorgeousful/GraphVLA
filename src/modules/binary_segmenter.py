import os
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
torch.set_float32_matmul_precision("high")



class BinarySegmenter:
    """Binary foreground/background segmentation using BiRefNet.

    Usage::

        segmenter = BinarySegmenter()
        fg_mask = segmenter.predict(frame_rgb, mode="foreground")
        bg_mask = segmenter.predict(frame_rgb, mode="background")

    The output mask is boolean with the same height and width as the input image.
    """

    def __init__(
        self,
        model_path="/data0/luokang/dataset/luokang/ckpts/BiRefNet",
        high_thres=0.5,
        low_thres=0.5,
        foreground_erode=1,
        foreground_dilate=1,
        device="cuda:0",
        fp32=False,
    ):
        self.model_path = model_path
        self.high_thres = float(high_thres)
        self.low_thres = float(low_thres)
        if self.low_thres > self.high_thres:
            raise ValueError(f"low_thres must be <= high_thres, got {self.low_thres} > {self.high_thres}")
        self.foreground_erode = int(foreground_erode)
        self.foreground_dilate = int(foreground_dilate)
        if self.foreground_erode < 0 or self.foreground_dilate < 0:
            raise ValueError("foreground_erode and foreground_dilate must be non-negative")
        self.device = device
        self.fp32 = fp32
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        if not fp32:
            self.model.half()

    def _to_pil(self, frame_rgb) -> Image.Image:
        if isinstance(frame_rgb, Image.Image):
            return frame_rgb.convert("RGB")
        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB image shape (H,W,3), got {frame_rgb.shape}")
        return Image.fromarray(frame_rgb.astype(np.uint8, copy=False)).convert("RGB")

    def predict_probability(self, frame_rgb) -> np.ndarray:
        """Return BiRefNet foreground probability map, shape (H, W), float32 in [0, 1]."""
        image = self._to_pil(frame_rgb)
        input_image = self.transform_image(image).unsqueeze(0).to(self.device)
        if not self.fp32:
            input_image = input_image.half()

        with torch.inference_mode():
            pred = self.model(input_image)[-1].sigmoid().detach().cpu()[0].squeeze()

        pred_pil = transforms.ToPILImage()(pred).resize(image.size, getattr(Image, "Resampling", Image).BILINEAR)
        return (np.asarray(pred_pil, dtype=np.float32) / 255.0).astype(np.float32, copy=False)

    @staticmethod
    def _morph(mask: np.ndarray, pixels: int, op: str) -> np.ndarray:
        if pixels <= 0 or not mask.any():
            return mask.astype(bool, copy=False)
        kernel_size = 2 * pixels + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = mask.astype(np.uint8)
        if op == "erode":
            return cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
        if op == "dilate":
            return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
        raise ValueError(f"Unsupported morph op: {op}")

    def predict(self, frame_rgb, mode="foreground"):
        """Predict a foreground or background bool mask.

        Args:
            frame_rgb: np.ndarray RGB image, shape (H, W, 3), uint8; or PIL image.
            mode: "foreground" returns probability >= high_thres, then erodes it by foreground_erode;
                "background" dilates the high-threshold foreground by foreground_dilate,
                inverts it, and optionally keeps only probability <= low_thres.

        Returns:
            np.ndarray: bool mask, shape (H, W).
        """
        if mode not in {"foreground", "background"}:
            raise ValueError(f"mode must be 'foreground' or 'background', got {mode!r}")
        probability = self.predict_probability(frame_rgb)
        foreground = probability >= self.high_thres
        if mode == "foreground":
            return self._morph(foreground, self.foreground_erode, "erode")

        expanded_foreground = self._morph(foreground, self.foreground_dilate, "dilate")
        background = ~expanded_foreground
        if self.low_thres < self.high_thres:
            background &= probability <= self.low_thres
        return background.astype(bool, copy=False)

    def draw_on_image(self, frame_rgb, mask, save_path=None, color=(0, 0, 255), alpha=0.45):
        """Overlay one mask on an RGB image and return BGR visualization."""
        image = np.asarray(frame_rgb).astype(np.uint8, copy=False)
        drawn = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != drawn.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {drawn.shape[:2]}")
        overlay = np.zeros_like(drawn)
        overlay[mask] = color
        drawn = np.where(mask[..., None], (drawn * (1.0 - alpha) + overlay * alpha).astype(np.uint8), drawn)
        if save_path is not None:
            cv2.imwrite(save_path, drawn)
        return drawn


if __name__ == "__main__":
    def open_ffmpeg_writer(save_path, fps, width, height):
        return subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
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


    def close_ffmpeg_writer(proc, save_path):
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed for {save_path}: {stderr}")


    video_path = "/data0/luokang/research/GraphVLA/__test__/data/data.mp4"
    save_dir = "/data0/luokang/research/GraphVLA/__tmp__"
    os.makedirs(save_dir, exist_ok=True)
    foreground_video = os.path.join(save_dir, "binary_segmenter_foreground.mp4")
    background_video = os.path.join(save_dir, "binary_segmenter_background.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    segmenter = BinarySegmenter()
    fg_proc = open_ffmpeg_writer(foreground_video, fps, width, height)
    bg_proc = open_ffmpeg_writer(background_video, fps, width, height)

    processed = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            foreground = segmenter.predict(frame_rgb, mode="foreground")
            background = segmenter.predict(frame_rgb, mode="background")

            foreground_bgr = np.zeros_like(frame_bgr)
            background_bgr = np.zeros_like(frame_bgr)
            foreground_bgr[foreground] = frame_bgr[foreground]
            background_bgr[background] = frame_bgr[background]

            fg_proc.stdin.write(foreground_bgr.tobytes())
            bg_proc.stdin.write(background_bgr.tobytes())

            processed += 1
            if processed % 25 == 0:
                total = frame_count if frame_count else "?"
                print(f"processed {processed}/{total}")
    finally:
        cap.release()
        close_ffmpeg_writer(fg_proc, foreground_video)
        close_ffmpeg_writer(bg_proc, background_video)

    print(f"foreground video: {foreground_video}")
    print(f"background video: {background_video}")
    print(f"processed_frames={processed} fps={fps:.2f} size={width}x{height}")
