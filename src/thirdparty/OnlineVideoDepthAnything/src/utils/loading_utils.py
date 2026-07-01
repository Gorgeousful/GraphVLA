import cv2
cv2.setNumThreads(1)
import numpy as np
import glob
import os
from natsort import natsorted
import imageio.v3 as iio
import tifffile
import matplotlib.cm as cm

from src.models.utils.preprocessing import VideoPreprocessor

def save_side_by_side(original, pred_color, out_path, fps=20):
    """
    original: [T,H,W,3] float32
    pred_color: [T,H,W,3] uint8
    """
    h, w = original.shape[1], original.shape[2]
    combined_w = w * 2

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (combined_w, h)
    )

    for orig, pred in zip(original, pred_color):
        orig_uint8 = (orig * 255).astype(np.uint8)[..., ::-1]  # RGB → BGR
        combined = np.concatenate([orig_uint8, pred], axis=1)
        writer.write(combined)

    writer.release()


def colorize_pred(pred):
    """Pred [T,H,W] oder [T,H,W,1] → färbiges uint8 Video."""
    import matplotlib.cm as cm

    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    vmin = float(pred.min())
    vmax = float(pred.max())
    cmap = cm.get_cmap("Spectral")

    frames = []
    for frame in pred:
        norm = (np.clip(frame, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
        rgb = cmap(norm)[..., :3] * 255
        rgb = rgb.astype(np.uint8)
        bgr = rgb[..., ::-1]  # RGB → BGR
        frames.append(bgr)

    return np.stack(frames)

def load_video_as_numpy(path):
    """
    Loads Video
    -----------------
    Loads a video as a numpy array of type float 32 and in range 0 to 1

    Parameters
    ----------
    path : str
        Path to the video you want to load

    Returns
    -------
    result : np.ndarray
        Numpy array of shape [Time, Height, Width, Channels]
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.stack(frames)

def load_frames_as_numpy(image_list: str):
    """
    Loads Frames from a Folder
    -----------------
    Loads a bunch of frames as a numpy array of type float 32 and in range 0 to 1
    Supported Formats: jpg, jpeg, png, webp, bmp, tiff

    Parameters
    ----------
    image_list : str
        Path to the folder containing the frames you want to load

    Returns
    -------
    result : np.ndarray
        Numpy array of shape [Time, Height, Width, Channels]
    """
    if isinstance(image_list, str):
        img_str = image_list
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tiff']
        image_list = []
        for ext in image_extensions:
            image_list.extend(glob.glob(os.path.join(img_str, ext)))
    
    image_list = natsorted(image_list)

    # Load first frame to get ref values
    first_frame = iio.imread(image_list)
    h, w, c = first_frame.shape

    frames = np.zeros((len(image_list), h, w, c), dtype=np.float32)

    for i, image_path in enumerate(image_list):
        image = iio.imread(image_path)
        frames[i] = image.astype(np.float32) / 255.0

    return frames

def save_video_mp4(video: np.ndarray, path: str, fps=20):
    vmin = float(video.min())
    vmax = float(video.max())

    h, w = video.shape[1], video.shape[2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    colormap = cm.get_cmap('Spectral')

    for frame in video:
        # Normalize globally
        norm = (np.clip(frame, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)

        # Apply matplotlib colormap → RGBA in [0..1]
        colored = colormap(norm)[..., :3]   # drop alpha channel

        # Convert to uint8 + BGR for OpenCV
        colored_bgr = (colored * 255).astype(np.uint8)[..., ::-1]

        writer.write(colored_bgr)

    writer.release()

def save_predictions_tiff(pred: np.ndarray, path: str):
    tifffile.imwrite(path, pred.astype(np.float32))
