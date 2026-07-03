import argparse
import os
import json
import subprocess
import shutil
import cv2
import numpy as np
import torch
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map
from tqdm import tqdm


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().squeeze(0)


def save_depth_video(depth, save_path, fps=10):
    depth = np.squeeze(depth, axis=-1) if depth.ndim == 4 and depth.shape[-1] == 1 else depth
    valid = np.isfinite(depth)
    vmin, vmax = np.percentile(depth[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    frames = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    frames = (frames * 255).astype(np.uint8)
    frames_bgr = [cv2.applyColorMap(frame, cv2.COLORMAP_TURBO) for frame in frames]
    height, width = frames_bgr[0].shape[:2]
    ffmpeg_bin = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen(
        [
            ffmpeg_bin, "-y",
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
    for frame in frames_bgr:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="ignore")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to write {save_path}:\n{stderr}")


def save_cameras_json(extrinsic, intrinsic, image_names, save_path):
    cameras = []
    for i in range(extrinsic.shape[0]):
        cameras.append({
            "frame": i,
            "image": image_names[i],
            "extrinsic": extrinsic[i].tolist(),
            "intrinsic": intrinsic[i].tolist(),
        })
    with open(save_path, "w") as f:
        json.dump(cameras, f, indent=2)


def save_predictions_with_fixed_camera(predictions, images_shape_hw, image_names, result_dir):
    os.makedirs(result_dir, exist_ok=True)

    extrinsic_pred, intrinsic_pred = pose_encoding_to_extri_intri(predictions["pose_enc"], images_shape_hw)
    depth = to_numpy(predictions["depth"])
    extrinsic_pred = to_numpy(extrinsic_pred)
    intrinsic_pred = to_numpy(intrinsic_pred)

    fixed_extrinsic = np.repeat(extrinsic_pred[:1], depth.shape[0], axis=0)
    fixed_intrinsic = np.repeat(intrinsic_pred[:1], depth.shape[0], axis=0)
    world_points = unproject_depth_map_to_point_map(depth, fixed_extrinsic, fixed_intrinsic)

    images_rgb = np.moveaxis(to_numpy(predictions["images"]), 1, -1)
    colors_rgb = np.clip(images_rgb * 255.0, 0, 255).astype(np.float32)
    world_points_rgb = np.concatenate([world_points.astype(np.float32), colors_rgb], axis=-1)

    save_depth_video(depth, os.path.join(result_dir, "depth_vis.mp4"))
    np.save(os.path.join(result_dir, "world_points.npy"), world_points_rgb)
    save_cameras_json(fixed_extrinsic, fixed_intrinsic, image_names, os.path.join(result_dir, "cameras.json"))
    save_cameras_json(extrinsic_pred, intrinsic_pred, image_names, os.path.join(result_dir, "cameras_predicted.json"))


def run_stream_test(model, images, image_names, mode, result_dir, window_size=5):
    session = StreamSession(model, mode=mode, window_size=window_size)
    predictions = None
    with torch.no_grad():
        for i in tqdm(range(images.shape[0]), total=images.shape[0], desc=mode):
            image = images[i : i + 1]
            predictions = session.forward_stream(image)
    save_predictions_with_fixed_camera(predictions, images.shape[-2:], image_names, result_dir)
    session.clear()
    print(f"Saved fixed-camera {mode} results to {result_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--mode", choices=["causal", "window", "both"], default="window")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_root = os.path.join(script_dir, "result_fixed_camera")
    example_dir = os.path.join(script_dir, "examples/dynamic_robot")

    model = STream3R.from_pretrained("/data0/luokang/dataset/luokang/ckpts/STream3R").to(device).eval()
    image_names = [os.path.join(example_dir, file) for file in sorted(os.listdir(example_dir))]
    images = load_and_preprocess_images(image_names).to(device)

    if args.mode in ["causal", "both"]:
        run_stream_test(
            model,
            images,
            image_names,
            mode="causal",
            result_dir=os.path.join(result_root, "causal"),
        )
    if args.mode in ["window", "both"]:
        run_stream_test(
            model,
            images,
            image_names,
            mode="window",
            result_dir=os.path.join(result_root, f"window_{args.window_size}"),
            window_size=args.window_size,
        )


if __name__ == "__main__":
    main()
