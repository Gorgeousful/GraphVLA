"""Overlay projected LIBERO state positions on the source video.

Default behavior:
  - task_index=31
  - the 0-th episode for that task
  - agentview / observation.images.image
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_all_no_noops_1.0.0_lerobot_10hz"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_episode(root: Path, task_index: int, trajectory_index: int) -> dict:
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    task = next(item for item in tasks if item["task_index"] == task_index)

    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    matches = [ep for ep in episodes if task["task"] in ep["tasks"]]
    if trajectory_index >= len(matches):
        raise IndexError(
            f"task_index={task_index} has only {len(matches)} trajectories, "
            f"but trajectory_index={trajectory_index} was requested"
        )
    return matches[trajectory_index]


def episode_path(root: Path, episode_index: int, kind: str, video_key: str) -> Path:
    chunk = episode_index // 1000
    if kind == "data":
        return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if kind == "video":
        return (
            root
            / "videos"
            / f"chunk-{chunk:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
    raise ValueError(f"unknown path kind: {kind}")


def load_states(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["observation.state"])
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)


def load_camera(root: Path, task_index: int, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    cameras = json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))
    record = next(item for item in cameras if item["task_index"] == task_index)
    camera = record["cameras"][camera_name]
    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
    return intrinsic, extrinsic


def project_world_points(points_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_camera = (np.linalg.inv(extrinsic) @ points_h.T).T[:, :3]
    pixels_h = (intrinsic @ points_camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, points_camera


def draw_point(frame: np.ndarray, pixel: np.ndarray, valid: bool) -> np.ndarray:
    if not valid:
        return frame
    x, y = np.rint(pixel).astype(int)
    h, w = frame.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return frame


def write_overlay_video(
    video_path: Path,
    output_path: Path,
    pixels: np.ndarray,
    points_camera: np.ndarray,
    ffmpeg: str,
    flip_horizontal: bool,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            "26",
            "-g",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_count = 0
    while frame_count < len(pixels):
        ok, frame = cap.read()
        if not ok:
            break
        valid = np.isfinite(pixels[frame_count]).all() and points_camera[frame_count, 2] > 1e-6
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        frame = draw_point(frame, pixels[frame_count], valid)
        proc.stdin.write(frame.tobytes())
        frame_count += 1

    cap.release()
    proc.stdin.close()
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task-index", type=int, default=31)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--flip-horizontal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GraphVLA/examples/libero/test/task31_traj0_state_projection.mp4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = find_episode(args.dataset_root, args.task_index, args.trajectory_index)
    episode_index = episode["episode_index"]

    parquet_path = episode_path(args.dataset_root, episode_index, "data", args.video_key)
    video_path = episode_path(args.dataset_root, episode_index, "video", args.video_key)

    states = load_states(parquet_path)
    state_points_world = states[:, :3]
    intrinsic, extrinsic = load_camera(args.dataset_root, args.task_index, args.camera_name)
    pixels, points_camera = project_world_points(state_points_world, intrinsic, extrinsic)

    frame_count = write_overlay_video(
        video_path, args.output, pixels, points_camera, args.ffmpeg, args.flip_horizontal
    )
    print(f"task_index: {args.task_index}")
    print(f"trajectory_index: {args.trajectory_index}")
    print(f"episode_index: {episode_index}")
    print(f"states shape: {states.shape}")
    print(f"video frames written: {frame_count}")
    print(f"flip horizontal: {args.flip_horizontal}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
