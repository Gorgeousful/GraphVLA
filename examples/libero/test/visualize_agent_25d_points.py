"""Overlay structured gripper 2.5D points on a LIBERO video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_all_no_noops_1.0.0_lerobot_10hz"
)
DEFAULT_AGENT_STATE = Path("GraphVLA/examples/libero/test/task31_traj0_agent_state_25d.npz")


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
            f"got trajectory_index={trajectory_index}"
        )
    return matches[trajectory_index]


def episode_video_path(root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def draw_uvd_point(
    frame: np.ndarray,
    uvd: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if not np.isfinite(uvd).all() or uvd[2] <= 1e-6:
        return
    x, y = np.rint(uvd[:2]).astype(int)
    h, w = frame.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def write_video(
    video_path: Path,
    output_path: Path,
    points: dict[str, np.ndarray],
    ffmpeg: str,
    flip_horizontal: bool,
    radius: int,
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

    colors = {
        "root_uvd": (255, 0, 0),
        "center_uvd": (0, 0, 255),
        "left_tip_uvd": (0, 255, 0),
        "right_tip_uvd": (0, 255, 255),
    }
    n_frames = min(len(next(iter(points.values()))), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9)

    frame_count = 0
    while frame_count < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        for name, color in colors.items():
            draw_uvd_point(frame, points[name][frame_count], color, radius)
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
    parser.add_argument("--agent-state", type=Path, default=DEFAULT_AGENT_STATE)
    parser.add_argument("--task-index", type=int, default=31)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--flip-horizontal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GraphVLA/examples/libero/test/task31_traj0_agent_25d_points.mp4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.agent_state)
    points = {
        "root_uvd": data["root_uvd"],
        "center_uvd": data["center_uvd"],
        "left_tip_uvd": data["left_tip_uvd"],
        "right_tip_uvd": data["right_tip_uvd"],
    }

    if "episode_index" in data.files:
        episode_index = int(data["episode_index"])
    else:
        episode_index = int(find_episode(args.dataset_root, args.task_index, args.trajectory_index)["episode_index"])
    video_path = episode_video_path(args.dataset_root, episode_index, args.video_key)

    frame_count = write_video(
        video_path, args.output, points, args.ffmpeg, args.flip_horizontal, args.radius
    )
    print(f"agent_state: {args.agent_state}")
    print(f"episode_index: {episode_index}")
    print(f"video frames written: {frame_count}")
    print("colors: root=blue, center=red, left_tip=green, right_tip=yellow")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
