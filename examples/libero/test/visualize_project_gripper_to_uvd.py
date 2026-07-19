"""Visualize GeomFrankaPanda.project_gripper_to_uvd on a LIBERO episode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.libero.embodiment.robot import GeomFrankaPanda  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_31_no_noops_1.0.0_lerobot_10hz"
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
            f"got trajectory_index={trajectory_index}"
        )
    return matches[trajectory_index]


def episode_path(root: Path, episode_index: int, kind: str, video_key: str) -> Path:
    chunk = episode_index // 1000
    if kind == "data":
        return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def load_camera(root: Path, task_index: int, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    cameras = json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))
    record = next(item for item in cameras if item["task_index"] == task_index)
    camera = record["cameras"][camera_name]
    return np.asarray(camera["intrinsic"], dtype=np.float64), np.asarray(camera["extrinsic"], dtype=np.float64)


def load_states(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["observation.state"])
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)


def draw_point(frame: np.ndarray, uvd: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    if not np.isfinite(uvd).all() or uvd[2] <= 1e-6:
        return
    x, y = np.rint(uvd[:2]).astype(int)
    h, w = frame.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def draw_pose_triangle(frame: np.ndarray, uvd: dict[str, np.ndarray]) -> None:
    points = [uvd[name] for name in ("root_uvd", "left_base_uvd", "right_base_uvd")]
    if any(not np.isfinite(point).all() or point[2] <= 1e-6 for point in points):
        return
    pixels = np.rint(np.stack(points)[:, :2]).astype(np.int32)
    cv2.polylines(frame, [pixels], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def write_video(
    video_path: Path,
    output_path: Path,
    states: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    geometry: GeomFrankaPanda,
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
        "left_base_uvd": (0, 255, 0),
        "right_base_uvd": (0, 255, 255),
    }

    frame_count = 0
    while frame_count < len(states):
        ok, frame = cap.read()
        if not ok:
            break
        state = states[frame_count]
        uvd = geometry.project_gripper_to_uvd(
            tcp_state=state[:6],
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        draw_pose_triangle(frame, uvd)
        for name, color in colors.items():
            draw_point(frame, uvd[name], color, radius)
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
    # The single-task libero_31 dataset remaps its source task index to 0.
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--ffmpeg", default=imageio_ffmpeg.get_ffmpeg_exe())
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--flip-horizontal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("task31_traj0_fixed_gripper_keypoints.mp4"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = find_episode(args.dataset_root, args.task_index, args.trajectory_index)
    episode_index = int(episode["episode_index"])
    data_path = episode_path(args.dataset_root, episode_index, "data", args.video_key)
    video_path = episode_path(args.dataset_root, episode_index, "video", args.video_key)
    states = load_states(data_path)
    intrinsic, extrinsic = load_camera(args.dataset_root, args.task_index, args.camera_name)
    geometry = GeomFrankaPanda()

    frame_count = write_video(
        video_path, args.output, states, intrinsic, extrinsic,
        geometry, args.ffmpeg, args.flip_horizontal, args.radius,
    )
    print(f"task_index: {args.task_index}")
    print(f"trajectory_index: {args.trajectory_index}")
    print(f"episode_index: {episode_index}")
    print(f"video frames written: {frame_count}")
    print("colors: root=blue, left-base=green, right-base=yellow")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
