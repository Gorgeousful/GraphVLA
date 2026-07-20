from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data0/luokang/dataset/luokang/lerobot/libero/libero_31_no_noops_1.0.0_lerobot_10hz"),
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--local-episode-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data0/luokang/research/GraphVLA/__tmp__/check"),
    )
    parser.add_argument("--video-key", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_video_key(info: dict, requested: str | None) -> str:
    if requested:
        return requested
    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    for key in ("observation.images.rgb_static", "observation.images.image"):
        if key in video_keys:
            return key
    for key in video_keys:
        lowered = key.lower()
        if "wrist" not in lowered and "gripper" not in lowered:
            return key
    if not video_keys:
        raise RuntimeError("No video feature found in meta/info.json")
    return video_keys[0]


def scan_task_episodes(dataset_dir: Path, info: dict, task_index: int) -> list[int]:
    chunk_size = int(info.get("chunks_size", 1000))
    episodes = []
    for parquet_path in sorted((dataset_dir / "data").glob("chunk-*/*.parquet")):
        meta = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        if int(meta["task_index"].iloc[0]) == int(task_index):
            episodes.append(int(meta["episode_index"].iloc[0]))
    episodes.sort()
    return episodes


def episode_parquet_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_dir / info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def episode_video_path(dataset_dir: Path, info: dict, video_key: str, episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_dir / info["video_path"].format(
        episode_chunk=episode_index // chunk_size,
        video_key=video_key,
        episode_index=episode_index,
    )


def is_libero_dataset(dataset_dir: Path) -> bool:
    return "libero" in dataset_dir.name.lower()


def nested_node(row) -> np.ndarray:
    return np.array([[np.asarray(point, dtype=np.float32) for point in node] for node in row], dtype=np.float32)


def as2d(value, dtype=None) -> np.ndarray:
    array = np.asarray(value.tolist() if hasattr(value, "tolist") else value)
    return array.astype(dtype) if dtype is not None else array


def add_panel_title(panel: np.ndarray, text: str) -> np.ndarray:
    height, width = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def add_header(canvas: np.ndarray, text: str, header_h: int) -> None:
    width = canvas.shape[1]
    cv2.rectangle(canvas, (0, 0), (width, header_h), (0, 0, 0), -1)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    x = (width - text_w) // 2
    y = (header_h + text_h) // 2 - 2
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def draw_node_panel(frame: np.ndarray, df: pd.DataFrame, frame_index: int) -> np.ndarray:
    colors = [(60, 60, 255), (255, 144, 30), (50, 205, 50), (0, 215, 255), (255, 0, 255), (255, 255, 0)]
    panel = frame.copy()
    tracks = nested_node(df["node_points_track"].iloc[frame_index])
    if "node_points_mask" in df.columns:
        node_points_mask = np.asarray(df["node_points_mask"].iloc[frame_index], dtype=bool)
    else:
        node_points_mask = np.ones(tracks.shape[0], dtype=bool)

    active = np.zeros(tracks.shape[0], dtype=bool)
    active[:len(node_points_mask)] = node_points_mask[:tracks.shape[0]]
    draw_order = list(np.where(~active)[0]) + list(np.where(active)[0])
    for node_index in draw_order:
        node = tracks[node_index]
        color = colors[node_index % len(colors)] if active[node_index] else (145, 145, 145)
        valid = node[:, 2] > 0.5
        for x, y, visible in node:
            if visible > 0.5:
                cv2.circle(panel, (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
        if valid.any():
            cx, cy = np.nanmean(node[valid, :2], axis=0)
            cv2.circle(panel, (int(round(cx)), int(round(cy))), 5, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(panel, f"N{node_index}", (int(cx) + 6, int(cy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return add_panel_title(panel, "node_points_track")


def draw_depth_panel(df: pd.DataFrame, frame_index: int) -> np.ndarray:
    depth = as2d(df["depths_rel"].iloc[frame_index], np.float32)
    finite = np.isfinite(depth)
    if finite.any():
        low, high = np.nanpercentile(depth[finite], [2, 98])
        normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    else:
        normalized = np.zeros_like(depth)
    panel = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return add_panel_title(panel, "depths_rel")


def draw_far_background_panel(frame: np.ndarray, df: pd.DataFrame, frame_index: int) -> np.ndarray:
    mask = as2d(df["far_background_mask"].iloc[frame_index], bool)
    overlay = np.zeros_like(frame)
    overlay[mask] = (255, 80, 0)
    panel = np.where(mask[..., None], (frame * 0.55 + overlay * 0.45).astype(np.uint8), frame)
    return add_panel_title(panel, "far_background_mask")


def draw_gripper_panel(frame: np.ndarray, df: pd.DataFrame, frame_index: int) -> np.ndarray:
    height, width = frame.shape[:2]
    panel = frame.copy()
    root, left_base, right_base, left_tip, right_tip, tcp = np.asarray(
        df["gripper_uvd"].iloc[frame_index], dtype=np.float64,
    )

    for name, p, color in (
        ("root", root, (255, 0, 0)),
        ("left_base", left_base, (0, 255, 0)),
        ("right_base", right_base, (0, 255, 255)),
        ("left_tip", left_tip, (255, 0, 255)),
        ("right_tip", right_tip, (255, 255, 0)),
        ("tcp", tcp, (0, 0, 255)),
    ):
        x, y = int(round(p[0])), int(round(p[1]))
        if -50 <= x < width + 50 and -50 <= y < height + 50:
            cv2.circle(panel, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(panel, (x, y), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(panel, name, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    for left, right in ((left_base, right_base), (left_tip, right_tip)):
        cv2.line(
            panel,
            tuple(np.round(left[:2]).astype(int)),
            tuple(np.round(right[:2]).astype(int)),
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    if "gripper_openness" in df.columns:
        openness = float(np.asarray(df["gripper_openness"].iloc[frame_index]).reshape(-1)[0])
        title = f"gripper_uvd  open={openness:.2f}"
    else:
        title = "gripper_uvd  open=N/A"
    return add_panel_title(panel, title)


def render_check_video(dataset_dir: Path, task_index: int, local_episode_index: int, output_dir: Path, video_key: str | None) -> Path:
    info = load_json(dataset_dir / "meta" / "info.json")
    video_key = infer_video_key(info, video_key)
    episodes = scan_task_episodes(dataset_dir, info, task_index)
    if not episodes:
        raise KeyError(f"No episodes found for task_index={task_index}")
    if local_episode_index < 0 or local_episode_index >= len(episodes):
        raise IndexError(f"local_episode_index={local_episode_index} out of range [0, {len(episodes)})")

    episode_index = episodes[local_episode_index]
    parquet_path = episode_parquet_path(dataset_dir, info, episode_index)
    video_path = episode_video_path(dataset_dir, info, video_key, episode_index)
    df = pd.read_parquet(parquet_path)

    cap = cv2.VideoCapture(os.fspath(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Empty video: {video_path}")
    if is_libero_dataset(dataset_dir):
        first = cv2.flip(first, 1)
    height, width = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    header_h = 34
    canvas_width = width * 2
    canvas_height = height * 2 + header_h
    fps = float(info.get("fps", 10) or 10)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"task{task_index:03d}-episode_{episode_index:06d}.mp4"

    proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{canvas_width}x{canvas_height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "24", "-g", "2",
            "-pix_fmt", "yuv420p",
            os.fspath(save_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_index = 0
    try:
        while frame_index < len(df):
            ok, frame = cap.read()
            if not ok:
                break
            if is_libero_dataset(dataset_dir):
                frame = cv2.flip(frame, 1)
            grid = np.vstack([
                np.hstack([draw_node_panel(frame, df, frame_index), draw_depth_panel(df, frame_index)]),
                np.hstack([draw_far_background_panel(frame, df, frame_index), draw_gripper_panel(frame, df, frame_index)]),
            ])
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            canvas[header_h:] = grid
            header = f"subtask_id={int(df['subtask_id'].iloc[frame_index])}  complete={bool(df['is_complete'].iloc[frame_index])}"
            add_header(canvas, header, header_h)
            proc.stdin.write(canvas.tobytes())
            frame_index += 1
        proc.stdin.close()
        proc.wait()
    finally:
        cap.release()
        if proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed for {save_path}: {stderr}")
    print(f"saved: {save_path}")
    print(f"frames: {frame_index}")
    return save_path


def main() -> None:
    args = parse_args()
    render_check_video(
        dataset_dir=args.dataset_dir,
        task_index=args.task_index,
        local_episode_index=args.local_episode_index,
        output_dir=args.output_dir,
        video_key=args.video_key,
    )


if __name__ == "__main__":
    main()
