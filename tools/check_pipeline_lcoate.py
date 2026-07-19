from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


COLORS = [
    (60, 60, 255),
    (255, 144, 30),
    (50, 205, 50),
    (0, 215, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw frame-0 node_points_track overlays in 4x4 episode grids.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/data0/luokang/dataset/luokang/lerobot/libero/"
            "libero_31_no_noops_1.0.0_lerobot_10hz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data0/luokang/research/GraphVLA/__tmp__/check_pipeline_lcoate"),
    )
    parser.add_argument("--video-key", default=None)
    return parser.parse_args()


def infer_video_key(info: dict, requested: str | None) -> str:
    if requested:
        return requested
    video_keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
    for key in ("observation.images.rgb_static", "observation.images.image"):
        if key in video_keys:
            return key
    for key in video_keys:
        if "wrist" not in key.lower() and "gripper" not in key.lower():
            return key
    if not video_keys:
        raise RuntimeError("No video feature found in meta/info.json")
    return video_keys[0]


def episode_video_path(dataset_dir: Path, info: dict, video_key: str, episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_dir / info["video_path"].format(
        episode_chunk=episode_index // chunk_size,
        video_key=video_key,
        episode_index=episode_index,
    )


def read_first_frame(video_path: Path, flip_horizontal: bool) -> np.ndarray:
    cap = cv2.VideoCapture(os.fspath(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Empty video: {video_path}")
    finally:
        cap.release()
    return cv2.flip(frame, 1) if flip_horizontal else frame


def draw_episode_panel(frame: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, int]:
    episode_index = int(df["episode_index"].iloc[0])
    task_index = int(df["task_index"].iloc[0])
    tracks = np.stack([
        np.stack(node).astype(np.float32)
        for node in df["node_points_track"].iloc[0]
    ])

    panel = frame.copy()
    valid_nodes = 0
    for node_index, node in enumerate(tracks):
        visible = node[:, 2] > 0.5
        if not visible.any():
            continue
        valid_nodes += 1
        color = COLORS[node_index % len(COLORS)]
        for x, y in node[visible, :2]:
            cv2.circle(
                panel,
                (int(round(x)), int(round(y))),
                2,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
        center = node[visible, :2].mean(axis=0)
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.circle(panel, (cx, cy), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            panel,
            f"N{node_index}",
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"ep={episode_index:06d} task={task_index} nodes={valid_nodes}",
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel, episode_index


def render_grids(dataset_dir: Path, output_dir: Path, video_key: str | None = None) -> list[Path]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    video_key = infer_video_key(info, video_key)
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No episode parquet files found under {dataset_dir / 'data'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    flip_horizontal = "libero" in dataset_dir.name.lower()
    saved_paths = []
    for page_start in range(0, len(parquet_paths), 16):
        panels = []
        episode_indices = []
        for parquet_path in parquet_paths[page_start:page_start + 16]:
            df = pd.read_parquet(
                parquet_path,
                columns=["episode_index", "task_index", "node_points_track"],
            )
            episode_index = int(df["episode_index"].iloc[0])
            frame = read_first_frame(
                episode_video_path(dataset_dir, info, video_key, episode_index),
                flip_horizontal,
            )
            panel, episode_index = draw_episode_panel(frame, df)
            panels.append(panel)
            episode_indices.append(episode_index)

        blank = np.zeros_like(panels[0])
        panels.extend(blank.copy() for _ in range(16 - len(panels)))
        grid = np.vstack([
            np.hstack(panels[row * 4:(row + 1) * 4])
            for row in range(4)
        ])
        save_path = output_dir / (
            f"episodes_{episode_indices[0]:06d}-{episode_indices[-1]:06d}.png"
        )
        if not cv2.imwrite(os.fspath(save_path), grid):
            raise RuntimeError(f"Failed to save grid: {save_path}")
        print(f"saved: {save_path}")
        saved_paths.append(save_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    render_grids(args.dataset_dir.resolve(), args.output_dir.resolve(), args.video_key)


if __name__ == "__main__":
    main()
