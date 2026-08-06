#!/usr/bin/env python3
"""Track episode masks with SAM3 and resample points from every visible mask."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.geom_utils import sample_points_from_mask
from src.module.node_segmenter import NodeSegmenter

DEFAULT_DATASET = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_with_depth_7_action"
)
COLORS = [(255, 80, 80), (80, 210, 80), (80, 150, 255), (240, 190, 60)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "examples/libero/test/output/sam3_mask_tracking_ep0",
    )
    return parser.parse_args()


def read_frames(table: pd.DataFrame) -> list[np.ndarray]:
    frames = []
    for item in table["image"]:
        payload = item.get("bytes") if isinstance(item, Mapping) else None
        if payload is None:
            raise ValueError("The parquet image column must contain embedded bytes")
        with Image.open(io.BytesIO(payload)) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
        frames.append(np.ascontiguousarray(np.fliplr(frame)))
    return frames


def load_intrinsic(dataset_dir: Path, task_index: int) -> np.ndarray:
    cameras = json.loads((dataset_dir / "meta/cameras.json").read_text())
    entry = next(row for row in cameras if int(row["task_index"]) == task_index)
    return np.asarray(entry["cameras"]["agentview"]["intrinsic"], dtype=np.float32)


def project_initial_points(row: pd.Series, intrinsic: np.ndarray) -> list[list[list[float]]]:
    prompts = []
    for node_xyz, valid in zip(row["node_points_xyz"], row["valid_node_mask"]):
        if not valid:
            continue
        xyz = np.stack(node_xyz).astype(np.float32)
        positive_z = xyz[:, 2] > 0
        xyz = xyz[positive_z]
        if not len(xyz):
            raise ValueError("A valid node has no positive-depth points in frame 0")
        uv = np.stack(
            [
                intrinsic[0, 0] * xyz[:, 0] / xyz[:, 2] + intrinsic[0, 2],
                intrinsic[1, 1] * xyz[:, 1] / xyz[:, 2] + intrinsic[1, 2],
            ],
            axis=-1,
        )
        prompts.append(uv.tolist())
    if not prompts:
        raise ValueError("Episode frame 0 has no valid object nodes")
    return prompts


def node_labels(dataset_dir: Path, task_index: int, count: int) -> list[str]:
    tasks = [json.loads(line) for line in (dataset_dir / "meta/tasks.jsonl").read_text().splitlines()]
    task_text = next(str(row["task"]) for row in tasks if int(row["task_index"]) == task_index)
    structures = [json.loads(line) for line in (dataset_dir / "meta/taskstructures.jsonl").read_text().splitlines()]
    structure = next(row for row in structures if row["task"] == task_text)
    labels = [
        node["name"]
        for subtask in structure["subtasks"]
        for node in subtask.get("nodes", [])
        if node.get("role") != "actor"
    ]
    labels = (labels + [f"node {index}" for index in range(len(labels), count)])[:count]
    totals = {label: labels.count(label) for label in set(labels)}
    seen: dict[str, int] = {}
    unique = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        unique.append(f"{label} #{seen[label]}" if totals[label] > 1 else label)
    return unique


def render(frame_rgb: np.ndarray, masks: list[np.ndarray], points: list[np.ndarray | None], labels: list[str], frame_index: int) -> np.ndarray:
    image = frame_rgb.copy()
    for index, (mask, sampled) in enumerate(zip(masks, points)):
        color = COLORS[index % len(COLORS)]
        if mask.any():
            overlay = image.copy()
            overlay[mask] = color
            image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
        if sampled is not None:
            for x, y in sampled:
                cv2.circle(image, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
        status = f"{labels[index]}: {int(mask.sum())} px" if mask.any() else f"{labels[index]}: MISSING"
        cv2.putText(image, status, (8, 38 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"SAM3 only | frame {frame_index}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_video_ffmpeg(frames_rgb: list[np.ndarray], save_path: Path, fps: float) -> None:
    if not frames_rgb:
        return
    first = np.asarray(frames_rgb[0], dtype=np.uint8)
    height, width = first.shape[:2]
    proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
            "-pix_fmt", "yuv420p",
            str(save_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    pipe_broken = False
    try:
        for frame in frames_rgb:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"Video frame shape mismatch: expected {(height, width, 3)}, got {frame.shape}"
                )
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pipe_broken = True
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pipe_broken = True

    stderr = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if pipe_broken or proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace") if stderr else "unknown ffmpeg error"
        raise RuntimeError(f"FFmpeg failed while saving {save_path}: {message}")
    print(f"saved video: {save_path} ({len(frames_rgb)} frames, {fps:.1f} fps)")


def main() -> None:
    args = parse_args()
    parquet = args.dataset_dir / "data" / f"chunk-{args.episode // 1000:03d}" / f"episode_{args.episode:06d}.parquet"
    table = pd.read_parquet(parquet)
    frames = read_frames(table)
    task_index = int(table.iloc[0]["task_index"])
    prompts = project_initial_points(table.iloc[0], load_intrinsic(args.dataset_dir, task_index))
    labels = node_labels(args.dataset_dir, task_index, len(prompts))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"episode_{args.episode:06d}.mp4"

    segmenter = NodeSegmenter(device=args.device)
    records = []
    rendered_frames = []
    for frame_index, frame in enumerate(frames):
        masks = segmenter.predict(
            frame,
            points=prompts if frame_index == 0 else None,
            anchor_frame=frame_index == 0,
        )
        if len(masks) != len(labels):
            raise RuntimeError(f"Frame {frame_index}: SAM3 returned {len(masks)} masks for {len(labels)} nodes")
        sampled = [sample_points_from_mask(mask, args.num_points) if mask.any() else None for mask in masks]
        areas = [int(mask.sum()) for mask in masks]
        missing = [labels[index] for index, area in enumerate(areas) if area == 0]
        records.append({"frame": frame_index, "mask_areas": dict(zip(labels, areas)), "missing": missing})
        rendered_frames.append(render(frame, masks, sampled, labels, frame_index))
        if frame_index % 25 == 0 or missing:
            print(f"frame={frame_index:03d} areas={areas} missing={missing}")
    save_video_ffmpeg(rendered_frames, video_path, args.fps)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "episode": args.episode,
        "task_index": task_index,
        "frames": len(frames),
        "labels": labels,
        "num_points": args.num_points,
        "missing_frame_counts": {
            label: sum(label in record["missing"] for record in records) for label in labels
        },
        "records": records,
    }
    metrics_path = args.output_dir / f"episode_{args.episode:06d}.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"video: {video_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
