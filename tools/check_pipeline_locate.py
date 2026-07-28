from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from src.common.schema import NodeRole, json_to_taskstructure


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
        description="Visualize stored frame-0 XYZ or rerun first-frame locator plus SAM3.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/data0/luokang/research/GraphVLA/examples/libero/extra/libero_with_depth_7"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data0/luokang/research/GraphVLA/__tmp__/check_pipeline_locate"),
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Run LocateAnything localization and SAM3 segmentation on each episode's first frame.",
    )
    return parser.parse_args()


def load_task_nodes(dataset_dir: Path) -> dict[int, list[str]]:
    task_indices = {}
    for line in (dataset_dir / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            task_indices[str(row["task"])] = int(row["task_index"])

    nodes_by_task = {}
    for line in (dataset_dir / "meta" / "taskstructures.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        structure = json_to_taskstructure(json.loads(line))
        if structure.task not in task_indices:
            continue
        nodes_by_task[task_indices[structure.task]] = [
            node.name
            for subtask in structure.subtask_list
            for node in (subtask.node_list or [])
            if node.role != NodeRole.ACTOR
        ]
    return nodes_by_task


def load_camera_intrinsics(dataset_dir: Path) -> dict[int, np.ndarray]:
    rows = json.loads((dataset_dir / "meta" / "cameras.json").read_text(encoding="utf-8"))
    return {
        int(row["task_index"]): np.asarray(
            row["cameras"]["agentview"]["intrinsic"], dtype=np.float32
        )
        for row in rows
    }


def read_first_frame(
    parquet_path: Path,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    row = pd.read_parquet(
        parquet_path,
        columns=["image", "episode_index", "task_index", "node_points_xyz", "valid_node_mask"],
    ).iloc[0]
    item: Any = row["image"]
    payload = item.get("bytes") if isinstance(item, dict) else None
    if payload is None:
        raise ValueError(f"image in {parquet_path} must contain embedded bytes")
    with Image.open(io.BytesIO(payload)) as image:
        frame_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return (
        np.ascontiguousarray(np.fliplr(frame_rgb)),
        int(row["episode_index"]),
        int(row["task_index"]),
        np.stack([np.stack(node) for node in row["node_points_xyz"]]).astype(np.float32),
        np.asarray(row["valid_node_mask"], dtype=bool),
    )


def read_first_rgb(parquet_path: Path) -> tuple[np.ndarray, int, int]:
    row = pd.read_parquet(
        parquet_path, columns=["image", "episode_index", "task_index"]
    ).iloc[0]
    item: Any = row["image"]
    payload = item.get("bytes") if isinstance(item, dict) else None
    if payload is None:
        raise ValueError(f"image in {parquet_path} must contain embedded bytes")
    with Image.open(io.BytesIO(payload)) as image:
        frame_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return (
        np.ascontiguousarray(np.fliplr(frame_rgb)),
        int(row["episode_index"]),
        int(row["task_index"]),
    )


def project_xyz_to_uv(xyz: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[:, 2] > 0.0)
    uv = np.zeros((len(xyz), 2), dtype=np.float32)
    uv[valid, 0] = xyz[valid, 0] / xyz[valid, 2] * intrinsic[0, 0] + intrinsic[0, 2]
    uv[valid, 1] = xyz[valid, 1] / xyz[valid, 2] * intrinsic[1, 1] + intrinsic[1, 2]
    return uv, valid


def draw_episode_panel(
    frame_rgb: np.ndarray,
    episode_index: int,
    task_index: int,
    node_names: list[str],
    node_points_xyz: np.ndarray,
    valid_node_mask: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    panel = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    valid_nodes = 0
    for node_index in np.flatnonzero(valid_node_mask):
        if node_index >= len(node_names):
            continue
        points, valid = project_xyz_to_uv(node_points_xyz[node_index], intrinsic)
        height, width = panel.shape[:2]
        valid &= (
            (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        points = points[valid]
        if not len(points):
            continue
        valid_nodes += 1
        node_name = node_names[node_index]
        color = COLORS[node_index % len(COLORS)]
        for x, y in points:
            cv2.circle(panel, (int(round(x)), int(round(y))), 2, color, -1, lineType=cv2.LINE_AA)
        center = points.mean(axis=0).round().astype(int)
        cv2.circle(panel, tuple(center), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            panel, f"N{node_index}:{node_name}", (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        panel, f"ep={episode_index:06d} task={task_index} projected_nodes={valid_nodes}",
        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def draw_initialization_panel(
    frame_rgb: np.ndarray,
    episode_index: int,
    task_index: int,
    node_names: list[str],
    locator: Any,
    segmenter: Any,
) -> np.ndarray:
    panel = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = panel.shape[:2]
    point_prompts = []
    for node_index, node_name in enumerate(node_names):
        result = locator.inference(
            text=node_name,
            image=Image.fromarray(frame_rgb),
            resize_scale=2.0,
        )
        points = np.asarray(result.get("points") or [], dtype=np.float32).reshape(-1, 2)
        if not len(points):
            raise RuntimeError(
                f"LocateAnything found no point for episode={episode_index}, node={node_name!r}"
            )
        points[:, 0] = points[:, 0] / 1000.0 * width
        points[:, 1] = points[:, 1] / 1000.0 * height
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        point_prompts.append(points.tolist())

    masks = segmenter.predict(frame_rgb, points=point_prompts, anchor_frame=True)
    if len(masks) != len(node_names):
        raise RuntimeError(
            f"SAM3 returned {len(masks)} masks for {len(node_names)} nodes in episode={episode_index}"
        )
    segmented = 0
    for node_index, (node_name, points, mask) in enumerate(
        zip(node_names, point_prompts, masks, strict=True)
    ):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        if mask.any():
            segmented += 1
        color = COLORS[node_index % len(COLORS)]
        overlay = np.zeros_like(panel)
        overlay[mask] = color
        panel = np.where(
            mask[..., None], (panel * 0.6 + overlay * 0.4).astype(np.uint8), panel
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, color, 1, cv2.LINE_AA)
        points = np.asarray(points, dtype=np.float32)
        for x, y in points:
            cv2.circle(panel, (int(round(x)), int(round(y))), 6, color, -1, cv2.LINE_AA)
        center = points.mean(axis=0).round().astype(int)
        cv2.putText(
            panel, f"N{node_index}:{node_name} A={int(mask.sum())}",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )

    cv2.rectangle(panel, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(
        panel, f"ep={episode_index:06d} task={task_index} segmented={segmented}/{len(node_names)}",
        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def save_grids(
    panels: list[np.ndarray],
    episode_indices: list[int],
    task_indices: list[int],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    grouped = {}
    for panel, episode_index, task_index in zip(
        panels, episode_indices, task_indices, strict=True
    ):
        grouped.setdefault(task_index, []).append((episode_index, panel))

    for task_index in sorted(grouped):
        task_items = sorted(grouped[task_index])
        for start in range(0, len(task_items), 16):
            batch_items = task_items[start:start + 16]
            batch_indices = [episode_index for episode_index, _ in batch_items]
            batch = [panel for _, panel in batch_items]
            blank = np.zeros_like(batch[0])
            batch.extend(blank.copy() for _ in range(16 - len(batch)))
            grid = np.vstack([
                np.hstack(batch[row * 4:(row + 1) * 4]) for row in range(4)
            ])
            save_path = output_dir / (
                f"task_{task_index:04d}_episodes_"
                f"{batch_indices[0]:06d}-{batch_indices[-1]:06d}.png"
            )
            if not cv2.imwrite(os.fspath(save_path), grid):
                raise RuntimeError(f"Failed to save grid: {save_path}")
            print(f"saved: {save_path}")
            saved_paths.append(save_path)
    return saved_paths

def render_initialization_grids(dataset_dir: Path, output_dir: Path) -> list[Path]:
    from src.module.node_locator import NodeLocatorRobo
    from src.module.node_segmenter import NodeSegmenterSAM2

    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No episode parquet files found under {dataset_dir / 'data'}")
    nodes_by_task = load_task_nodes(dataset_dir)
    locator = NodeLocatorRobo()
    segmenter = NodeSegmenterSAM2()
    panels = []
    episode_indices = []
    task_indices = []
    for parquet_path in parquet_paths:
        frame_rgb, episode_index, task_index = read_first_rgb(parquet_path)
        panels.append(draw_initialization_panel(
            frame_rgb, episode_index, task_index, nodes_by_task[task_index], locator, segmenter,
        ))
        episode_indices.append(episode_index)
        task_indices.append(task_index)
        print(f"initialized episode {episode_index:06d}")
    return save_grids(panels, episode_indices, task_indices, output_dir)


def render_grids(dataset_dir: Path, output_dir: Path) -> list[Path]:
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No episode parquet files found under {dataset_dir / 'data'}")
    nodes_by_task = load_task_nodes(dataset_dir)
    intrinsics = load_camera_intrinsics(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    episode_indices = []
    task_indices = []
    for parquet_path in parquet_paths:
        frame_rgb, episode_index, task_index, node_xyz, node_mask = read_first_frame(parquet_path)
        if task_index not in nodes_by_task:
            raise KeyError(f"No taskstructure nodes for task_index={task_index}")
        panels.append(draw_episode_panel(
            frame_rgb, episode_index, task_index, nodes_by_task[task_index],
            node_xyz, node_mask, intrinsics[task_index],
        ))
        episode_indices.append(episode_index)
        task_indices.append(task_index)
    return save_grids(panels, episode_indices, task_indices, output_dir)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.initialize_only:
        render_initialization_grids(dataset_dir, output_dir / "locator_sam3")
    else:
        render_grids(dataset_dir, output_dir)



if __name__ == "__main__":
    main()
