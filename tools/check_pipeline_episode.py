from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


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
        description="Visualize camera-XYZ node tracks for one processed LIBERO episode.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/data0/luokang/research/GraphVLA/examples/libero/extra/libero_with_depth_7"
        ),
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--local-episode-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data0/luokang/research/GraphVLA/__tmp__/check_pipeline_episode"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def scan_task_episodes(dataset_dir: Path, task_index: int) -> list[int]:
    episodes = []
    for parquet_path in sorted((dataset_dir / "data").glob("chunk-*/*.parquet")):
        meta = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        if int(meta["task_index"].iloc[0]) == int(task_index):
            episodes.append(int(meta["episode_index"].iloc[0]))
    return sorted(episodes)


def episode_parquet_path(
    dataset_dir: Path,
    info: dict[str, Any],
    episode_index: int,
) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_dir / info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def load_agentview_intrinsic(dataset_dir: Path, task_index: int) -> np.ndarray:
    for row in load_json(dataset_dir / "meta" / "cameras.json"):
        if int(row["task_index"]) == int(task_index):
            return np.asarray(
                row["cameras"]["agentview"]["intrinsic"],
                dtype=np.float32,
            )
    raise KeyError(f"No agentview camera calibration for task_index={task_index}")


def decode_frame(item: Any) -> np.ndarray:
    payload = item.get("bytes") if isinstance(item, dict) else None
    if payload is None:
        raise ValueError("image must contain embedded bytes")
    with Image.open(io.BytesIO(payload)) as image:
        frame_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    frame_rgb = np.ascontiguousarray(np.fliplr(frame_rgb))
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def project_xyz_to_uv(
    xyz: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    xyz = np.asarray(xyz, dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[:, 2] > 0.0)
    uv = np.zeros((len(xyz), 2), dtype=np.float32)
    uv[valid, 0] = xyz[valid, 0] / xyz[valid, 2] * intrinsic[0, 0] + intrinsic[0, 2]
    uv[valid, 1] = xyz[valid, 1] / xyz[valid, 2] * intrinsic[1, 1] + intrinsic[1, 2]
    valid &= (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv[valid], valid


def draw_tracks(
    frame: np.ndarray,
    node_points_xyz: Any,
    valid_node_mask: Any,
    subtask_node_mask: Any,
    intrinsic: np.ndarray,
    node_points_vis: Any | None = None,
) -> np.ndarray:
    panel = frame.copy()
    nodes = np.stack([np.stack(node) for node in node_points_xyz]).astype(np.float32)
    point_visibility = (
        None
        if node_points_vis is None
        else np.stack([np.asarray(node, dtype=bool) for node in node_points_vis])
    )
    if point_visibility is not None and point_visibility.shape != nodes.shape[:2]:
        raise ValueError(
            f"node_points_vis shape {point_visibility.shape} does not match "
            f"node_points_xyz {nodes.shape[:2]}"
        )
    valid_nodes = np.asarray(valid_node_mask, dtype=bool)
    active_nodes = np.asarray(subtask_node_mask, dtype=bool)
    draw_order = list(np.flatnonzero(~active_nodes)) + list(np.flatnonzero(active_nodes))
    for node_index in draw_order:
        if (
            node_index >= len(nodes)
            or node_index >= len(valid_nodes)
            or not valid_nodes[node_index]
        ):
            continue
        points, projected_mask = project_xyz_to_uv(
            nodes[node_index], intrinsic, panel.shape[:2]
        )
        if not len(points):
            continue
        active = node_index < len(active_nodes) and active_nodes[node_index]
        color = COLORS[node_index % len(COLORS)] if active else (145, 145, 145)
        pale_color = tuple(int(round(channel * 0.35 + 255 * 0.65)) for channel in color)
        visible = (
            np.ones(len(points), dtype=bool)
            if point_visibility is None
            else point_visibility[node_index, projected_mask]
        )
        for (x, y), is_visible in zip(points, visible, strict=True):
            cv2.circle(
                panel,
                (int(round(x)), int(round(y))),
                2,
                color if is_visible else pale_color,
                -1,
                lineType=cv2.LINE_AA,
            )
        center = np.mean(points, axis=0).round().astype(int)
        cv2.circle(panel, tuple(center), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            panel,
            f"N{node_index}",
            (int(center[0]) + 6, int(center[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return panel


def add_header(canvas: np.ndarray, text: str, header_height: int) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_height), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def render_check_video(
    dataset_dir: Path,
    task_index: int,
    local_episode_index: int,
    output_dir: Path,
) -> Path:
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = scan_task_episodes(dataset_dir, task_index)
    if not episodes:
        raise KeyError(f"No episodes found for task_index={task_index}")
    if not 0 <= local_episode_index < len(episodes):
        raise IndexError(
            f"local_episode_index={local_episode_index} out of range [0, {len(episodes)})"
        )

    episode_index = episodes[local_episode_index]
    parquet_path = episode_parquet_path(dataset_dir, info, episode_index)
    columns = [
        "image",
        "node_points_xyz",
        "valid_node_mask",
        "subtask_node_mask",
        "subtask_id",
        "is_complete",
    ]
    if "node_points_vis" in pq.read_schema(parquet_path).names:
        columns.append("node_points_vis")
    df = pd.read_parquet(
        parquet_path,
        columns=columns,
    )
    intrinsic = load_agentview_intrinsic(dataset_dir, task_index)

    first_frame = decode_frame(df["image"].iloc[0])
    height, width = first_frame.shape[:2]
    header_height = 34
    fps = float(info.get("fps", 10) or 10)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / (
        f"task{task_index:03d}-episode_{episode_index:06d}-node-tracking.mp4"
    )

    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height + header_height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            "24",
            "-g",
            "2",
            "-pix_fmt",
            "yuv420p",
            os.fspath(save_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None

    try:
        for frame_index, row in df.iterrows():
            frame = first_frame if frame_index == 0 else decode_frame(row["image"])
            panel = draw_tracks(
                frame,
                row["node_points_xyz"],
                row["valid_node_mask"],
                row["subtask_node_mask"],
                intrinsic,
                row.get("node_points_vis"),
            )
            canvas = np.zeros((height + header_height, width, 3), dtype=np.uint8)
            canvas[header_height:] = panel
            add_header(
                canvas,
                (
                    f"ep={episode_index} f={frame_index} "
                    f"st={int(row['subtask_id'])} "
                    f"done={int(bool(row['is_complete']))}"
                ),
                header_height,
            )
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()

    return_code = process.wait()
    if return_code != 0:
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr is not None
            else ""
        )
        raise RuntimeError(f"ffmpeg failed for {save_path}: {stderr}")

    print(f"saved: {save_path}")
    print(f"frames: {len(df)}")
    return save_path


def main() -> None:
    args = parse_args()
    render_check_video(
        dataset_dir=args.dataset_dir.resolve(),
        task_index=args.task_index,
        local_episode_index=args.local_episode_index,
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
