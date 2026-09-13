"""Add fixed-size agent-view scene point clouds to a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pytorch3d.ops import sample_farthest_points
from tqdm import tqdm

DEFAULT_DEPTH_FIELD = "observation.depth.agentview"
DEFAULT_OUTPUT_FIELD = "observation.point_cloud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Back-project agent-view metric depth and add a fixed-size camera-frame "
            "scene point cloud to every parquet row."
        )
    )
    parser.add_argument("dataset_dir", type=Path, help="Local LeRobot dataset root.")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Frames processed per GPU batch."
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8192,
        help=(
            "Deterministically reduce each depth image to this many candidates "
            "before FPS; 0 uses all pixels."
        ),
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device, for example cuda:0 or cpu."
    )
    parser.add_argument("--depth-field", default=DEFAULT_DEPTH_FIELD)
    parser.add_argument("--output-field", default=DEFAULT_OUTPUT_FIELD)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute episodes already containing the output field.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({value}) but is not available")
    return device


def load_camera_intrinsics(dataset_dir: Path) -> dict[int, np.ndarray]:
    path = dataset_dir / "meta" / "cameras.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = {}
    for row in rows:
        task_index = int(row["task_index"])
        matrix = np.asarray(row["cameras"]["agentview"]["intrinsic"], dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"task {task_index} agentview intrinsic has shape {matrix.shape}, "
                "expected (3, 3)"
            )
        intrinsics[task_index] = matrix
    return intrinsics


def depth_column_to_numpy(column: pa.ChunkedArray, field: str) -> np.ndarray:
    array = column.combine_chunks()
    first_shape = np.asarray(array[0].as_py()).shape
    values = array
    while (
        pa.types.is_list(values.type)
        or pa.types.is_large_list(values.type)
        or pa.types.is_fixed_size_list(values.type)
    ):
        values = values.values
    depths = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
    expected_size = len(array) * int(np.prod(first_shape))
    if depths.size != expected_size:
        raise ValueError(f"{field} contains variable-size depth frames")
    depths = depths.reshape((len(array), *first_shape))
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    if depths.ndim != 3:
        raise ValueError(
            f"{field} must have shape [T,H,W] or [T,H,W,1], got {depths.shape}"
        )
    return depths


def candidate_pixel_indices(
    height: int, width: int, max_candidates: int, device: torch.device
) -> torch.Tensor:
    total = height * width
    if max_candidates <= 0 or max_candidates >= total:
        return torch.arange(total, device=device)
    return (
        torch.linspace(0, total - 1, steps=max_candidates, device=device).round().long()
    )


@torch.inference_mode()
def depth_to_fps_point_clouds(
    depths: np.ndarray,
    intrinsic: np.ndarray,
    *,
    num_points: int,
    batch_size: int,
    max_candidates: int,
    device: torch.device,
) -> np.ndarray:
    if num_points <= 0 or batch_size <= 0:
        raise ValueError("num_points and batch_size must be positive")

    num_frames, height, width = depths.shape
    pixel_indices = candidate_pixel_indices(height, width, max_candidates, device)
    u = (pixel_indices % width).to(torch.float32)
    v = torch.div(pixel_indices, width, rounding_mode="floor").to(torch.float32)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    rays = torch.stack(((u - cx) / fx, (v - cy) / fy, torch.ones_like(u)), dim=-1)

    output = np.empty((num_frames, num_points, 3), dtype=np.float32)
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        depth = torch.tensor(depths[start:end], device=device).flatten(1)[
            :, pixel_indices
        ]
        valid = torch.isfinite(depth) & (depth > 0)
        lengths = valid.sum(dim=1)
        if torch.any(lengths < num_points):
            bad = (torch.nonzero(lengths < num_points).flatten() + start).tolist()
            raise ValueError(
                f"frames {bad} have fewer than {num_points} valid depth candidates; "
                "increase --max-candidates or use --max-candidates 0"
            )

        points = rays.unsqueeze(0) * depth.unsqueeze(-1)
        max_valid = int(lengths.max().item())
        compact = torch.zeros(
            (end - start, max_valid, 3), device=device, dtype=torch.float32
        )
        row, column = torch.nonzero(valid, as_tuple=True)
        destination = valid.cumsum(dim=1)[row, column] - 1
        compact[row, destination] = points[row, column]
        sampled, _ = sample_farthest_points(
            compact,
            lengths=lengths,
            K=num_points,
            random_start_point=False,
        )
        output[start:end] = sampled.cpu().numpy()
    return output


def point_cloud_array(points: np.ndarray) -> pa.Array:
    flat = pa.array(points.reshape(-1), type=pa.float32())
    xyz = pa.FixedSizeListArray.from_arrays(flat, 3)
    frames = pa.FixedSizeListArray.from_arrays(xyz, points.shape[1])
    return frames.cast(pa.list_(pa.list_(pa.float32())))


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path)
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_episode(
    path: Path,
    *,
    intrinsics: dict[int, np.ndarray],
    depth_field: str,
    output_field: str,
    num_points: int,
    batch_size: int,
    max_candidates: int,
    device: torch.device,
    overwrite: bool,
) -> bool:
    schema_names = pq.read_schema(path).names
    if output_field in schema_names and not overwrite:
        return False
    required = {depth_field, "task_index"}
    missing = required.difference(schema_names)
    if missing:
        raise KeyError(f"{path} is missing required fields: {sorted(missing)}")

    table = pq.read_table(path)
    task_indices = np.asarray(
        table["task_index"].combine_chunks().to_numpy(), dtype=np.int64
    )
    unique_tasks = np.unique(task_indices)
    if unique_tasks.size != 1:
        raise ValueError(
            f"{path} contains multiple task indices: {unique_tasks.tolist()}"
        )
    task_index = int(unique_tasks[0])
    if task_index not in intrinsics:
        raise KeyError(
            f"No agentview camera intrinsic found for task_index={task_index}"
        )

    depths = depth_column_to_numpy(table[depth_field], depth_field)
    points = depth_to_fps_point_clouds(
        depths,
        intrinsics[task_index],
        num_points=num_points,
        batch_size=batch_size,
        max_candidates=max_candidates,
        device=device,
    )
    column = point_cloud_array(points)
    if output_field in table.column_names:
        table = table.set_column(
            table.column_names.index(output_field), output_field, column
        )
    else:
        table = table.append_column(output_field, column)
    write_parquet_atomic(table, path)
    return True


def update_info(dataset_dir: Path, output_field: str, num_points: int) -> None:
    path = dataset_dir / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    feature = {
        "dtype": "float32",
        "shape": [num_points, 3],
        "names": ["point", "xyz"],
    }
    if info.setdefault("features", {}).get(output_field) == feature:
        return
    info["features"][output_field] = feature
    original_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".info.", suffix=".json.tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    device = resolve_device(args.device)
    intrinsics = load_camera_intrinsics(dataset_dir)
    parquet_paths = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No episode parquet files found under {dataset_dir / 'data'}"
        )

    processed = 0
    for path in tqdm(
        parquet_paths, desc=f"scene point clouds ({device})", unit="episode"
    ):
        processed += process_episode(
            path,
            intrinsics=intrinsics,
            depth_field=args.depth_field,
            output_field=args.output_field,
            num_points=args.num_points,
            batch_size=args.batch_size,
            max_candidates=args.max_candidates,
            device=device,
            overwrite=args.overwrite,
        )
    update_info(dataset_dir, args.output_field, args.num_points)
    print(
        f"Processed {processed}/{len(parquet_paths)} episodes; field={args.output_field!r}, shape=[{args.num_points}, 3]"
    )


if __name__ == "__main__":
    main()
