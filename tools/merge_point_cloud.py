#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EPS = 1e-6
DEFAULT_COLOR = (0.0, 80.0, 255.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two point-cloud .npy files into one colored point cloud.")
    parser.add_argument("point_cloud_a", nargs="?", type=Path, help="Path to A point cloud, shape (H,W,3/6) or (T,H,W,3/6).")
    parser.add_argument("point_cloud_b", nargs="?", type=Path, help="Path to B point cloud, shape (H,W,3/6) or (T,H,W,3/6).")
    parser.add_argument("--output", "-o", type=Path, help="Output .npy path. Default: A path with _merged suffix.")
    parser.add_argument(
        "--color-target",
        choices=("a", "b", "none"),
        default=None,
        help="Which input to recolor uniformly. Default: b.",
    )
    parser.add_argument(
        "--color",
        nargs=3,
        type=float,
        default=None,
        metavar=("R", "G", "B"),
        help="Uniform color for --color-target, in 0-255 or 0-1 range. Default: blue.",
    )
    parser.add_argument("--keep-single-frame", action="store_true", help="For single-frame inputs, save (N,1,6) instead of (1,N,1,6).")
    return parser.parse_args()


def default_output_path(point_cloud_a: Path) -> Path:
    suffix = point_cloud_a.suffix if point_cloud_a.suffix else ".npy"
    return point_cloud_a.with_name(f"{point_cloud_a.stem}_merged{suffix}")


def prompt_path(label: str, default: Path | None = None, required: bool = False) -> Path:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return Path(value).expanduser()
        if default is not None:
            return default
        if not required:
            return Path()
        print(f"{label} is required.")


def prompt_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    choices_text = "/".join(choices)
    while True:
        value = input(f"{label} ({choices_text}) [{default}]: ").strip().lower()
        if not value:
            return default
        if value in choices:
            return value
        print(f"Please choose one of: {choices_text}.")


def prompt_color(default: tuple[float, float, float]) -> tuple[float, float, float]:
    default_text = " ".join(str(v).rstrip("0").rstrip(".") for v in default)
    while True:
        value = input(f"Color RGB [{default_text}]: ").strip()
        if not value:
            return default
        parts = value.replace(",", " ").split()
        if len(parts) == 3:
            try:
                return tuple(float(part) for part in parts)
            except ValueError:
                pass
        print("Please input three numbers, for example: 0 80 255.")


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.point_cloud_a = args.point_cloud_a or prompt_path("A point cloud path", required=True)
    args.point_cloud_b = args.point_cloud_b or prompt_path("B point cloud path", required=True)

    output_default = default_output_path(args.point_cloud_a)
    args.output = args.output or prompt_path("Output path", default=output_default)
    args.color_target = args.color_target or prompt_choice("Color target", ("a", "b", "none"), "b")
    if args.color is None:
        args.color = prompt_color(DEFAULT_COLOR)
    return args


def load_point_cloud(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    if data.ndim not in (3, 4):
        raise ValueError(f"Expected {path} shape (H,W,C) or (T,H,W,C), got {data.shape}.")
    if data.shape[-1] not in (3, 6):
        raise ValueError(f"Expected {path} last dim C=3 or C=6, got shape {data.shape}.")
    return data.astype(np.float32, copy=False)


def normalize_color(color: tuple[float, float, float] | list[float]) -> np.ndarray:
    color_array = np.asarray(color, dtype=np.float32).reshape(3)
    if np.nanmax(color_array) <= 1.5:
        color_array = color_array * 255.0
    return np.clip(color_array, 0.0, 255.0).astype(np.float32)


def as_temporal(data: np.ndarray) -> tuple[np.ndarray, bool]:
    if data.ndim == 3:
        return data[None], True
    return data, False


def frame_to_points(frame: np.ndarray, fallback_color: np.ndarray) -> np.ndarray:
    xyz = frame[..., :3].reshape(-1, 3).astype(np.float32, copy=False)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > EPS)
    xyz = xyz[valid]
    if frame.shape[-1] == 6:
        rgb = frame[..., 3:6].reshape(-1, 3).astype(np.float32, copy=False)[valid]
        rgb_valid = np.isfinite(rgb).all(axis=1)
        xyz = xyz[rgb_valid]
        rgb = rgb[rgb_valid]
        if len(rgb) and float(np.nanmax(rgb)) <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.float32, copy=False)
    else:
        rgb = np.broadcast_to(fallback_color[None], (len(xyz), 3)).astype(np.float32, copy=True)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32, copy=False)


def recolor(points: np.ndarray, color: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    points = points.copy()
    points[:, 3:6] = color[None]
    return points


def merge_point_clouds(
    point_cloud_a: np.ndarray,
    point_cloud_b: np.ndarray,
    color_target: str,
    color: np.ndarray,
) -> tuple[np.ndarray, bool, list[int]]:
    a, a_single = as_temporal(point_cloud_a)
    b, b_single = as_temporal(point_cloud_b)
    frame_count = max(a.shape[0], b.shape[0])
    if a.shape[0] not in (1, frame_count) or b.shape[0] not in (1, frame_count):
        raise ValueError(f"Frame counts must match or be single-frame broadcastable, got A={a.shape[0]} B={b.shape[0]}.")

    merged_frames = []
    counts = []
    fallback_a = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    fallback_b = np.array([180.0, 180.0, 180.0], dtype=np.float32)
    for frame_index in range(frame_count):
        a_frame = frame_to_points(a[0 if a.shape[0] == 1 else frame_index], fallback_a)
        b_frame = frame_to_points(b[0 if b.shape[0] == 1 else frame_index], fallback_b)
        if color_target == "a":
            a_frame = recolor(a_frame, color)
        elif color_target == "b":
            b_frame = recolor(b_frame, color)
        merged = np.concatenate([a_frame, b_frame], axis=0).astype(np.float32, copy=False)
        merged_frames.append(merged)
        counts.append(int(len(merged)))

    max_count = max(counts) if counts else 0
    output = np.full((frame_count, max_count, 1, 6), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(merged_frames):
        output[frame_index, : len(frame), 0, :] = frame
    return output, a_single and b_single, counts


def main() -> None:
    args = resolve_args(parse_args())
    color = normalize_color(args.color)
    a = load_point_cloud(args.point_cloud_a)
    b = load_point_cloud(args.point_cloud_b)
    merged, single_frame, counts = merge_point_clouds(a, b, args.color_target, color)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if single_frame and args.keep_single_frame:
        np.save(output, merged[0])
        saved_shape = tuple(merged[0].shape)
    else:
        np.save(output, merged)
        saved_shape = tuple(merged.shape)

    print(f"A: {args.point_cloud_a} shape={tuple(a.shape)}")
    print(f"B: {args.point_cloud_b} shape={tuple(b.shape)}")
    print(f"color_target: {args.color_target}")
    if args.color_target != "none":
        print(f"color: {color.tolist()}")
    print(f"frame_count: {len(counts)}")
    print(f"points_per_frame_min_max: {min(counts) if counts else 0} {max(counts) if counts else 0}")
    print(f"saved: {output} shape={saved_shape}")


if __name__ == "__main__":
    main()
