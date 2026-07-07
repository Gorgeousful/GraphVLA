#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]
    q01: list[float]
    q99: list[float]


class RunningStats:
    def __init__(self, num_quantile_bins: int = 5000) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.mean_of_squares: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.histograms: list[np.ndarray] | None = None
        self.bin_edges: list[np.ndarray] | None = None
        self.num_quantile_bins = num_quantile_bins

    def update(self, batch: Any) -> None:
        array = np.asarray(batch)
        if np.issubdtype(array.dtype, np.str_):
            raise TypeError("Cannot compute norm stats for string fields")
        if array.size == 0:
            return

        array = array.astype(np.float64, copy=False)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, array.shape[-1])
        else:
            array = array.reshape(-1, array.shape[-1])

        num_elements, vector_length = array.shape
        if self.count == 0:
            self.mean = np.mean(array, axis=0)
            self.mean_of_squares = np.mean(array**2, axis=0)
            self.min = np.min(array, axis=0)
            self.max = np.max(array, axis=0)
            self.histograms = [np.zeros(self.num_quantile_bins, dtype=np.float64) for _ in range(vector_length)]
            self.bin_edges = [
                np.linspace(self.min[i] - 1e-10, self.max[i] + 1e-10, self.num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            assert self.mean is not None
            assert self.mean_of_squares is not None
            assert self.min is not None
            assert self.max is not None
            if vector_length != self.mean.size:
                raise ValueError(f"Vector length changed from {self.mean.size} to {vector_length}")

            new_min = np.min(array, axis=0)
            new_max = np.max(array, axis=0)
            range_changed = np.any(new_min < self.min) or np.any(new_max > self.max)
            self.min = np.minimum(self.min, new_min)
            self.max = np.maximum(self.max, new_max)
            if range_changed:
                self._adjust_histograms()

        self.count += num_elements
        batch_mean = np.mean(array, axis=0)
        batch_mean_of_squares = np.mean(array**2, axis=0)

        assert self.mean is not None
        assert self.mean_of_squares is not None
        self.mean += (batch_mean - self.mean) * (num_elements / self.count)
        self.mean_of_squares += (batch_mean_of_squares - self.mean_of_squares) * (num_elements / self.count)
        self._update_histograms(array)

    def get_statistics(self) -> NormStats:
        if self.count < 2:
            raise ValueError("Cannot compute norm stats from fewer than 2 values")
        assert self.mean is not None
        assert self.mean_of_squares is not None

        variance = self.mean_of_squares - self.mean**2
        std = np.sqrt(np.maximum(variance, 0.0))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self.mean.astype(float).tolist(),
            std=std.astype(float).tolist(),
            q01=q01.astype(float).tolist(),
            q99=q99.astype(float).tolist(),
        )

    def _adjust_histograms(self) -> None:
        assert self.histograms is not None
        assert self.bin_edges is not None
        assert self.min is not None
        assert self.max is not None

        new_histograms = []
        new_bin_edges = []
        for i, old_histogram in enumerate(self.histograms):
            old_edges = self.bin_edges[i]
            new_edges = np.linspace(self.min[i] - 1e-10, self.max[i] + 1e-10, self.num_quantile_bins + 1)
            new_histogram, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=old_histogram)
            new_histograms.append(new_histogram.astype(np.float64, copy=False))
            new_bin_edges.append(new_edges)

        self.histograms = new_histograms
        self.bin_edges = new_bin_edges

    def _update_histograms(self, array: np.ndarray) -> None:
        assert self.histograms is not None
        assert self.bin_edges is not None
        for i in range(array.shape[-1]):
            hist, _ = np.histogram(array[:, i], bins=self.bin_edges[i])
            self.histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        assert self.histograms is not None
        assert self.bin_edges is not None
        results = []
        for quantile in quantiles:
            target_count = quantile * self.count
            values = []
            for hist, edges in zip(self.histograms, self.bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                index = min(np.searchsorted(cumsum, target_count), len(edges) - 2)
                values.append(edges[index])
            results.append(np.asarray(values))
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LeRobot normalization stats for mapped fields.")
    parser.add_argument("dataset_dir", type=Path, help="Local LeRobot dataset directory.")
    parser.add_argument(
        "feature_map",
        help="JSON dict or JSON file path mapping output names to dataset fields.",
    )
    parser.add_argument("--level", choices=("suite", "task", "episode"), default="suite")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick stats debugging.")
    parser.add_argument(
        "--task-indices",
        "--tasks-index",
        "--tasks_index",
        dest="task_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional task_index list. If provided, only frames from these tasks are used.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="CPU batch size for stats computation.")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU DataLoader workers.")
    parser.add_argument("--num-quantile-bins", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <dataset_dir>/meta/norm_stats_<level>.json.")
    return parser.parse_args()


def parse_feature_map(value: str) -> dict[str, str]:
    path = Path(value)
    if path.exists():
        mapping = json.loads(path.read_text(encoding="utf-8"))
    else:
        mapping = json.loads(value)

    if not isinstance(mapping, dict):
        raise TypeError("feature_map must be a JSON object mapping output names to dataset fields")
    result: dict[str, str] = {}
    for output_name, dataset_field in mapping.items():
        if not isinstance(output_name, str) or not isinstance(dataset_field, str):
            raise TypeError("feature_map keys and values must both be strings")
        result[output_name] = dataset_field
    return result


def _dataset_task_indices(dataset: LeRobotDataset) -> np.ndarray:
    if hasattr(dataset, "hf_dataset") and "task_index" in dataset.hf_dataset.column_names:
        return np.asarray(dataset.hf_dataset["task_index"], dtype=np.int64)
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        episodes = dataset.meta.episodes
        if hasattr(episodes, "column_names") and {"episode_index", "task_index"} <= set(episodes.column_names):
            episode_to_task = {
                int(row["episode_index"]): int(row["task_index"])
                for row in episodes
            }
            episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
            return np.asarray([episode_to_task[int(index)] for index in episode_indices], dtype=np.int64)
    raise KeyError("Cannot find task_index metadata in LeRobotDataset.")


def _filter_dataset_by_tasks(dataset: LeRobotDataset, task_indices: list[int] | None):
    if task_indices is None:
        return dataset
    allowed = set(int(index) for index in task_indices)
    frame_task_indices = _dataset_task_indices(dataset)
    selected_indices = np.flatnonzero(np.isin(frame_task_indices, list(allowed))).tolist()
    if not selected_indices:
        raise ValueError(f"No frames found for task_indices={sorted(allowed)}")
    print(f"Selected {len(selected_indices)} / {len(dataset)} frames from task_indices={sorted(allowed)}")
    return Subset(dataset, selected_indices)


def _make_lerobot_dataset(dataset_dir: Path, **kwargs: Any) -> LeRobotDataset:
    dataset_dir = Path(dataset_dir)
    try:
        return LeRobotDataset(repo_id=dataset_dir.name, root=dataset_dir, **kwargs)
    except TypeError:
        return LeRobotDataset(repo_id=str(dataset_dir), **kwargs)


FIELD_ALIASES = {"depth_rel": "depths_rel"}


def _canonical_field(field: str) -> str:
    return FIELD_ALIASES.get(field, field)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return np.asarray(value)


def _extract_special_field_values(field: str, batch: dict[str, Any]) -> np.ndarray | None:
    canonical = _canonical_field(field)
    if canonical == "depths_rel":
        if "depths_rel" not in batch or "far_background_mask" not in batch:
            raise KeyError("Field 'depths_rel' stats require both 'depths_rel' and 'far_background_mask'.")
        depths = _to_numpy(batch["depths_rel"]).astype(np.float64, copy=False)
        far_background = _to_numpy(batch["far_background_mask"]).astype(bool, copy=False)
        if depths.shape != far_background.shape:
            raise ValueError(f"depths_rel shape {depths.shape} does not match far_background_mask shape {far_background.shape}")
        valid = (~far_background) & np.isfinite(depths)
        return depths[valid].reshape(-1, 1)

    if canonical == "gripper_uvd":
        if "gripper_uvd" not in batch:
            raise KeyError("Field 'gripper_uvd' not found.")
        gripper_uvd = _to_numpy(batch["gripper_uvd"]).astype(np.float64, copy=False)
        if gripper_uvd.ndim < 3 or gripper_uvd.shape[-1] < 3:
            raise ValueError(f"Expected gripper_uvd shape (..., 3, 3), got {gripper_uvd.shape}")
        d_values = gripper_uvd[..., 2]
        return d_values[np.isfinite(d_values)].reshape(-1, 1)

    return None


def _extract_special_field_values_by_sample(field: str, batch: dict[str, Any]) -> list[np.ndarray] | None:
    canonical = _canonical_field(field)
    if canonical == "depths_rel":
        if "depths_rel" not in batch or "far_background_mask" not in batch:
            raise KeyError("Field 'depths_rel' stats require both 'depths_rel' and 'far_background_mask'.")
        depths = _to_numpy(batch["depths_rel"]).astype(np.float64, copy=False)
        far_background = _to_numpy(batch["far_background_mask"]).astype(bool, copy=False)
        if depths.shape != far_background.shape:
            raise ValueError(f"depths_rel shape {depths.shape} does not match far_background_mask shape {far_background.shape}")
        values = []
        for depth, mask in zip(depths, far_background, strict=True):
            valid = (~mask) & np.isfinite(depth)
            values.append(depth[valid].reshape(-1, 1))
        return values

    if canonical == "gripper_uvd":
        if "gripper_uvd" not in batch:
            raise KeyError("Field 'gripper_uvd' not found.")
        gripper_uvd = _to_numpy(batch["gripper_uvd"]).astype(np.float64, copy=False)
        if gripper_uvd.ndim < 3 or gripper_uvd.shape[-1] < 3:
            raise ValueError(f"Expected gripper_uvd shape (..., 3, 3), got {gripper_uvd.shape}")
        values = []
        for sample in gripper_uvd:
            d_values = sample[..., 2]
            values.append(d_values[np.isfinite(d_values)].reshape(-1, 1))
        return values

    return None


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = args.output or dataset_dir / "meta" / f"norm_stats_{args.level}.json"

    dataset = _make_lerobot_dataset(
        dataset_dir,
        video_backend=args.video_backend,
    )
    dataset = _filter_dataset_by_tasks(dataset, args.task_indices)
    if args.max_frames is not None:
        dataset = Subset(dataset, range(min(len(dataset), args.max_frames)))

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    feature_map = parse_feature_map(args.feature_map)
    if args.level == "suite":
        stats = {
            output_name: RunningStats(num_quantile_bins=args.num_quantile_bins)
            for output_name in feature_map
        }
    else:
        stats = {output_name: {} for output_name in feature_map}

    for batch in tqdm(data_loader, desc="Computing norm stats", unit="batch"):
        if args.level == "suite":
            for output_name, running_stats in stats.items():
                field = feature_map[output_name]
                special_values = _extract_special_field_values(field, batch)
                if special_values is not None:
                    running_stats.update(special_values)
                    continue
                if field not in batch:
                    raise KeyError(f"Field {field!r} not found. Available keys: {sorted(batch.keys())}")
                running_stats.update(batch[field])
        else:
            index_field = "task_index" if args.level == "task" else "episode_index"
            if index_field not in batch:
                raise KeyError(f"Field {index_field!r} not found. Available keys: {sorted(batch.keys())}")
            group_indices = _to_numpy(batch[index_field]).reshape(-1)
            for output_name, field_stats in stats.items():
                field = feature_map[output_name]
                special_values_by_sample = _extract_special_field_values_by_sample(field, batch)
                if special_values_by_sample is not None:
                    if len(special_values_by_sample) != group_indices.shape[0]:
                        raise ValueError(
                            f"Field {field!r} batch dimension {len(special_values_by_sample)} does not match "
                            f"{index_field!r} dimension {group_indices.shape[0]}"
                        )
                    for group_index, value in zip(group_indices, special_values_by_sample, strict=True):
                        group_key = str(int(group_index.item() if hasattr(group_index, "item") else group_index))
                        running_stats = field_stats.setdefault(
                            group_key,
                            RunningStats(num_quantile_bins=args.num_quantile_bins),
                        )
                        running_stats.update(value)
                    continue

                if field not in batch:
                    raise KeyError(f"Field {field!r} not found. Available keys: {sorted(batch.keys())}")
                values = _to_numpy(batch[field])
                if values.shape[0] != group_indices.shape[0]:
                    raise ValueError(
                        f"Field {field!r} batch dimension {values.shape[0]} does not match "
                        f"{index_field!r} dimension {group_indices.shape[0]}"
                    )
                for group_index, value in zip(group_indices, values, strict=True):
                    group_key = str(int(group_index.item() if hasattr(group_index, "item") else group_index))
                    running_stats = field_stats.setdefault(
                        group_key,
                        RunningStats(num_quantile_bins=args.num_quantile_bins),
                    )
                    running_stats.update(value)

    if args.level == "suite":
        norm_stats = {field: asdict(running_stats.get_statistics()) for field, running_stats in stats.items()}
    else:
        norm_stats = {
            field: {
                group_key: asdict(running_stats.get_statistics())
                for group_key, running_stats in sorted(field_stats.items(), key=lambda item: int(item[0]))
            }
            for field, field_stats in stats.items()
        }
    output = {"level": args.level, "norm_stats": norm_stats}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote norm stats to {output_path}")


if __name__ == "__main__":
    main()
