#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
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
    parser = argparse.ArgumentParser(description="Compute LeRobot normalization stats for selected fields.")
    parser.add_argument("dataset_dir", type=Path, help="Local LeRobot dataset directory.")
    parser.add_argument("fields", nargs="+", help="Field names to compute, e.g. observation.state action.")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick stats debugging.")
    parser.add_argument("--batch-size", type=int, default=256, help="CPU batch size for stats computation.")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU DataLoader workers.")
    parser.add_argument("--num-quantile-bins", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <dataset_dir>/meta/norm_stats.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = args.output or dataset_dir / "meta" / "norm_stats.json"

    dataset = LeRobotDataset(
        repo_id=str(dataset_dir),
        video_backend=args.video_backend,
    )
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
    stats = {field: RunningStats(num_quantile_bins=args.num_quantile_bins) for field in args.fields}

    for batch in tqdm(data_loader, desc="Computing norm stats", unit="batch"):
        for field, running_stats in stats.items():
            if field not in batch:
                raise KeyError(f"Field {field!r} not found. Available keys: {sorted(batch.keys())}")
            running_stats.update(batch[field])

    norm_stats = {field: asdict(running_stats.get_statistics()) for field, running_stats in stats.items()}
    output = {"norm_stats": norm_stats}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote norm stats to {output_path}")


if __name__ == "__main__":
    main()
