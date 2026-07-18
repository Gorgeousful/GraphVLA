#!/usr/bin/env python
"""Extract selected tasks into an independent, reindexed LeRobot v2 dataset."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
STATS_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Source LeRobot v2 dataset directory.")
    parser.add_argument("task_indices", type=int, nargs="+", help="Source task_index values to extract.")
    parser.add_argument("--output", type=Path, help="Defaults to <dataset>_<ids>.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total: np.ndarray | None = None
        self.total_sq: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        values = values.reshape(-1, values.shape[-1])
        if not len(values):
            return
        batch_total = values.sum(axis=0)
        batch_total_sq = np.square(values).sum(axis=0)
        batch_min = values.min(axis=0)
        batch_max = values.max(axis=0)
        if self.total is None:
            self.total = batch_total
            self.total_sq = batch_total_sq
            self.minimum = batch_min
            self.maximum = batch_max
        else:
            self.total += batch_total
            self.total_sq += batch_total_sq
            self.minimum = np.minimum(self.minimum, batch_min)
            self.maximum = np.maximum(self.maximum, batch_max)
        self.count += len(values)

    def result(self, keep_image_dims: bool = False) -> dict[str, list[float]]:
        if not self.count or self.total is None:
            raise ValueError("Cannot compute statistics from an empty feature")
        mean = self.total / self.count
        variance = self.total_sq / self.count - np.square(mean)
        shape = (-1, 1, 1) if keep_image_dims else (-1,)
        return {
            "mean": mean.reshape(shape).tolist(),
            "std": np.sqrt(np.maximum(variance, 0.0)).reshape(shape).tolist(),
            "max": self.maximum.reshape(shape).tolist(),
            "min": self.minimum.reshape(shape).tolist(),
        }


def numeric_values(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type) or pa.types.is_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(array), -1)
    return array.to_numpy(zero_copy_only=False)


def image_values(column: pa.ChunkedArray) -> np.ndarray:
    images = []
    for item in column.to_pylist():
        payload = item.get("bytes") if isinstance(item, dict) else None
        if payload is None:
            raise ValueError("Path-backed image fields cannot be copied independently")
        with Image.open(io.BytesIO(payload)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        images.append(array.reshape(-1, array.shape[-1]))
    return np.concatenate(images, axis=0)


def update_stats(
    accumulators: dict[str, RunningStats], table: pa.Table, features: dict[str, dict[str, Any]]
) -> None:
    for offset in range(0, table.num_rows, STATS_BATCH_SIZE):
        batch = table.slice(offset, STATS_BATCH_SIZE)
        for name, feature in features.items():
            if name not in batch.column_names:
                continue
            dtype = feature.get("dtype")
            if dtype in {"video", "string"}:
                continue
            values = image_values(batch[name]) if dtype == "image" else numeric_values(batch[name])
            accumulators.setdefault(name, RunningStats()).update(values)


def replace_index_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        raise ValueError(f"Missing required parquet column: {name}")
    position = table.column_names.index(name)
    return table.set_column(position, name, pa.array(values, type=table.schema.field(name).type))


def source_episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info["chunks_size"])
    relative = info["data_path"].format(
        episode_chunk=episode_index // chunks_size, episode_index=episode_index
    )
    return root / relative


def output_episode_path(root: Path, chunks_size: int, episode_index: int) -> Path:
    return root / DATA_PATH.format(
        episode_chunk=episode_index // chunks_size, episode_index=episode_index
    )


def video_keys(features: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, feature in features.items() if feature.get("dtype") == "video"]


def copy_episode_videos(
    source: Path, output: Path, info: dict[str, Any], keys: list[str], old_episode: int, new_episode: int
) -> None:
    chunks_size = int(info["chunks_size"])
    for key in keys:
        source_path = source / info["video_path"].format(
            episode_chunk=old_episode // chunks_size, video_key=key, episode_index=old_episode
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing episode video: {source_path}")
        output_path = output / VIDEO_PATH.format(
            episode_chunk=new_episode // chunks_size, video_key=key, episode_index=new_episode
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def sequence_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [len(values)],
    }


def reindex_episode_stats(
    source_row: dict[str, Any], new_episode: int, new_task: int, global_index: int, length: int
) -> dict[str, Any]:
    row = copy.deepcopy(source_row)
    row["episode_index"] = new_episode
    row["stats"].update(
        frame_index=sequence_stats(np.arange(length)),
        episode_index=sequence_stats(np.full(length, new_episode)),
        index=sequence_stats(np.arange(global_index, global_index + length)),
        task_index=sequence_stats(np.full(length, new_task)),
    )
    return row


def validate_selection(tasks: list[dict[str, Any]], requested: list[int]) -> list[dict[str, Any]]:
    if len(requested) != len(set(requested)):
        raise ValueError("task_indices must not contain duplicates")
    by_index = {int(task["task_index"]): task for task in tasks}
    missing = [index for index in requested if index not in by_index]
    if missing:
        raise ValueError(f"Unknown task_index values: {missing}; available: {sorted(by_index)}")
    return [by_index[index] for index in requested]


def selected_episodes(
    episodes: list[dict[str, Any]], tasks: list[dict[str, Any]], requested: set[int]
) -> list[tuple[dict[str, Any], int]]:
    task_by_text = {task["task"]: int(task["task_index"]) for task in tasks}
    selected = []
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        indices = {task_by_text[text] for text in episode["tasks"] if text in task_by_text}
        if len(indices) != 1:
            raise ValueError(
                f"Episode {episode['episode_index']} must map to exactly one task_index, got {sorted(indices)}"
            )
        task_index = indices.pop()
        if task_index in requested:
            selected.append((episode, task_index))
    if not selected:
        raise ValueError("No episodes matched the requested task_indices")
    return selected


def make_info(
    source: dict[str, Any], episodes: int, frames: int, tasks: int, videos_per_episode: int
) -> dict[str, Any]:
    chunks_size = int(source["chunks_size"])
    info = dict(source)
    info.update(
        total_episodes=episodes,
        total_frames=frames,
        total_tasks=tasks,
        total_videos=episodes * videos_per_episode,
        total_chunks=math.ceil(episodes / chunks_size),
        splits={"train": f"0:{episodes}"},
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
    )
    return info


def extract_dataset(source: Path, output: Path, requested: list[int]) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source == output.resolve():
        raise ValueError("Output must differ from the source dataset")

    info = load_json(source / "meta" / "info.json")
    if not str(info.get("codebase_version", "")).startswith("v2"):
        raise ValueError("Only LeRobot v2 datasets are supported")
    tasks = load_jsonl(source / "meta" / "tasks.jsonl")
    episodes = load_jsonl(source / "meta" / "episodes.jsonl")
    chosen_tasks = validate_selection(tasks, requested)
    chosen_episodes = selected_episodes(episodes, tasks, set(requested))
    task_remap = {old: new for new, old in enumerate(requested)}
    new_tasks = [{**task, "task_index": new} for new, task in enumerate(chosen_tasks)]
    keys = video_keys(info["features"])
    uses_episode_stats = info["codebase_version"] != "v2.0"
    if keys and not uses_episode_stats:
        raise ValueError(
            "v2.0 video datasets are unsupported because they lack per-episode video stats"
        )
    source_episode_stats: dict[int, dict[str, Any]] = {}
    if uses_episode_stats:
        stats_rows = load_jsonl(source / "meta" / "episodes_stats.jsonl")
        source_episode_stats = {int(row["episode_index"]): row for row in stats_rows}

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    staging = temp_parent / output.name
    try:
        accumulators: dict[str, RunningStats] = {}
        new_episodes = []
        new_episode_stats = []
        global_index = 0
        chunks_size = int(info["chunks_size"])
        for new_episode, (episode, old_task) in enumerate(chosen_episodes):
            old_episode = int(episode["episode_index"])
            source_path = source_episode_path(source, info, old_episode)
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing episode parquet: {source_path}")
            table = pq.read_table(source_path)
            length = table.num_rows
            if length != int(episode["length"]):
                raise ValueError(
                    f"Episode {old_episode} metadata length is {episode['length']}, parquet has {length} rows"
                )
            source_tasks = set(table["task_index"].to_pylist())
            if source_tasks != {old_task}:
                raise ValueError(
                    f"Episode {old_episode} task mismatch: metadata={old_task}, parquet={sorted(source_tasks)}"
                )

            table = replace_index_column(table, "frame_index", np.arange(length, dtype=np.int64))
            table = replace_index_column(
                table, "episode_index", np.full(length, new_episode, dtype=np.int64)
            )
            table = replace_index_column(
                table, "index", np.arange(global_index, global_index + length, dtype=np.int64)
            )
            table = replace_index_column(
                table, "task_index", np.full(length, task_remap[old_task], dtype=np.int64)
            )
            destination = output_episode_path(staging, chunks_size, new_episode)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            copy_episode_videos(source, staging, info, keys, old_episode, new_episode)
            if not uses_episode_stats:
                update_stats(accumulators, table, info["features"])
            if uses_episode_stats:
                if old_episode not in source_episode_stats:
                    raise ValueError(f"Missing episodes_stats entry for episode {old_episode}")
                new_episode_stats.append(
                    reindex_episode_stats(
                        source_episode_stats[old_episode], new_episode, task_remap[old_task], global_index, length
                    )
                )
            new_episodes.append({**episode, "episode_index": new_episode, "length": length})
            global_index += length
            print(
                f"[{new_episode + 1}/{len(chosen_episodes)}] "
                f"episode {old_episode} -> {new_episode} ({length} frames)"
            )

        write_json(
            staging / "meta" / "info.json",
            make_info(info, len(new_episodes), global_index, len(new_tasks), len(keys)),
        )
        write_jsonl(staging / "meta" / "tasks.jsonl", new_tasks)
        write_jsonl(staging / "meta" / "episodes.jsonl", new_episodes)
        if uses_episode_stats:
            write_jsonl(staging / "meta" / "episodes_stats.jsonl", new_episode_stats)
        else:
            write_json(
                staging / "meta" / "stats.json",
                {
                    name: stats.result(info["features"][name].get("dtype") == "image")
                    for name, stats in accumulators.items()
                },
            )
        attributes = source / ".gitattributes"
        if attributes.is_file():
            shutil.copy2(attributes, staging / ".gitattributes")
        staging.rename(output)
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def main() -> None:
    args = parse_args()
    suffix = "_" + "_".join(str(index) for index in args.task_indices)
    output = args.output or args.dataset.with_name(args.dataset.name + suffix)
    extract_dataset(args.dataset, output, args.task_indices)
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
