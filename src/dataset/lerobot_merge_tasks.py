#!/usr/bin/env python
"""Merge local LeRobot v2 datasets into one reindexed dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+", help="LeRobot v2 dataset directories to merge.")
    parser.add_argument("--output", type=Path, help="Defaults to <first_dataset>_merged.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel episode workers (default: 8).")
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


def replace_index_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        raise ValueError(f"Missing required parquet column: {name}")
    position = table.column_names.index(name)
    return table.set_column(position, name, pa.array(values, type=table.schema.field(name).type))


def episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
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
    source: Path,
    output: Path,
    info: dict[str, Any],
    keys: list[str],
    old_episode: int,
    new_episode: int,
    output_chunks_size: int,
) -> None:
    source_chunks_size = int(info["chunks_size"])
    for key in keys:
        source_path = source / info["video_path"].format(
            episode_chunk=old_episode // source_chunks_size,
            video_key=key,
            episode_index=old_episode,
        )
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing episode video: {source_path}")
        output_path = output / VIDEO_PATH.format(
            episode_chunk=new_episode // output_chunks_size,
            video_key=key,
            episode_index=new_episode,
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


def validate_dataset(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    info = load_json(root / "meta" / "info.json")
    if not str(info.get("codebase_version", "")).startswith("v2"):
        raise ValueError(f"Only LeRobot v2 datasets are supported: {root}")
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    task_indices = [int(task["task_index"]) for task in tasks]
    task_texts = [str(task["task"]) for task in tasks]
    if len(task_indices) != len(set(task_indices)):
        raise ValueError(f"Duplicate task_index values in {root}")
    if len(task_texts) != len(set(task_texts)):
        raise ValueError(f"Duplicate task texts in {root}")
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    if len(episode_indices) != len(set(episode_indices)):
        raise ValueError(f"Duplicate episode_index values in {root}")
    return info, tasks, sorted(episodes, key=lambda row: int(row["episode_index"]))


def validate_compatibility(
    first_root: Path,
    first_info: dict[str, Any],
    root: Path,
    info: dict[str, Any],
) -> None:
    for key in ("codebase_version", "fps", "features"):
        if info.get(key) != first_info.get(key):
            raise ValueError(
                f"Incompatible {key}: {root} has {info.get(key)!r}, "
                f"but {first_root} has {first_info.get(key)!r}"
            )


def episode_task_index(
    root: Path, episode: dict[str, Any], task_index_by_text: dict[str, int]
) -> int:
    texts = episode.get("tasks", [])
    indices = {task_index_by_text[text] for text in texts if text in task_index_by_text}
    if len(indices) != 1:
        raise ValueError(
            f"Episode {episode['episode_index']} in {root} must map to exactly one task_index, "
            f"got {sorted(indices)} from tasks={texts!r}"
        )
    return indices.pop()


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


def merge_datasets(sources: list[Path], output: Path, workers: int = 8) -> None:
    if not sources:
        raise ValueError("At least one source dataset is required")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    sources = [source.resolve() for source in sources]
    if len(sources) != len(set(sources)):
        raise ValueError("Source datasets must not contain duplicates")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output in sources:
        raise ValueError("Output must differ from every source dataset")

    datasets = []
    for source in sources:
        info, tasks, episodes = validate_dataset(source)
        datasets.append((source, info, tasks, episodes))

    first_source, first_info, _, _ = datasets[0]
    keys = video_keys(first_info["features"])
    uses_episode_stats = first_info["codebase_version"] != "v2.0"
    if keys and not uses_episode_stats:
        raise ValueError("v2.0 video datasets are unsupported because they lack per-episode video stats")

    merged_tasks: list[dict[str, Any]] = []
    merged_task_by_text: dict[str, int] = {}
    jobs = []
    global_index = 0
    new_episode = 0
    for source, info, tasks, episodes in datasets:
        validate_compatibility(first_source, first_info, source, info)
        source_task_by_text = {str(task["task"]): int(task["task_index"]) for task in tasks}
        source_task_text_by_index = {index: text for text, index in source_task_by_text.items()}
        source_stats = {}
        if uses_episode_stats:
            stats_rows = load_jsonl(source / "meta" / "episodes_stats.jsonl")
            source_stats = {int(row["episode_index"]): row for row in stats_rows}
        for task in tasks:
            text = str(task["task"])
            if text not in merged_task_by_text:
                merged_task_by_text[text] = len(merged_tasks)
                merged_tasks.append({**task, "task_index": len(merged_tasks)})
        for episode in episodes:
            old_episode = int(episode["episode_index"])
            old_task = episode_task_index(source, episode, source_task_by_text)
            task_text = source_task_text_by_index[old_task]
            length = int(episode["length"])
            jobs.append(
                (
                    source,
                    info,
                    episode,
                    source_stats.get(old_episode),
                    old_task,
                    merged_task_by_text[task_text],
                    new_episode,
                    global_index,
                )
            )
            new_episode += 1
            global_index += length

    if not jobs:
        raise ValueError("No episodes found in the source datasets")

    reference_source, reference_info, reference_episode = jobs[0][0], jobs[0][1], jobs[0][2]
    reference_path = episode_path(
        reference_source, reference_info, int(reference_episode["episode_index"])
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {reference_path}")
    reference_schema = pq.read_schema(reference_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    staging = temp_parent / output.name
    chunks_size = int(first_info["chunks_size"])
    try:
        def process_episode(job: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any] | None, int, int, int]:
            (
                source,
                info,
                episode,
                source_stats,
                old_task,
                new_task,
                new_episode,
                episode_global_index,
            ) = job
            old_episode = int(episode["episode_index"])
            source_path = episode_path(source, info, old_episode)
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing episode parquet: {source_path}")
            table = pq.read_table(source_path)
            if not table.schema.equals(reference_schema, check_metadata=False):
                raise ValueError(
                    f"Incompatible parquet schema in {source_path}:\n"
                    f"expected:\n{reference_schema}\nactual:\n{table.schema}"
                )
            length = table.num_rows
            if length != int(episode["length"]):
                raise ValueError(
                    f"Episode {old_episode} in {source} metadata length is {episode['length']}, "
                    f"parquet has {length} rows"
                )
            parquet_tasks = set(table["task_index"].to_pylist())
            if parquet_tasks != {old_task}:
                raise ValueError(
                    f"Episode {old_episode} in {source} task mismatch: "
                    f"metadata={old_task}, parquet={sorted(parquet_tasks)}"
                )

            table = replace_index_column(table, "frame_index", np.arange(length, dtype=np.int64))
            table = replace_index_column(
                table, "episode_index", np.full(length, new_episode, dtype=np.int64)
            )
            table = replace_index_column(
                table,
                "index",
                np.arange(episode_global_index, episode_global_index + length, dtype=np.int64),
            )
            table = replace_index_column(
                table, "task_index", np.full(length, new_task, dtype=np.int64)
            )
            destination = output_episode_path(staging, chunks_size, new_episode)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)
            copy_episode_videos(
                source, staging, info, keys, old_episode, new_episode, chunks_size
            )

            episode_stats = None
            if uses_episode_stats:
                if source_stats is None:
                    raise ValueError(f"Missing episodes_stats entry for episode {old_episode} in {source}")
                episode_stats = reindex_episode_stats(
                    source_stats, new_episode, new_task, episode_global_index, length
                )
            return (
                {**episode, "episode_index": new_episode, "length": length},
                episode_stats,
                old_episode,
                new_episode,
                length,
            )

        merged_episodes = []
        merged_episode_stats = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(process_episode, jobs)
            for completed, (episode, stats, old_episode, new_episode, length) in enumerate(results, 1):
                merged_episodes.append(episode)
                if stats is not None:
                    merged_episode_stats.append(stats)
                print(
                    f"[{completed}/{len(jobs)}] episode {old_episode} -> {new_episode} "
                    f"({length} frames)"
                )

        write_json(
            staging / "meta" / "info.json",
            make_info(first_info, len(merged_episodes), global_index, len(merged_tasks), len(keys)),
        )
        write_jsonl(staging / "meta" / "tasks.jsonl", merged_tasks)
        write_jsonl(staging / "meta" / "episodes.jsonl", merged_episodes)
        if uses_episode_stats:
            write_jsonl(staging / "meta" / "episodes_stats.jsonl", merged_episode_stats)
        attributes = first_source / ".gitattributes"
        if attributes.is_file():
            shutil.copy2(attributes, staging / ".gitattributes")
        staging.rename(output)
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def main() -> None:
    args = parse_args()
    output = args.output or args.datasets[0].with_name(args.datasets[0].name + "_merged")
    merge_datasets(args.datasets, output, args.workers)
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
