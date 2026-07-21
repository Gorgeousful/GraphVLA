from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "extract_lerobot_tasks.py"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_episode(path: Path, episode_index: int, task_index: int, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = len(values)
    table = pa.table(
        {
            "image": [
                {"bytes": _png_bytes([int(value), int(2 * value), int(3 * value)]), "path": None}
                for value in values
            ],
            "depth": pa.array(
                [[value] * 65536 for value in values], type=pa.list_(pa.float32(), 65536)
            ),
            "value": pa.array([[value, value + 0.5] for value in values], type=pa.list_(pa.float32(), 2)),
            "timestamp": pa.array([index / 10 for index in range(length)], type=pa.float32()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
            "index": pa.array(range(100 + episode_index, 100 + episode_index + length), type=pa.int64()),
            "task_index": pa.array([task_index] * length, type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def _png_bytes(rgb: list[int]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray([[rgb]], dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def make_dataset(root: Path) -> None:
    tasks = [
        {"task_index": 3, "task": "task three"},
        {"task_index": 7, "task": "task seven"},
        {"task_index": 9, "task": "task nine"},
    ]
    episodes = [
        {"episode_index": 4, "tasks": ["task three"], "length": 2},
        {"episode_index": 9, "tasks": ["task seven"], "length": 3},
        {"episode_index": 12, "tasks": ["task seven"], "length": 1},
        {"episode_index": 15, "tasks": ["task nine"], "length": 2},
    ]
    info = {
        "codebase_version": "v2.0",
        "robot_type": "test_robot",
        "total_episodes": 4,
        "total_frames": 8,
        "total_tasks": 3,
        "total_videos": 0,
        "total_chunks": 2,
        "chunks_size": 10,
        "fps": 10,
        "splits": {"train": "0:4"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": [1, 1, 3], "names": ["height", "width", "channel"]},
            "depth": {"dtype": "float32", "shape": [65536], "names": ["pixel"]},
            "value": {"dtype": "float32", "shape": [2], "names": ["value"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "stats.json", {"stale": {"mean": [999]}})
    write_jsonl(root / "meta" / "tasks.jsonl", tasks)
    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n", encoding="utf-8")

    for episode, values in zip(episodes, ([1, 2], [3, 4, 5], [6], [7, 8]), strict=True):
        episode_index = episode["episode_index"]
        task_index = next(task["task_index"] for task in tasks if task["task"] == episode["tasks"][0])
        make_episode(
            root / "data" / f"chunk-{episode_index // 10:03d}" / f"episode_{episode_index:06d}.parquet",
            episode_index,
            task_index,
            values,
        )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_extracts_independent_dataset_and_rebuilds_all_indices(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "7", "3"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = tmp_path / "source_7_3"
    assert read_jsonl(output / "meta" / "tasks.jsonl") == [
        {"task_index": 0, "task": "task seven"},
        {"task_index": 1, "task": "task three"},
    ]
    assert read_jsonl(output / "meta" / "episodes.jsonl") == [
        {"episode_index": 0, "tasks": ["task three"], "length": 2},
        {"episode_index": 1, "tasks": ["task seven"], "length": 3},
        {"episode_index": 2, "tasks": ["task seven"], "length": 1},
    ]

    tables = [
        pq.read_table(output / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
        for index in range(3)
    ]
    assert [table["frame_index"].to_pylist() for table in tables] == [[0, 1], [0, 1, 2], [0]]
    assert [table["episode_index"].to_pylist() for table in tables] == [[0, 0], [1, 1, 1], [2]]
    assert [table["index"].to_pylist() for table in tables] == [[0, 1], [2, 3, 4], [5]]
    assert [table["task_index"].to_pylist() for table in tables] == [[1, 1], [0, 0, 0], [0]]

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 6
    assert info["total_tasks"] == 2
    assert info["total_chunks"] == 1
    assert info["splits"] == {"train": "0:3"}

    stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert stats["value"] == {
        "mean": [3.5, 4.0],
        "std": pytest.approx([1.7078251277, 1.7078251277]),
        "max": [6.0, 6.5],
        "min": [1.0, 1.5],
    }
    assert "stale" not in stats
    assert np.asarray(stats["image"]["mean"]).shape == (3, 1, 1)
    assert len(stats["depth"]["mean"]) == 65536

    source_file = source / "data" / "chunk-000" / "episode_000004.parquet"
    output_file = output / "data" / "chunk-000" / "episode_000000.parquet"
    assert source_file.stat().st_ino != output_file.stat().st_ino
    source_file.write_bytes(b"source changed after extraction")
    assert pq.read_table(output_file).num_rows == 2


@pytest.mark.parametrize(
    ("task_indices", "message"),
    [([7, 7], "must not contain duplicates"), ([99], "Unknown task_index values")],
)
def test_cli_rejects_invalid_task_indices(
    tmp_path: Path, task_indices: list[int], message: str
) -> None:
    source = tmp_path / "source"
    make_dataset(source)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), *map(str, task_indices)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    expected_output = tmp_path / ("source_" + "_".join(map(str, task_indices)))
    assert not expected_output.exists()


def make_video_dataset(root: Path) -> None:
    make_dataset(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["codebase_version"] = "v2.1"
    info["total_videos"] = info["total_episodes"] * 2
    info["features"]["camera"] = {
        "dtype": "video",
        "shape": [1, 1, 3],
        "names": ["height", "width", "rgb"],
        "info": {"video.fps": 10, "video.channels": 3},
    }
    info["features"]["wrist"] = dict(info["features"]["camera"])
    write_json(root / "meta" / "info.json", info)

    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    tasks = {row["task"]: row["task_index"] for row in read_jsonl(root / "meta" / "tasks.jsonl")}
    episode_stats = []
    for episode in episodes:
        episode_index = episode["episode_index"]
        length = episode["length"]
        task_index = tasks[episode["tasks"][0]]
        for video_key in ("camera", "wrist"):
            video = (
                root
                / "videos"
                / f"chunk-{episode_index // info['chunks_size']:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"{video_key}-{episode_index}".encode())
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "camera": {
                        "min": [[[0.0]], [[0.0]], [[0.0]]],
                        "max": [[[1.0]], [[1.0]], [[1.0]]],
                        "mean": [[[0.1]], [[0.2]], [[0.3]]],
                        "std": [[[0.01]], [[0.02]], [[0.03]]],
                        "count": [length],
                    },
                    "frame_index": {"min": [0], "max": [length - 1], "mean": [(length - 1) / 2], "std": [0.0], "count": [length]},
                    "episode_index": {"min": [episode_index], "max": [episode_index], "mean": [float(episode_index)], "std": [0.0], "count": [length]},
                    "index": {"min": [100], "max": [100 + length - 1], "mean": [100 + (length - 1) / 2], "std": [0.0], "count": [length]},
                    "task_index": {"min": [task_index], "max": [task_index], "mean": [float(task_index)], "std": [0.0], "count": [length]},
                },
            }
        )
    write_jsonl(root / "meta" / "episodes_stats.jsonl", episode_stats)


def test_cli_copies_and_reindexes_v21_episode_videos(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_video_dataset(source)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "7"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = tmp_path / "source_7"
    first_source_video = source / "videos" / "chunk-000" / "camera" / "episode_000009.mp4"
    first_output_video = output / "videos" / "chunk-000" / "camera" / "episode_000000.mp4"
    second_output_video = output / "videos" / "chunk-000" / "camera" / "episode_000001.mp4"
    assert first_output_video.read_bytes() == b"camera-9"
    assert second_output_video.read_bytes() == b"camera-12"
    assert (output / "videos" / "chunk-000" / "wrist" / "episode_000000.mp4").read_bytes() == b"wrist-9"
    assert first_source_video.stat().st_ino != first_output_video.stat().st_ino

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_videos"] == 4
    stats = read_jsonl(output / "meta" / "episodes_stats.jsonl")
    assert [row["episode_index"] for row in stats] == [0, 1]
    assert stats[0]["stats"]["camera"]["mean"] == [[[0.1]], [[0.2]], [[0.3]]]
    assert stats[0]["stats"]["episode_index"]["mean"] == [0.0]
    assert stats[0]["stats"]["index"]["min"] == [0.0]
    assert stats[1]["stats"]["index"]["min"] == [3.0]
    assert stats[0]["stats"]["task_index"]["mean"] == [0.0]


def test_cli_rejects_v20_video_without_per_episode_stats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_video_dataset(source)
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    info["codebase_version"] = "v2.0"
    write_json(source / "meta" / "info.json", info)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "7"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "v2.0 video datasets" in result.stderr
    assert not (tmp_path / "source_7").exists()
