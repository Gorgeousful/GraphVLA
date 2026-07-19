from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.dataset.pipeline import OfflinePipeline
from src.dataset.pipeline import PipelineConfig


def test_process_episode_writes_normalized_libero_gripper_openness(tmp_path: Path) -> None:
    meta_dir = tmp_path / "meta"
    data_dir = tmp_path / "data" / "chunk-000"
    meta_dir.mkdir()
    data_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps({
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "features": {"observation.images.image": {"dtype": "video"}},
    }))
    (meta_dir / "tasks.jsonl").write_text('{"task_index": 0, "task": "test task"}\n')

    states = [
        [0.0] * 8,
        [0.0] * 6 + [0.02, -0.02],
        [0.0] * 6 + [0.04, -0.04],
    ]
    rows = len(states)
    parquet_path = data_dir / "episode_000000.parquet"
    pd.DataFrame({
        "task_index": [0] * rows,
        "observation.state": states,
        "node_points_track": [[[[0.0, 0.0, 0.0]]]] * rows,
        "node_points_mask": [[False]] * rows,
        "depths_rel": [[[0.0]]] * rows,
        "far_background_mask": [[[False]]] * rows,
        "is_complete": [False] * rows,
        "gripper_uvd": [[[]]] * rows,
        "subtask_id": [0] * rows,
    }).to_parquet(parquet_path, index=False)

    pipeline = OfflinePipeline(PipelineConfig(
        dataset_dir=tmp_path,
        episode_selector={0: [0]},
        overwrite={
            "taskstructure": False,
            "node_points_track": False,
            "node_points_mask": False,
            "depths_rel": False,
            "far_background_mask": False,
            "is_complete": False,
            "gripper_uvd": False,
            "gripper_openness": True,
            "subtask_id": False,
        },
    ))
    pipeline.process_episodes([0], {0: object()})

    result = pq.read_table(parquet_path)
    assert result["gripper_openness"].to_pylist() == [0.0, 0.5, 1.0]
    assert str(result.schema.field("gripper_openness").type) == "float"
