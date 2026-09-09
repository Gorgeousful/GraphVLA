"""Data configuration for task-specific LIBERO ACT training."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.act.model_config import LIBERO_MODEL_CONFIG
from src.dataset.transform import Normalize, Unnormalize
from src.policy.act.data import ACTBatchTransform


@dataclass
class DataConfig:
    dataset_dir: Path
    episodes: list[int] | None = None
    tasks: list[int] | None = None
    horizon: dict[str, list[int]] | None = None
    video_backend: str = "pyav"
    load_videos: bool = True
    camera_keys: tuple[str, ...] = ()
    feature_keys: tuple[str, ...] | None = None
    load_episode_stats: bool = True
    transforms: tuple[Any, ...] = ()
    out_transforms: tuple[Any, ...] = ()
    action_delta: bool = True

    def __post_init__(self) -> None:
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("ACT camera_keys must be non-empty and unique")
        norm_stats_path = self.dataset_dir / "meta" / "norm_stats_suite.json"
        if not self.transforms:
            self.transforms = (
                ACTBatchTransform(image_keys=self.camera_keys),
                Normalize(
                    norm_stats=norm_stats_path,
                    field_map={"state": "state", "actions": "action"},
                    use_quantiles=False,
                ),
            )
        if not self.out_transforms:
            self.out_transforms = (
                Unnormalize(
                    norm_stats=norm_stats_path,
                    field_map={"action": "action"},
                    use_quantiles=False,
                ),
            )

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "episodes": self.episodes,
            "tasks": self.tasks,
            "horizon": self.horizon,
            "video_backend": self.video_backend,
            "load_videos": self.load_videos,
            "camera_keys": self.camera_keys,
            "feature_keys": self.feature_keys,
            "load_episode_stats": self.load_episode_stats,
            "transforms": self.transforms,
            "out_transforms": self.out_transforms,
        }


def select_task_episodes(
    dataset_dir: str | Path,
    task_index: int,
    count: int,
) -> list[int]:
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "meta" / "tasks.jsonl").open(encoding="utf-8") as handle:
        task_names = {
            int(row["task_index"]): str(row["task"])
            for row in map(json.loads, handle)
        }
    task_name = task_names[task_index]
    episodes = []
    with (dataset_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            if task_name in row["tasks"]:
                episodes.append(int(row["episode_index"]))
                if len(episodes) == count:
                    return episodes
    raise ValueError(f"Task {task_index} has only {len(episodes)} episodes, expected {count}")


LIBERO_DATASET_DIR = os.environ.get(
    "LIBERO_DATASET_DIR",
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz",
)
LIBERO_TASK_INDEX = 1
LIBERO_EPISODES = select_task_episodes(LIBERO_DATASET_DIR, LIBERO_TASK_INDEX, count=10)
LIBERO_HORIZON = {"action": list(range(LIBERO_MODEL_CONFIG.chunk_size))}

LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=Path(LIBERO_DATASET_DIR),
    episodes=LIBERO_EPISODES,
    tasks=None,
    horizon=LIBERO_HORIZON,
    load_videos=True,
    camera_keys=LIBERO_MODEL_CONFIG.camera_keys,
    feature_keys=("observation.state", "action"),
    load_episode_stats=False,
    action_delta=True,
)
