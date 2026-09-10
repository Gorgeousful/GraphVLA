"""Data configuration for multi-task LIBERO official image U-Net Diffusion Policy training."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.dp.model_config import LIBERO_MODEL_CONFIG
from src.dataset.transform import Normalize, PromptFromTask, Unnormalize
from src.policy.dp.data import DiffusionPolicyBatchTransform


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
            raise ValueError("Diffusion Policy camera_keys must be non-empty and unique")
        norm_stats_path = self.dataset_dir / "meta" / "norm_stats_suite.json"
        if not self.transforms:
            self.transforms = (
                PromptFromTask(
                    tasks=load_lerobot_tasks(self.dataset_dir),
                    task_index_path=("task_index",),
                    output_key="language",
                ),
                DiffusionPolicyBatchTransform(image_keys=self.camera_keys),
                Normalize(
                    norm_stats=norm_stats_path,
                    field_map={"state": "state", "actions": "action"},
                    use_quantiles=True,
                    clip_quantiles=True,
                ),
            )
        if not self.out_transforms:
            self.out_transforms = (
                Unnormalize(
                    norm_stats=norm_stats_path,
                    field_map={"action": "action"},
                    use_quantiles=True,
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


def load_lerobot_tasks(dataset_dir: str | Path) -> dict[int, str]:
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "meta" / "tasks.jsonl").open(encoding="utf-8") as handle:
        return {
            int(row["task_index"]): str(row["task"])
            for row in map(json.loads, handle)
        }


def select_first_episodes_per_task(
    dataset_dir: str | Path,
    task_indices: list[int],
    count: int,
) -> list[int]:
    dataset_dir = Path(dataset_dir)
    task_names = load_lerobot_tasks(dataset_dir)
    selected = {task_index: [] for task_index in task_indices}
    task_by_name = {task: task_index for task_index, task in task_names.items()}
    with (dataset_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            task_index = task_by_name[str(row["tasks"][0])]
            if task_index in selected and len(selected[task_index]) < count:
                selected[task_index].append(int(row["episode_index"]))
    incomplete = {
        task_index: len(episodes)
        for task_index, episodes in selected.items()
        if len(episodes) != count
    }
    if incomplete:
        raise ValueError(f"Expected {count} episodes per task, got {incomplete}")
    return [episode for task_index in task_indices for episode in selected[task_index]]


LIBERO_DATASET_DIR = os.environ.get(
    "LIBERO_DATASET_DIR",
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz",
)
LIBERO_TASKS = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 16, 17]
LIBERO_EPISODES = select_first_episodes_per_task(
    LIBERO_DATASET_DIR,
    LIBERO_TASKS,
    count=10,
)
LIBERO_OBSERVATION_FRAMES = list(range(1 - LIBERO_MODEL_CONFIG.obs_steps, 1))
LIBERO_HORIZON = {
    "observation.state": LIBERO_OBSERVATION_FRAMES,
    **{
        camera_key: LIBERO_OBSERVATION_FRAMES
        for camera_key in LIBERO_MODEL_CONFIG.camera_keys
    },
    "action": list(
        range(
            1 - LIBERO_MODEL_CONFIG.obs_steps,
            LIBERO_MODEL_CONFIG.horizon + 1 - LIBERO_MODEL_CONFIG.obs_steps,
        )
    ),
}

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
