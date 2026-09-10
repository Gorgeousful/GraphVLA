"""Shared multi-task split; precomputed scene XYZ, state history, future actions."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.dp3.model_config import LIBERO_MODEL_CONFIG
from src.dataset.transform import Normalize, PromptFromTask, Unnormalize
from src.policy.dp3.data import DP3BatchTransform


@dataclass
class DataConfig:
    dataset_dir: Path
    episodes: list[int] | None = None
    tasks: list[int] | None = None
    horizon: dict[str, list[int]] | None = None
    video_backend: str = "pyav"
    load_videos: bool = False
    camera_keys: tuple[str, ...] = ()
    feature_keys: tuple[str, ...] = ("observation.point_cloud", "observation.state", "action")
    load_episode_stats: bool = False
    transforms: tuple[Any, ...] = ()
    out_transforms: tuple[Any, ...] = ()
    action_delta: bool = True
    point_cloud_max_candidates: int = 8192

    def to_kwargs(self):
        return {name: getattr(self, name) for name in (
            "dataset_dir", "episodes", "tasks", "horizon", "video_backend", "load_videos",
            "camera_keys", "feature_keys", "load_episode_stats", "transforms", "out_transforms",
        )}


LIBERO_DATASET_DIR = os.environ.get(
    "LIBERO_DATASET_DIR", "/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz",
)
LIBERO_TASKS = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 16, 17]
with (Path(LIBERO_DATASET_DIR) / "meta/tasks.jsonl").open() as handle:
    TASK_NAMES = {int(row["task_index"]): str(row["task"]) for row in map(json.loads, handle)}
TASK_BY_NAME = {name: index for index, name in TASK_NAMES.items()}
SELECTED_EPISODES = {index: [] for index in LIBERO_TASKS}
with (Path(LIBERO_DATASET_DIR) / "meta/episodes.jsonl").open() as handle:
    for row in map(json.loads, handle):
        task_index = TASK_BY_NAME[str(row["tasks"][0])]
        if task_index in SELECTED_EPISODES and len(SELECTED_EPISODES[task_index]) < 10:
            SELECTED_EPISODES[task_index].append(int(row["episode_index"]))
if any(len(episodes) != 10 for episodes in SELECTED_EPISODES.values()):
    raise ValueError("DP3 requires 10 episodes per selected task")
LIBERO_NORM_STATS_PATH = Path(LIBERO_DATASET_DIR) / "meta/norm_stats_suite.json"
LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=Path(LIBERO_DATASET_DIR),
    episodes=[episode for episodes in SELECTED_EPISODES.values() for episode in episodes],
    tasks=LIBERO_TASKS,
    horizon={
        "observation.point_cloud": list(range(1 - LIBERO_MODEL_CONFIG.obs_steps, 1)),
        "observation.state": list(range(1 - LIBERO_MODEL_CONFIG.obs_steps, 1)),
        "action": list(range(1 - LIBERO_MODEL_CONFIG.obs_steps,
                             1 - LIBERO_MODEL_CONFIG.obs_steps + LIBERO_MODEL_CONFIG.horizon)),
    },
    transforms=(
        PromptFromTask(tasks=TASK_NAMES, task_index_path=("task_index",), output_key="language"),
        DP3BatchTransform(num_points=LIBERO_MODEL_CONFIG.num_points, obs_steps=LIBERO_MODEL_CONFIG.obs_steps),
        Normalize(norm_stats=LIBERO_NORM_STATS_PATH,
                  field_map={"state": "state", "actions": "action"},
                  use_quantiles=True, clip_quantiles=True),
    ),
    out_transforms=(Unnormalize(norm_stats=LIBERO_NORM_STATS_PATH,
                               field_map={"action": "action"}, use_quantiles=True),),
)
