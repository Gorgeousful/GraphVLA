"""All-task tracked point slots; no subtask progress or subtask-mask supervision."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.act.data_config import load_lerobot_tasks, select_first_episodes_per_task
from examples.libero.config.point_policy.model_config import LIBERO_MODEL_CONFIG
from src.dataset.transform import PromptFromTask
from src.policy.point_policy.data import PointPolicyTransform, PointPolicyOutputTransform


@dataclass
class DataConfig:
    dataset_dir: Path
    episodes: list[int] | None = None
    tasks: list[int] | None = None
    horizon: dict[str, list[int]] | None = None
    load_videos: bool = False
    video_backend: str = "pyav"
    feature_keys: tuple[str, ...] = ("node_points_xyz_track", "valid_node_mask", "gripper_points_xyz", "action")
    load_episode_stats: bool = False
    transforms: tuple[Any, ...] = ()
    out_transforms: tuple[Any, ...] = ()
    action_delta: bool = False

    def to_kwargs(self):
        return {k: getattr(self, k) for k in (
            "dataset_dir", "episodes", "tasks", "horizon", "load_videos", "video_backend",
            "feature_keys", "load_episode_stats", "transforms", "out_transforms")}


LIBERO_DATASET_DIR = os.environ.get("LIBERO_DATASET_DIR", "/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz")
LIBERO_TASKS = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 16, 17]
LIBERO_EPISODES = select_first_episodes_per_task(LIBERO_DATASET_DIR, LIBERO_TASKS, count=10)
LIBERO_NORM_STATS_PATH = Path(LIBERO_DATASET_DIR) / "meta" / "norm_stats_suite.json"
LIBERO_HISTORY = list(LIBERO_MODEL_CONFIG.history_frames) + [0]
LIBERO_HORIZON = {
    "node_points_xyz_track": LIBERO_HISTORY,
    "valid_node_mask": LIBERO_HISTORY,
    "gripper_points_xyz": LIBERO_HISTORY + list(range(1, LIBERO_MODEL_CONFIG.future_horizon + 1)),
    "action": list(range(LIBERO_MODEL_CONFIG.future_horizon)),
}
LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=Path(LIBERO_DATASET_DIR), episodes=LIBERO_EPISODES, tasks=LIBERO_TASKS,
    horizon=LIBERO_HORIZON,
    transforms=(
        PromptFromTask(tasks=load_lerobot_tasks(LIBERO_DATASET_DIR), task_index_path=("task_index",), output_key="language"),
        PointPolicyTransform(LIBERO_NORM_STATS_PATH, LIBERO_MODEL_CONFIG.history_horizon,
                             LIBERO_MODEL_CONFIG.future_horizon, LIBERO_MODEL_CONFIG.actor_point_indices,
                             LIBERO_MODEL_CONFIG.num_points, LIBERO_MODEL_CONFIG.max_objects),
    ),
    out_transforms=(PointPolicyOutputTransform(LIBERO_NORM_STATS_PATH),),
)
