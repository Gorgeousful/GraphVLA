"""32-point segmented object clouds, two observations, and 7D delta actions."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.cbf_code.model_config import LIBERO_MODEL_CONFIG
from src.dataset.transform import (
    AddHorizon,
    CustomTransform,
    Normalize,
    PromptFromTask,
    RepackTransform,
    SubtaskBoundryPadding,
)
from src.policy.cbf_code.data import CbFCodeBatchTransform


def _tasks(dataset_dir: str | Path) -> dict[int, str]:
    with (Path(dataset_dir) / "meta/tasks.jsonl").open(encoding="utf-8") as handle:
        return {int(row["task_index"]): str(row["task"]) for row in map(json.loads, handle)}


def _episodes(dataset_dir: str | Path, task_indices: list[int], count: int) -> list[int]:
    task_names = _tasks(dataset_dir)
    task_by_name = {name: index for index, name in task_names.items()}
    selected = {index: [] for index in task_indices}
    with (Path(dataset_dir) / "meta/episodes.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            task_index = task_by_name[str(row["tasks"][0])]
            if task_index in selected and len(selected[task_index]) < count:
                selected[task_index].append(int(row["episode_index"]))
    incomplete = {key: len(value) for key, value in selected.items() if len(value) != count}
    if incomplete:
        raise ValueError(f"CbF-Code expected {count} episodes per task, got {incomplete}")
    return [episode for task_index in task_indices for episode in selected[task_index]]


@dataclass
class DataConfig:
    dataset_dir: Path
    episodes: list[int] | None = None
    tasks: list[int] | None = None
    horizon: dict[str, list[int]] | None = None
    video_backend: str = "pyav"
    load_videos: bool = False
    feature_keys: tuple[str, ...] = ()
    load_episode_stats: bool = False
    transforms: tuple[Any, ...] = ()
    out_transforms: tuple[Any, ...] = ()
    action_delta: bool = True

    def to_kwargs(self):
        return {key: getattr(self, key) for key in (
            "dataset_dir", "episodes", "tasks", "horizon", "video_backend",
            "load_videos", "feature_keys", "load_episode_stats", "transforms", "out_transforms",
        )}


LIBERO_DATASET_DIR = os.environ.get(
    "LIBERO_DATASET_DIR",
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0904_20hz",
)
LIBERO_NORM_STATS_PATH = Path(LIBERO_DATASET_DIR) / "meta/norm_stats_suite.json"
# Established 0904 ID training split; the remaining three tasks are OOD evaluation.
LIBERO_TASKS = [0, 1, 2, 5, 7, 8, 9]
LIBERO_EPISODES = _episodes(LIBERO_DATASET_DIR, LIBERO_TASKS, 10)
LIBERO_OFFSETS = list(range(1 - LIBERO_MODEL_CONFIG.obs_steps,
                            1 - LIBERO_MODEL_CONFIG.obs_steps + LIBERO_MODEL_CONFIG.horizon))
LIBERO_HORIZON = {
    key: LIBERO_OFFSETS for key in (
        "subtask_id", "node_points_xyz", "valid_node_mask", "subtask_node_mask",
        "gripper_points_xyz", "observation.state", "action",
    )
}
LIBERO_REPACK = {
    "metadata": {"task_index": "task_index", "episode_index": "episode_index"},
    "subtask_id": "subtask_id",
    "node_points_xyz": "node_points_xyz",
    "valid_node_mask": "valid_node_mask",
    "subtask_node_mask": "subtask_node_mask",
    "gripper_points_xyz": "gripper_points_xyz",
    "state": "observation.state",
    "action": "action",
}
LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=Path(LIBERO_DATASET_DIR),
    episodes=LIBERO_EPISODES,
    tasks=LIBERO_TASKS,
    horizon=LIBERO_HORIZON,
    feature_keys=tuple(LIBERO_HORIZON),
    transforms=(
        RepackTransform(structure=LIBERO_REPACK),
        PromptFromTask(tasks=_tasks(LIBERO_DATASET_DIR)),
        AddHorizon(history_horizon=1, future_horizon=15, history_frames=(-1,)),
        CustomTransform(mode="add_subtaskstructure", dataset_dir=LIBERO_DATASET_DIR),
        SubtaskBoundryPadding(),
        CbFCodeBatchTransform(
            num_points=LIBERO_MODEL_CONFIG.num_points,
            obs_steps=LIBERO_MODEL_CONFIG.obs_steps,
            horizon=LIBERO_MODEL_CONFIG.horizon,
        ),
        # The original datasets use min/max normalization. Suite q01/q99 are
        # the available fixed LIBERO bounds and preserve the same [-1,1] contract.
        Normalize(
            norm_stats=LIBERO_NORM_STATS_PATH,
            field_map={"state": "state", "actions": "action"},
            use_quantiles=True,
            clip_quantiles=True,
        ),
    ),
    out_transforms=(
        CustomTransform(mode="build_model_output", extra={
            "norm_stats_path": LIBERO_NORM_STATS_PATH,
            "use_quantiles": True,
            "quantile_to_neg_one_one": True,
            "action_field": "action",
            "action_mode": "delta_action",
        }),
    ),
)
