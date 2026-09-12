from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.graphpoint.model_config import LIBERO_MODEL_CONFIG
from src.dataset.transform import (
    AddHorizon,
    CenterOnCurrentTCP,
    CustomTransform,
    Normalize,
    PromptFromTask,
    RandomCollapseNodePoints,
    RepackTransform,
    SubtaskBoundryPadding,
)


def load_lerobot_tasks(dataset_dir: str | Path) -> dict[int, str]:
    path = Path(dataset_dir) / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def select_first_episodes_per_task(
    dataset_dir: str | Path,
    task_indices: list[int],
    episodes_per_task: int,
) -> list[int]:
    tasks = load_lerobot_tasks(dataset_dir)
    selected = {task_index: [] for task_index in task_indices}
    task_by_name = {task: task_index for task_index, task in tasks.items()}
    path = Path(dataset_dir) / "meta" / "episodes.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            task_index = task_by_name[str(row["tasks"][0])]
            if task_index in selected and len(selected[task_index]) < episodes_per_task:
                selected[task_index].append(int(row["episode_index"]))

    incomplete = {
        task_index: len(episodes)
        for task_index, episodes in selected.items()
        if len(episodes) != episodes_per_task
    }
    if incomplete:
        raise ValueError(
            f"Expected {episodes_per_task} episodes per task, got {incomplete}"
        )
    return [episode for task_index in task_indices for episode in selected[task_index]]


@dataclass
class DataConfig:
    dataset_dir: Path
    episodes: list[int] | None = None
    tasks: list[int] | None = None
    horizon: dict[str, list[int]] | None = None
    video_backend: str = "pyav"
    load_videos: bool = True
    transforms: tuple[Any, ...] = ()
    out_transforms: tuple[Any, ...] = ()
    action_delta: bool = True
    graphpoint_subtask_start: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "episodes": self.episodes,
            "tasks": self.tasks,
            "horizon": self.horizon,
            "video_backend": self.video_backend,
            "load_videos": self.load_videos,
            "transforms": self.transforms,
            "out_transforms": self.out_transforms,
            "graphpoint_subtask_start": self.graphpoint_subtask_start,
        }

LIBERO_DATASET_DIR = os.environ.get(
    "LIBERO_DATASET_DIR",
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz",
)
LIBERO_NORM_STATS_PATH = Path(LIBERO_DATASET_DIR) / "meta" / "norm_stats_suite.json"
LIBERO_ACTION_DELTA = bool(LIBERO_MODEL_CONFIG.action_delta)
# Raw action contains delta controller commands; abs_action uses future measured state.
LIBERO_ACTION_FIELD = "action"
LIBERO_POINT_COORDINATE_FRAME = LIBERO_MODEL_CONFIG.point_coordinate_frame
LIBERO_POINT_STATS_FIELD = (
    "tcp_relative_xyz" if LIBERO_POINT_COORDINATE_FRAME == "tcp_relative" else "camera_xyz"
)
LIBERO_POINT_TRANSFORMS = (
    (CenterOnCurrentTCP(),) if LIBERO_POINT_COORDINATE_FRAME == "tcp_relative" else ()
)
LIBERO_POINT_SHAPE_DROPOUT_PROB = 0.05  # baseline 0
LIBERO_TARGET_SHAPE_DROPOUT_PROB = 0.05
LIBERO_PATIENT_NEAREST_POINTS = 4

LIBERO_REPACK = {
    # "images.image": "observation.images.image",
    # "state": "observation.state",
    # "images.wrist_image": "observation.images.wrist_image",
    # "action": "action",
    "metadata": {
        "episode_index": "episode_index",
        "frame_index": "frame_index",
        "task_index": "task_index",
        "timestamp": "timestamp",
        "index": "index",
    },
    "subtask_id": "subtask_id",
    "subtask_progress": "subtask_progress",
    "node_points_xyz": "node_points_xyz",
    "valid_node_mask": "valid_node_mask",
    "subtask_node_mask": "subtask_node_mask",
    "gripper_points_xyz": "gripper_points_xyz",
    "action": LIBERO_ACTION_FIELD,
    "initial_node_points_xyz": "initial_node_points_xyz",
    "initial_valid_node_mask": "initial_valid_node_mask",
    "initial_subtask_node_mask": "initial_subtask_node_mask",
}
if LIBERO_MODEL_CONFIG.action_mode == "abs_action":
    LIBERO_REPACK["state"] = "observation.state"

LIBERO_HISTORY_FRAMES = list(LIBERO_MODEL_CONFIG.history_frames)
LIBERO_HISTORY_HORIZON = LIBERO_MODEL_CONFIG.history_horizon
LIBERO_FUTURE_HORIZON = LIBERO_MODEL_CONFIG.future_horizon
LIBERO_FRAME_OFFSETS = LIBERO_HISTORY_FRAMES + list(range(LIBERO_FUTURE_HORIZON + 1))
LIBERO_HORIZON = {
    # "observation.images.image": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    # "observation.state": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    # "observation.images.wrist_image": list(range(-15, 1)),
    # "action": list(range(16)),
    # "observation.state": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "subtask_id": list(LIBERO_FRAME_OFFSETS),
    "subtask_progress": list(LIBERO_FRAME_OFFSETS),
    "node_points_xyz": list(LIBERO_FRAME_OFFSETS),
    "valid_node_mask": list(LIBERO_FRAME_OFFSETS),
    "subtask_node_mask": list(LIBERO_FRAME_OFFSETS),
    "gripper_points_xyz": list(LIBERO_FRAME_OFFSETS),
    LIBERO_ACTION_FIELD: list(LIBERO_FRAME_OFFSETS),
}
if LIBERO_MODEL_CONFIG.action_mode == "abs_action":
    LIBERO_HORIZON["observation.state"] = list(LIBERO_FRAME_OFFSETS)

LIBERO_TRANSFORM = (
    RepackTransform(structure=LIBERO_REPACK),
    PromptFromTask(tasks=load_lerobot_tasks(LIBERO_DATASET_DIR)),
    AddHorizon(
        history_horizon=LIBERO_HISTORY_HORIZON,
        future_horizon=LIBERO_FUTURE_HORIZON,
        history_frames=LIBERO_HISTORY_FRAMES,
    ),
    CustomTransform(mode="add_subtaskstructure", dataset_dir=LIBERO_DATASET_DIR),
    *LIBERO_POINT_TRANSFORMS,
    RandomCollapseNodePoints(
        probability=LIBERO_POINT_SHAPE_DROPOUT_PROB,
        target_probability=LIBERO_TARGET_SHAPE_DROPOUT_PROB,
        patient_nearest_points=LIBERO_PATIENT_NEAREST_POINTS,
    ),

    Normalize(
        norm_stats=LIBERO_NORM_STATS_PATH,
        field_map={
            "node_points_xyz": LIBERO_POINT_STATS_FIELD,
            "gripper_points_xyz": LIBERO_POINT_STATS_FIELD,
            "initial_node_points_xyz": LIBERO_POINT_STATS_FIELD,
        },
        use_quantiles=True,
        quantile_to_neg_one_one=True,
    ),
    SubtaskBoundryPadding(),

    CustomTransform(
        mode="build_model_input",
        dataset_dir=LIBERO_DATASET_DIR,
        extra={
            "actor_point_indices": LIBERO_MODEL_CONFIG.actor_point_indices,
            "action_mode": LIBERO_MODEL_CONFIG.action_mode,
            "norm_stats_path": LIBERO_NORM_STATS_PATH,
            "point_stats_field": LIBERO_POINT_STATS_FIELD,
        },
    ),
)

LIBERO_OUT_TRANSFORM = (
    CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats_path": LIBERO_NORM_STATS_PATH,
            "use_quantiles": True,
            "quantile_to_neg_one_one": True,
            "action_field": "action" if LIBERO_MODEL_CONFIG.action_mode == "delta_action" else None,
            "action_mode": LIBERO_MODEL_CONFIG.action_mode,
            "point_stats_field": LIBERO_POINT_STATS_FIELD,
            "point_coordinate_frame": LIBERO_POINT_COORDINATE_FRAME,
        },
    ),
)

TASKS = [1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 16, 17]
EPISODES = select_first_episodes_per_task(
    LIBERO_DATASET_DIR,
    TASKS,
    episodes_per_task=10,
)

LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=LIBERO_DATASET_DIR,
    episodes=EPISODES,
    horizon=LIBERO_HORIZON,
    load_videos=False,
    transforms=LIBERO_TRANSFORM,
    out_transforms=LIBERO_OUT_TRANSFORM,
    action_delta=LIBERO_ACTION_DELTA,
    tasks=TASKS,
)
