from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.libero.config.model_config import LIBERO_MODEL_CONFIG

from src.dataset.transform import (
    PromptFromTask,
    AddHorizon,
    CenterOnCurrentTCP,
    RandomCollapseNodePoints,
    RepackTransform,
    Normalize,
    CustomTransform,
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
        }

LIBERO_DATASET_DIR = os.environ.get(
    "LIBERO_DATASET_DIR",
    "/data0/luokang/dataset/luokang/lerobot/libero/libero_with_depth_6_7_8_0830_rel",
)
LIBERO_NORM_STATS_PATH = Path(LIBERO_DATASET_DIR) / "meta" / "norm_stats_suite.json"
LIBERO_ACTION_DELTA = bool(LIBERO_MODEL_CONFIG.action_delta)
LIBERO_ACTION_FIELD = "actions_camera" if LIBERO_ACTION_DELTA else "absolute_actions_camera"
LIBERO_ACTION_STATS_FIELD = "camera_action" if LIBERO_ACTION_DELTA else "absolute_camera_action"
LIBERO_POINT_COORDINATE_FRAME = LIBERO_MODEL_CONFIG.point_coordinate_frame
LIBERO_POINT_STATS_FIELD = (
    "tcp_relative_xyz" if LIBERO_POINT_COORDINATE_FRAME == "tcp_relative" else "camera_xyz"
)
LIBERO_POINT_TRANSFORMS = (
    (CenterOnCurrentTCP(),) if LIBERO_POINT_COORDINATE_FRAME == "tcp_relative" else ()
)
LIBERO_POINT_SHAPE_DROPOUT_PROB = 0.0 # baseline 0

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
}

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
    RandomCollapseNodePoints(probability=LIBERO_POINT_SHAPE_DROPOUT_PROB),

    Normalize(
        norm_stats=LIBERO_NORM_STATS_PATH,
        field_map={
            "node_points_xyz": LIBERO_POINT_STATS_FIELD,
            "gripper_points_xyz": LIBERO_POINT_STATS_FIELD,
            "action": LIBERO_ACTION_STATS_FIELD,
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
            "action_field": LIBERO_ACTION_STATS_FIELD,
            "point_stats_field": LIBERO_POINT_STATS_FIELD,
            "point_coordinate_frame": LIBERO_POINT_COORDINATE_FRAME,
        },
    ),
)

TASKS = None

LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=LIBERO_DATASET_DIR,
    horizon=LIBERO_HORIZON,
    load_videos=False,
    transforms=LIBERO_TRANSFORM,
    out_transforms=LIBERO_OUT_TRANSFORM,
    action_delta=LIBERO_ACTION_DELTA,
    tasks=TASKS
)
