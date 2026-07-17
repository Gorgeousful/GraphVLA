from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.transform import (
    PromptFromTask,
    AddHorizon,
    RepackTransform,
    Normalize,
    CustomTransform,
    FlattenTransform,
    SubtaskBoundryPadding,
    FlipTransform
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

def load_norm_stats(dataset_dir: str | Path, level: str = "suite") -> dict[str, Any]:
    path = Path(dataset_dir) / "meta" / f"norm_stats_{level}.json"
    with path.open("r", encoding="utf-8") as f:
        norm_stats = json.load(f)

    file_level = norm_stats.get("level", "suite")
    if file_level != level:
        raise ValueError(f"Norm stats level mismatch: expected {level!r}, got {file_level!r} from {path}")
    return norm_stats


@dataclass
class DataConfig:
    dataset_dir: Path
    episodes: list[int] | None = None
    tasks: list[int] | None = None
    horizon: dict[str, list[int]] | None = None
    video_backend: str = "pyav"
    transforms: tuple[Any, ...] = ()
    out_transforms: tuple[Any, ...] = ()

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "episodes": self.episodes,
            "tasks": self.tasks,
            "horizon": self.horizon,
            "video_backend": self.video_backend,
            "transforms": self.transforms,
            "out_transforms": self.out_transforms,
        }


LIBERO_DATASET_DIR = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_31_no_noops_1.0.0_lerobot_10hz"
)

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
    "is_complete": "is_complete",
    "depths.depth_rel": "depths_rel",
    "node_points_track": "node_points_track",
    "node_points_mask": "node_points_mask",
    "gripper_uvd": "gripper_uvd",
}

LIBERO_HISTORY_HORIZON = 15
LIBERO_FUTURE_HORIZON = 8
LIBERO_HORIZON = {
    # "observation.images.image": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    # "observation.state": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    # "observation.images.wrist_image": list(range(-15, 1)),
    # "action": list(range(16)),
    # "observation.state": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "subtask_id": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "is_complete": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "node_points_track": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "node_points_mask": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "gripper_uvd": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
    "depths_rel": list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON+1)),
}

LIBERO_TRANSFORM = (
    RepackTransform(structure=LIBERO_REPACK),
    PromptFromTask(tasks=load_lerobot_tasks(LIBERO_DATASET_DIR)),
    AddHorizon(history_horizon=LIBERO_HISTORY_HORIZON, future_horizon=LIBERO_FUTURE_HORIZON),
    CustomTransform(mode="add_subtaskstructure", dataset_dir=LIBERO_DATASET_DIR),

    CustomTransform(mode="split_gripper_uvd"),
    FlattenTransform(fields=("depths.depth_rel", "gripper_d")),
    Normalize(
        norm_stats=load_norm_stats(LIBERO_DATASET_DIR, level="suite"), 
        use_quantiles=True, 
        quantile_to_neg_one_one=True
    ),
    SubtaskBoundryPadding(),

    FlipTransform(mode="horizontal"),
    CustomTransform(mode="build_model_input"),
)

LIBERO_OUT_TRANSFORM = (
    CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": load_norm_stats(LIBERO_DATASET_DIR, level="suite"),
            "use_quantiles": True,
            "quantile_to_neg_one_one": True,
            "height": 256,
            "width": 256,
            "sigmoid_is_complete": True,
        },
    ),
)

LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=LIBERO_DATASET_DIR,
    horizon=LIBERO_HORIZON,
    transforms=LIBERO_TRANSFORM,
    out_transforms=LIBERO_OUT_TRANSFORM,
)
