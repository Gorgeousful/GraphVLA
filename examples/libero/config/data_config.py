from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dataset.transform import PromptFromTask, RepackTransform


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
    transforms: tuple[Any, ...] = ()


LIBERO_DATASET_DIR = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_all_no_noops_1.0.0_lerobot_10hz"
)

LIBERO_REPACK = {
    "images": {
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
    },
    "state": "observation.state",
    "action": "action",
    "metadata": {
        "episode_index": "episode_index",
        "frame_index": "frame_index",
        "task_index": "task_index",
        "timestamp": "timestamp",
        "index": "index",
    },
}

LIBERO_OPTIONAL_REPACK = {
    "grounding": "grounding",
    "subtask": "subtask",
    "phase": "phase",
    "focus": "focus",
}

LIBERO_HORIZON = {
    "observation.images.image": list(range(-9, 1)),
    "observation.images.wrist_image": list(range(-9, 1)),
    "observation.state": list(range(-9, 1)),
    "action": list(range(16)),
}

LIBERO_TRANSFORM = (
    RepackTransform(
        structure=LIBERO_REPACK,
        optional_structure=LIBERO_OPTIONAL_REPACK,
    ),
    PromptFromTask(tasks=load_lerobot_tasks(LIBERO_DATASET_DIR)),
)

LIBERO_DATA_CONFIG = DataConfig(
    dataset_dir=LIBERO_DATASET_DIR,
    horizon=LIBERO_HORIZON,
    transforms=LIBERO_TRANSFORM,
)
