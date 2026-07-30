#!/usr/bin/env python3
"""Fine-tune one LIBERO task from the shared 0728 200k checkpoint."""

from __future__ import annotations

import argparse
import sys
from copy import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.config.data_config import LIBERO_DATA_CONFIG
from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
from examples.libero.config.training_config import LIBERO_TRAINING_CONFIG
from src.training.training import train


SUITE_TO_DATASET_TASK = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4}
DEFAULT_CHECKPOINT = Path(
    "examples/libero/result/0728/checkpoints/step_200000.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-task", type=int, required=True,
        choices=tuple(SUITE_TO_DATASET_TASK),
    )
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--save-dir", type=Path, default=Path("examples/libero/result")
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not args.ckpt_path.is_file():
        parser.error(f"checkpoint does not exist: {args.ckpt_path}")

    data_config = copy(LIBERO_DATA_CONFIG)
    data_config.tasks = [SUITE_TO_DATASET_TASK[args.suite_task]]

    training_config = copy(LIBERO_TRAINING_CONFIG)
    training_config.resume = False
    training_config.max_steps = args.max_steps
    training_config.ckpt_path = args.ckpt_path
    training_config.save_dir = args.save_dir
    training_config.wandb_name = f"0728-{args.suite_task}"

    train(data_config, LIBERO_MODEL_CONFIG, training_config)


if __name__ == "__main__":
    main()
