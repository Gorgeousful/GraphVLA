#!/usr/bin/env python3
"""Train LIBERO with an explicit actor CLS-token count and run name."""

from __future__ import annotations

import argparse
import sys
from copy import copy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cls-token-num", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, default=30_000)
    args = parser.parse_args()
    if args.cls_token_num <= 0:
        parser.error("--cls-token-num must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    from examples.libero.config.data_config import LIBERO_DATA_CONFIG
    from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
    from examples.libero.config.training_config import LIBERO_TRAINING_CONFIG
    from src.training.training import train

    model_config = copy(LIBERO_MODEL_CONFIG)
    model_config.cls_token_num = args.cls_token_num

    training_config = copy(LIBERO_TRAINING_CONFIG)
    training_config.max_steps = args.max_steps
    training_config.ckpt_path = None
    training_config.wandb_name = args.run_name

    train(LIBERO_DATA_CONFIG, model_config, training_config)


if __name__ == "__main__":
    main()
