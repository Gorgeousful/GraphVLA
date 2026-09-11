#!/usr/bin/env bash
set -euo pipefail
# Connect to a running policy server. Override defaults with trailing CLI arguments.
# Task IDs follow alphabetical task-name order; split matches the MimicLabs suite.
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0904 --tasks 0 1 2 5 7 8 9 \
  --num-trials-per-task 30 --max-steps 1000 --control-freq 20 \
  --port 8001 --num-workers 3 "$@"
python -m examples.libero.eval.client \
  --task-suite-name libero_custom_0904 --tasks 3 4 6 \
  --num-trials-per-task 30 --max-steps 1000 --control-freq 20 \
  --port 8001 --num-workers 3 "$@"
