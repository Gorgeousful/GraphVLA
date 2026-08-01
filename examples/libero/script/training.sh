#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_GPU="${TRAIN_GPU:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-30000}"
TRAIN_WANDB_MODE="${TRAIN_WANDB_MODE:-offline}"

run_training() {
    local cls_tokens="$1"
    local run_name="$2"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
    WANDB_MODE="$TRAIN_WANDB_MODE" \
    torchrun --standalone --nproc_per_node=1 \
        examples/libero/script/train_cls_tokens.py \
        --cls-token-num "$cls_tokens" \
        --run-name "$run_name" \
        --max-steps "$TRAIN_STEPS"
}

run_training 1 "0801-contact-finger-5point-cls1"
run_training 4 "0801-contact-finger-5point-cls4"
