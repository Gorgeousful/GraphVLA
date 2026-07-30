#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

checkpoint="examples/libero/result/0728/checkpoints/step_200000.pt"
if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
fi

for suite_task in 0 5 6 7 8; do
    output_dir="examples/libero/result/0728-${suite_task}"
    if compgen -G "${output_dir}/checkpoints/step_*.pt" > /dev/null; then
        echo "Refusing to overwrite existing checkpoints in ${output_dir}" >&2
        exit 1
    fi

    echo "Fine-tuning LIBERO task ${suite_task} -> ${output_dir}"
    torchrun --nproc_per_node 2 \
        examples/libero/script/finetune_task.py \
        --suite-task "$suite_task" \
        --max-steps 30000 \
        --ckpt-path "$checkpoint"

    if [[ "${SYNC_WANDB:-0}" == "1" ]]; then
        wandb sync "${output_dir}/wandb/latest-run"
    fi
done
