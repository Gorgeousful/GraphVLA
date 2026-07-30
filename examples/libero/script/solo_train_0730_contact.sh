#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

max_steps="${MAX_STEPS:-30000}"
nproc_per_node="${NPROC_PER_NODE:-2}"

for suite_task in 6 7 8; do
    output_dir="examples/libero/result/0730-contact-${suite_task}"
    if compgen -G "${output_dir}/checkpoints/step_*.pt" > /dev/null; then
        echo "Refusing to overwrite existing checkpoints in ${output_dir}" >&2
        exit 1
    fi

    echo "Training LIBERO task ${suite_task} from scratch -> ${output_dir}"
    torchrun --nproc_per_node "$nproc_per_node" \
        examples/libero/script/solo_train_task.py \
        --suite-task "$suite_task" \
        --max-steps "$max_steps"

    if [[ "${SYNC_WANDB:-0}" == "1" ]]; then
        wandb sync "${output_dir}/wandb/latest-run"
    fi
done
