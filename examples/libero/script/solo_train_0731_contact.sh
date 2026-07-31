#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
torchrun_bin="/data0/luokang/miniconda3/envs/robobrain/bin/torchrun"
max_steps="${MAX_STEPS:-30000}"
nproc_per_node="${NPROC_PER_NODE:-2}"
master_port="${MASTER_PORT:-29501}"

cd "$project_root"
for suite_task in 6 7 8; do
    dataset_dir="examples/libero/extra/libero_with_depth_${suite_task}"
    output_dir="examples/libero/result/0731-contact-${suite_task}"

    [[ -f "${dataset_dir}/meta/norm_stats_suite.json" ]] || {
        echo "Missing dataset or statistics: ${dataset_dir}" >&2
        exit 1
    }
    if [[ -e "$output_dir" || -L "$output_dir" ]]; then
        echo "Refusing to overwrite existing output directory: ${output_dir}" >&2
        exit 1
    fi

    echo "Training LIBERO task ${suite_task} from scratch for ${max_steps} steps -> ${output_dir}"
    "$torchrun_bin" \
        --nproc_per_node "$nproc_per_node" \
        --master_port "$master_port" \
        examples/libero/script/solo_train_task.py \
        --suite-task "$suite_task" \
        --dataset-dir "$dataset_dir" \
        --run-prefix "0731-contact" \
        --max-steps "$max_steps"
done
