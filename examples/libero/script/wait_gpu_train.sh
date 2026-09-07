#!/usr/bin/env bash
set -euo pipefail

# Configurable settings (memory is measured in MiB; 10240 MiB = 10 GiB).
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-10240}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
TRAIN_COMMAND=(
  torchrun --nproc_per_node 1
  --master_port 29501
  src/training/training.py
  --example libero
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi was not found." >&2
  exit 1
fi

if [[ ! "$GPU_ID" =~ ^[0-9]+$ || ! "$MIN_FREE_MEMORY_MB" =~ ^[0-9]+$ || ! "$POLL_INTERVAL" =~ ^[0-9]+$ || "$POLL_INTERVAL" -eq 0 ]]; then
  echo "Error: GPU_ID and MIN_FREE_MEMORY_MB must be non-negative integers; POLL_INTERVAL must be a positive integer." >&2
  exit 1
fi

echo "Waiting for GPU ${GPU_ID} to have at least ${MIN_FREE_MEMORY_MB} MiB free (polling every ${POLL_INTERVAL}s)."

while true; do
  if FREE_MEMORY_MB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
    FREE_MEMORY_MB="${FREE_MEMORY_MB//[[:space:]]/}"
    if [[ "$FREE_MEMORY_MB" =~ ^[0-9]+$ ]]; then
      echo "$(date '+%F %T') GPU ${GPU_ID}: ${FREE_MEMORY_MB} MiB free."
      if (( FREE_MEMORY_MB >= MIN_FREE_MEMORY_MB )); then
        echo "Memory threshold reached; starting training."
        cd "$REPO_ROOT"
        exec env CUDA_VISIBLE_DEVICES="$GPU_ID" WANDB_MODE=offline "${TRAIN_COMMAND[@]}"
      fi
    else
      echo "Warning: unexpected nvidia-smi output: ${FREE_MEMORY_MB}" >&2
    fi
  else
    echo "Warning: failed to query GPU ${GPU_ID}; retrying." >&2
  fi

  sleep "$POLL_INTERVAL"
done
