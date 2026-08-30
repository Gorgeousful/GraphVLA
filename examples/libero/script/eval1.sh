#!/usr/bin/env bash
set -Eeuo pipefail

WEIGHTS=(
  "examples/libero/result/0807-contact-basetcpfinger-cls4-rolechain-sam-7/checkpoints/step_30000.pt",
  "examples/libero/result/0807-contact-basetcpfinger-cls4-vggt-full-sam-7/checkpoints/step_30000.pt",
  "examples/libero/result/0807-contact-basetcpfinger-cls4-vggt-rolechain-sam-7/checkpoints/step_30000.pt"
)

# Format: "suite_name|task_ids|episodes_per_task".
# task_ids follows --tasks syntax, e.g. 1 4 6.
EVAL_CONFIGS=(
  "libero_swap_test|3|10"
)

PORT=8003
SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  for ((attempt = 1; attempt <= 180; attempt++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited before becoming ready." >&2
      return 1
    fi
    if python - "$PORT" >/dev/null 2>&1 <<'PY'
import asyncio
import json
import sys
import websockets

async def check():
    async with websockets.connect(f"ws://127.0.0.1:{sys.argv[1]}", open_timeout=1) as ws:
        await ws.send(json.dumps({"type": "server_info"}))
        response = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
        assert response.get("ckpt_path")

asyncio.run(check())
PY
    then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for the server on port $PORT." >&2
  return 1
}

failures=()
for weight in "${WEIGHTS[@]}"; do
  echo "===== Starting server: $weight ====="
  python -m script.server \
    --example libero \
    --ckpt-path "$weight" \
    --execute-chunk-len 5 \
    --progress-window 1 \
    --locator-scale 2.0 \
    --port "$PORT" \
    --devices '{"inference":"cuda:0","node_segmenter":"cuda:0","point_tracker":"cuda:0","node_locator":"cuda:1"}' \
    --sam-only \
    --locator-mode box &
  SERVER_PID=$!

  if ! wait_for_server; then
    failures+=("$weight|server")
    stop_server
    continue
  fi

  for config in "${EVAL_CONFIGS[@]}"; do
    IFS='|' read -r suite tasks episodes <<< "$config"
    read -ra task_args <<< "$tasks"
    echo "===== Evaluating: weight=$weight suite=$suite tasks=$tasks episodes=$episodes ====="
    if ! python -m examples.libero.eval.client \
      --task-suite-name "$suite" \
      --tasks "${task_args[@]}" \
      --num-trials-per-task "$episodes" \
      --max-steps 1000 \
      --control-freq 10 \
      --port "$PORT"; then
      failures+=("$weight|$suite|$tasks|$episodes")
    fi
  done

  stop_server
done

if ((${#failures[@]})); then
  printf 'Failed combinations:\n  %s\n' "${failures[@]}" >&2
  exit 1
fi

echo "All evaluation combinations completed successfully."
