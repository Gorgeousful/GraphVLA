#!/usr/bin/env python3
"""Sample one frozen LIBERO observation repeatedly through a running server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.eval.client import (
    LIBERO_CAMERA_NAME,
    LIBERO_ENV_RESOLUTION,
    ObservationDeltaBuffer,
    _dummy_action,
    _get_libero_env,
    _prepare_observation,
    _to_libero_action,
    camera_matrices_from_env,
)
from src.common.observation_wire import encode_observation


cs = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--init-state-id", type=int, default=9)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--response-timeout", type=float, default=300.0)
    parser.add_argument(
        "--direction-tolerance-mm",
        type=float,
        default=1.0,
        help="Camera-X magnitude below this value is reported as neutral.",
    )
    args = parser.parse_args()
    if args.num_samples < 2:
        parser.error("--num-samples must be at least 2")
    if args.num_steps_wait < 0:
        parser.error("--num-steps-wait must be non-negative")
    return args


def build_frozen_request(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task id {args.task_id} is outside [0, {suite.n_tasks - 1}]")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state_id < len(initial_states):
        raise ValueError(
            f"init state id {args.init_state_id} is outside [0, {len(initial_states) - 1}]"
        )

    env, task_description = _get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        seed=args.env_seed,
        control_freq=args.control_freq,
        action_delta=False,
    )
    try:
        env.reset()
        observation = env.set_init_state(initial_states[args.init_state_id])
        for _ in range(args.num_steps_wait):
            hold_action = _dummy_action(observation, action_delta=False)
            controller_action = _to_libero_action(hold_action, action_delta=False)
            observation, _, _, _ = env.step(controller_action.tolist())

        frozen_observation = _prepare_observation(observation, env)
        intrinsic, extrinsic = camera_matrices_from_env(env, camera_name=LIBERO_CAMERA_NAME)
    finally:
        env.close()

    buffer = ObservationDeltaBuffer()
    buffer.append(frozen_observation)
    request = {
        "benchmark": "libero",
        "session_id": f"noise-test-{uuid.uuid4().hex[:8]}",
        "language": task_description,
        "reset": True,
        "return_model_input": True,
        **buffer.to_request_fields(intrinsic=intrinsic, extrinsic=extrinsic),
    }
    return request, np.asarray(frozen_observation["state"], dtype=np.float64), extrinsic


async def request_samples(
    uri: str,
    request: dict[str, Any],
    count: int,
    response_timeout: float,
) -> list[dict[str, Any]]:
    responses = []
    async with websockets.connect(
        uri,
        max_size=None,
        open_timeout=120,
        ping_interval=None,
        ping_timeout=None,
    ) as websocket:
        for sample_index in range(count):
            start = time.monotonic()
            await websocket.send(encode_observation(request))
            message = await asyncio.wait_for(websocket.recv(), timeout=response_timeout)
            response = json.loads(message)
            if "error" in response:
                raise RuntimeError(f"server sample {sample_index} failed: {response['error']}")
            response["_elapsed_seconds"] = time.monotonic() - start
            responses.append(response)
    return responses


def compare_model_input(reference: dict[str, Any], current: dict[str, Any]) -> tuple[float, int]:
    max_abs_diff = 0.0
    non_numeric_mismatch_count = 0
    if reference.keys() != current.keys():
        raise ValueError(
            f"model input keys changed: {sorted(reference)} vs {sorted(current)}"
        )
    for key in reference:
        left = np.asarray(reference[key])
        right = np.asarray(current[key])
        if left.shape != right.shape:
            raise ValueError(f"model input {key!r} shape changed: {left.shape} vs {right.shape}")
        if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
            if left.size:
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(left - right))))
        else:
            non_numeric_mismatch_count += int(np.count_nonzero(left != right))
    return max_abs_diff, non_numeric_mismatch_count


def summarize(
    responses: list[dict[str, Any]],
    state: np.ndarray,
    extrinsic: np.ndarray,
    tolerance_mm: float,
) -> None:
    reference_input = responses[0].get("model_input")
    if not isinstance(reference_input, dict):
        raise KeyError("server response does not contain model_input; check return_model_input support")

    camera_to_world = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    rows = []
    table = Table(title="Frozen-observation flow-noise samples")
    table.add_column("sample", justify="right")
    table.add_column("mean camera delta (mm)", justify="right")
    table.add_column("camera X", justify="center")
    table.add_column("input max diff", justify="right")
    table.add_column("non-numeric mismatch", justify="right")
    table.add_column("time (s)", justify="right")

    for sample_index, response in enumerate(responses):
        actions = np.asarray(response.get("action"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError(f"sample {sample_index} returned invalid action shape {actions.shape}")
        world_deltas = actions[:, :3] - state[None, :3]
        camera_deltas = world_deltas @ camera_to_world
        mean_delta_mm = camera_deltas.mean(axis=0) * 1000.0
        if mean_delta_mm[0] < -tolerance_mm:
            direction = "LEFT"
        elif mean_delta_mm[0] > tolerance_mm:
            direction = "RIGHT"
        else:
            direction = "NEUTRAL"

        model_input = response.get("model_input")
        if not isinstance(model_input, dict):
            raise KeyError(f"sample {sample_index} does not contain model_input")
        max_diff, non_numeric_mismatch = compare_model_input(reference_input, model_input)
        rows.append((mean_delta_mm, direction, max_diff, non_numeric_mismatch))
        table.add_row(
            str(sample_index),
            np.array2string(mean_delta_mm, precision=2, suppress_small=True),
            direction,
            f"{max_diff:.3e}",
            str(non_numeric_mismatch),
            f"{response['_elapsed_seconds']:.2f}",
        )

    cs.print(table)
    directions = [row[1] for row in rows]
    camera_x_mm = np.asarray([row[0][0] for row in rows])
    inputs_identical = all(row[2] <= 1e-6 and row[3] == 0 for row in rows)
    counts = {name: directions.count(name) for name in ("LEFT", "RIGHT", "NEUTRAL")}
    cs.print(
        f"Input tensors identical: {inputs_identical} | "
        f"directions={counts} | camera-X mean={camera_x_mm.mean():.2f} mm "
        f"std={camera_x_mm.std():.2f} mm range=[{camera_x_mm.min():.2f}, {camera_x_mm.max():.2f}] mm"
    )
    if not inputs_identical:
        cs.print("[yellow]Perception/model inputs changed, so this is not a noise-only comparison.[/yellow]")
    elif counts["LEFT"] and counts["RIGHT"]:
        cs.print("[yellow]The same model input produces both left and right motion across flow noises.[/yellow]")
    elif counts["RIGHT"] == len(rows):
        cs.print("[red]All sampled noises move right; this looks systematic for the frozen input.[/red]")
    elif counts["LEFT"] == len(rows):
        cs.print("[green]All sampled noises move left for the frozen input.[/green]")


def main() -> None:
    args = parse_args()
    np.random.seed(args.env_seed)
    request, state, extrinsic = build_frozen_request(args)
    cs.print(
        f"suite={args.task_suite_name} task={args.task_id} init_state={args.init_state_id} "
        f"wait_steps={args.num_steps_wait} samples={args.num_samples} state_xyz={state[:3].round(4).tolist()}"
    )
    responses = asyncio.run(
        request_samples(
            f"ws://{args.host}:{args.port}",
            request,
            args.num_samples,
            args.response_timeout,
        )
    )
    summarize(responses, state, extrinsic, args.direction_tolerance_mm)


if __name__ == "__main__":
    main()
