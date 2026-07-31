#!/usr/bin/env python3
"""Hold the LIBERO end effector still while cycling the gripper and scoring contact."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
from websockets.exceptions import ConnectionClosed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/graphvla_libero_config")
libero_config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
libero_config_dir.mkdir(parents=True, exist_ok=True)
libero_config_file = libero_config_dir / "config.yaml"
if not libero_config_file.exists():
    benchmark_root = LIBERO_ROOT / "libero" / "libero"
    libero_config_file.write_text(
        "benchmark_root: {0}\n"
        "bddl_files: {0}/bddl_files\n"
        "init_states: {0}/init_files\n"
        "datasets: {1}/datasets\n"
        "assets: {0}/assets\n".format(benchmark_root, LIBERO_ROOT / "libero"),
        encoding="utf-8",
    )

from examples.libero.eval.client import (
    LIBERO_ENV_RESOLUTION,
    InferenceClient,
    _current_score,
    _draw_response_points,
    _draw_text_rgb,
    _dummy_action,
    _get_libero_env,
    _prepare_observation,
    benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cycle a stationary LIBERO gripper and record model contact scores."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--phase-steps", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to examples/libero/test/output/contact_gripper_score-<timestamp>.",
    )
    return parser.parse_args()


def query_server(
    client: InferenceClient,
    obs: dict[str, Any],
    env: Any,
    task_description: str,
) -> dict[str, Any]:
    # Discard the server action chunk so every physical step receives a fresh score.
    client.action_chunk.clear()
    client.action_frame_ids.clear()
    client.infer(_prepare_observation(obs, env), task_description)
    if client.last_response is None:
        raise RuntimeError("server returned no response")
    return client.last_response


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(steps, [row["contact_score"] for row in rows], label="Contact score", linewidth=2)
    ax.plot(steps, [row["complete_score"] for row in rows], label="Complete score", alpha=0.7)
    ax.plot(steps, [row["closedness"] for row in rows], label="Measured closedness", alpha=0.8)
    ax.step(steps, [row["command"] for row in rows], where="post", label="Gripper command", linestyle="--")
    ax.set(xlabel="Environment step", ylabel="Value", ylim=(-1.05, 1.05), title="Contact score during stationary gripper cycling")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264",
        pix_fmt_in="rgb24", output_params=["-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    if args.phase_steps <= 0 or args.cycles <= 0:
        raise ValueError("--phase-steps and --cycles must be positive")

    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "output" /
        f"contact_gripper_score-{datetime.now():%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state < len(initial_states):
        raise ValueError(f"init state must be in [0, {len(initial_states) - 1}]")

    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed, args.control_freq
    )
    client = InferenceClient(host=args.host, port=args.port)
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.init_state])
        for _ in range(args.settle_steps):
            action = _dummy_action(obs)
            action[-1] = -1.0
            obs, _, _, _ = env.step(action.tolist())

        client.reset_episode(env=env)
        query_server(client, obs, env, task_description)  # Initialize tracking and history.

        total_steps = args.cycles * 2 * args.phase_steps
        for step in range(total_steps):
            phase_index = step // args.phase_steps
            phase = "closed" if phase_index % 2 == 0 else "open"
            command = 1.0 if phase == "closed" else -1.0

            action = _dummy_action(obs)
            action[-1] = command
            obs, _, done, _ = env.step(action.tolist())
            try:
                response = query_server(client, obs, env, task_description)
            except (ConnectionClosed, ConnectionError, OSError) as error:
                print(f"server disconnected after {len(rows)} samples: {error}")
                break

            qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
            width = float(np.abs(qpos).sum())
            closedness = float(1.0 - 2.0 * np.clip(width / 0.08, 0.0, 1.0))
            contact_score = _current_score(response, "is_contact")
            complete_score = _current_score(response, "is_complete")
            rows.append({
                "step": step,
                "cycle": phase_index // 2,
                "phase": phase,
                "command": command,
                "qpos_left": float(qpos[0]),
                "qpos_right": float(qpos[1]),
                "closedness": closedness,
                "contact_score": np.nan if contact_score is None else contact_score,
                "complete_score": np.nan if complete_score is None else complete_score,
            })

            frame = _draw_response_points(
                np.ascontiguousarray(obs["agentview_image"][::-1, :]),
                response,
                None,
                mode="tracking",
            )
            _draw_text_rgb(frame, f"phase={phase} command={command:+.0f}", (8, 58))
            _draw_text_rgb(frame, f"closedness={closedness:+.3f}", (8, 78))
            frames.append(frame)
            print(
                f"step={step:03d} phase={phase:6s} command={command:+.0f} "
                f"closedness={closedness:+.3f} contact={contact_score}"
            )
            if done or client.episode_done:
                print("episode ended early; consider starting the server with --complete-threshold 1.1")
                break
    finally:
        client.close()
        env.close()

    if not rows:
        raise RuntimeError("test produced no samples")
    save_csv(rows, output_dir / "scores.csv")
    save_plot(rows, output_dir / "contact_score.png")
    save_video(frames, output_dir / "contact_gripper.mp4", args.video_fps)

    for phase in ("open", "closed"):
        scores = np.asarray(
            [row["contact_score"] for row in rows if row["phase"] == phase], dtype=np.float64
        )
        print(f"{phase} contact mean={np.nanmean(scores):.6f} std={np.nanstd(scores):.6f}")
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
