#!/usr/bin/env python3
"""Move task-6 pudding toward a stationary open gripper and score contact."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.test.contact_gripper_score import (
    query_server,
    save_csv,
    save_video,
)
from examples.libero.eval.client import (
    LIBERO_ENV_RESOLUTION,
    InferenceClient,
    _current_score,
    _draw_response_points,
    _draw_text_rgb,
    _dummy_action,
    _get_libero_env,
    benchmark,
)

PUDDING_JOINT = "chocolate_pudding_1_joint0"
PUDDING_BODY = "chocolate_pudding_1_main"
PUDDING_FIRST_LANGUAGE = (
    "put the chocolate pudding to the right of the plate and put the white mug on the plate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the task-6 pudding while holding the gripper open and record contact scores."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--phase-steps", type=int, default=30)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--approach-fraction", type=float, default=0.85,
        help="Fraction of the initial pudding-to-gripper XY offset traversed at closest approach.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, score_ax = plt.subplots(figsize=(11, 4.5))
    score_ax.plot(steps, [row["contact_score"] for row in rows], label="Contact score", linewidth=2)
    score_ax.plot(steps, [row["complete_score"] for row in rows], label="Complete score", alpha=0.65)
    score_ax.set(xlabel="Environment step", ylabel="Model score", ylim=(-0.02, 1.02))
    score_ax.grid(alpha=0.25)

    distance_ax = score_ax.twinx()
    distance_ax.plot(
        steps, [row["patient_gripper_distance"] for row in rows],
        color="tab:green", linestyle="--", label="Patient-gripper distance",
    )
    distance_ax.set_ylabel("World distance (m)")
    lines = score_ax.lines + distance_ax.lines
    score_ax.legend(lines, [line.get_label() for line in lines], loc="best")
    score_ax.set_title("Contact score while moving the patient toward an open gripper")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.phase_steps <= 0 or args.cycles <= 0:
        raise ValueError("--phase-steps and --cycles must be positive")
    if not 0.0 < args.approach_fraction <= 1.0:
        raise ValueError("--approach-fraction must be in (0, 1]")

    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "output" /
        f"contact_patient_motion-{datetime.now():%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.control_freq)
    client = InferenceClient(host=args.host, port=args.port)
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    interrupted_error: BaseException | None = None

    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.init_state])
        for _ in range(args.settle_steps):
            action = _dummy_action(obs)
            action[-1] = -1.0
            obs, _, _, _ = env.step(action.tolist())

        sim = env.env.sim if hasattr(env, "env") else env.sim
        initial_qpos = np.asarray(sim.data.get_joint_qpos(PUDDING_JOINT), dtype=np.float64).copy()
        initial_xyz = initial_qpos[:3].copy()
        gripper_xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
        closest_xyz = initial_xyz.copy()
        closest_xyz[:2] += args.approach_fraction * (gripper_xyz[:2] - initial_xyz[:2])

        client.reset_episode(env=env)
        query_server(client, obs, env, PUDDING_FIRST_LANGUAGE)

        for step in range(args.cycles * 2 * args.phase_steps):
            phase_index = step // args.phase_steps
            phase_step = step % args.phase_steps
            phase = "approach" if phase_index % 2 == 0 else "recede"
            progress = (phase_step + 1) / args.phase_steps
            alpha = progress if phase == "approach" else 1.0 - progress

            desired_qpos = initial_qpos.copy()
            desired_qpos[:3] = initial_xyz + alpha * (closest_xyz - initial_xyz)
            sim.data.set_joint_qpos(PUDDING_JOINT, desired_qpos)
            sim.forward()

            action = _dummy_action(obs)
            action[-1] = -1.0
            obs, _, done, _ = env.step(action.tolist())
            response = query_server(client, obs, env, PUDDING_FIRST_LANGUAGE)

            patient_xyz = np.asarray(sim.data.get_body_xpos(PUDDING_BODY), dtype=np.float64).copy()
            current_gripper_xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            distance = float(np.linalg.norm(patient_xyz - current_gripper_xyz))
            contact_score = _current_score(response, "is_contact")
            complete_score = _current_score(response, "is_complete")
            rows.append({
                "step": step,
                "cycle": phase_index // 2,
                "phase": phase,
                "path_alpha": alpha,
                "patient_x": float(patient_xyz[0]),
                "patient_y": float(patient_xyz[1]),
                "patient_z": float(patient_xyz[2]),
                "gripper_x": float(current_gripper_xyz[0]),
                "gripper_y": float(current_gripper_xyz[1]),
                "gripper_z": float(current_gripper_xyz[2]),
                "patient_gripper_distance": distance,
                "contact_score": np.nan if contact_score is None else contact_score,
                "complete_score": np.nan if complete_score is None else complete_score,
            })

            frame = _draw_response_points(
                np.ascontiguousarray(obs["agentview_image"][::-1, :]),
                response, None, mode="tracking",
            )
            _draw_text_rgb(frame, f"patient phase={phase} alpha={alpha:.2f}", (8, 58))
            _draw_text_rgb(frame, f"distance={distance:.3f} m gripper=open", (8, 78))
            frames.append(frame)
            print(
                f"step={step:03d} phase={phase:8s} alpha={alpha:.3f} "
                f"distance={distance:.4f} contact={contact_score}"
            )
            if done or client.episode_done:
                print("episode ended early; start the server with --complete-threshold 1.1")
                break
    except (ConnectionClosed, ConnectionError, KeyboardInterrupt, OSError) as error:
        interrupted_error = error
        print(f"test interrupted after {len(rows)} samples: {error}")
    finally:
        client.close()
        env.close()

    if not rows:
        if interrupted_error is not None:
            raise RuntimeError("test stopped before producing samples") from interrupted_error
        raise RuntimeError("test produced no samples")
    save_csv(rows, output_dir / "scores.csv")
    save_plot(rows, output_dir / "contact_score.png")
    save_video(frames, output_dir / "contact_patient_motion.mp4", args.video_fps)
    print(f"saved {len(rows)} samples to {output_dir}")


if __name__ == "__main__":
    main()
