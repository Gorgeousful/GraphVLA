#!/usr/bin/env python3
"""Thin LIBERO evaluation client for the GraphVLA inference server.

The client owns benchmark rollout, observation history, and execution of
server-returned action chunks. Planning, perception, model inference, and action
recovery live in script/server.py.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import pathlib
import cv2
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

import numpy as np
from rich.console import Console
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)
import websockets


cs = Console()

LIBERO_ENV_RESOLUTION = 256
LIBERO_CAMERA_NAME = "agentview"

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output"


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8001
    control_freq: int = 20
    task_suite_name: str = "libero_10"
    tasks: list[int] | None = None
    num_steps_wait: int = 30
    num_trials_per_task: int = 5
    trials_init_state: list[int] | None = None
    max_steps: int | None = None
    seed: int = 42
    save_video: bool = True


class ObservationDeltaBuffer:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.metric_depths: list[np.ndarray] = []
        self.states: list[np.ndarray] = []

    def reset(self) -> None:
        self.images.clear()
        self.metric_depths.clear()
        self.states.clear()

    def append(self, observation: dict[str, Any]) -> None:
        self.images.append(np.asarray(observation["agentview_image"], dtype=np.uint8))
        self.metric_depths.append(np.asarray(observation["agentview_metric_depth"], dtype=np.float32))
        self.states.append(np.asarray(observation["state"], dtype=np.float64))

    def to_request_fields(self, *, intrinsic: np.ndarray, extrinsic: np.ndarray) -> dict[str, Any]:
        if not self.images:
            raise RuntimeError("observation delta buffer is empty")
        count = len(self.images)
        return {
            "observation.images.image": [image.tolist() for image in self.images],
            "observation.depth.metric": [depth.tolist() for depth in self.metric_depths],
            "observation.state": [state.tolist() for state in self.states],
            "camera.intrinsics": [np.asarray(intrinsic, dtype=np.float64).tolist()] * count,
            "camera.extrinsics": [np.asarray(extrinsic, dtype=np.float64).tolist()] * count,
        }


class InferenceClient:
    def __init__(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.pending_observations = ObservationDeltaBuffer()
        self.intrinsic: np.ndarray | None = None
        self.extrinsic: np.ndarray | None = None
        self.session_id = f"libero-{uuid.uuid4().hex[:8]}"
        self.action_chunk: list[list[float]] = []
        self.action_frame_ids: list[int] = []
        self.first_request = True
        self.last_response: dict[str, Any] | None = None
        self.last_action_frame_id: int | None = None
        self.episode_done = False

    def reset_episode(self, *, env: Any) -> None:
        self.pending_observations.reset()
        self.action_chunk.clear()
        self.action_frame_ids.clear()
        self.intrinsic, self.extrinsic = camera_matrices_from_env(env, camera_name=LIBERO_CAMERA_NAME)
        self.first_request = True
        self.last_response = None
        self.last_action_frame_id = None
        self.episode_done = False

    def infer(self, observation: dict[str, Any], task_description: str) -> np.ndarray:
        self.pending_observations.append(observation)
        if not self.action_chunk:
            response = self._call_server(task_description)
            self.last_response = response
            episode_done = response.get("episode_done", False)
            if not isinstance(episode_done, bool):
                raise TypeError(f"server episode_done must be a boolean, got {episode_done!r}")
            self.episode_done = episode_done
            self.action_chunk = self._validated_action_chunk(response)
            self.action_frame_ids = list(range(1, len(self.action_chunk) + 1))
        self.last_action_frame_id = self.action_frame_ids.pop(0)
        return np.asarray(self.action_chunk.pop(0), dtype=np.float32)

    def _call_server(self, task_description: str) -> dict[str, Any]:
        if self.intrinsic is None or self.extrinsic is None:
            raise RuntimeError("camera matrices are not initialized; call reset_episode first")
        request = {
            "benchmark": "libero",
            "session_id": self.session_id,
            "language": task_description,
            **self.pending_observations.to_request_fields(intrinsic=self.intrinsic, extrinsic=self.extrinsic),
        }
        if self.first_request:
            request["reset"] = True
            self.first_request = False

        uri = f"ws://{self.host}:{self.port}"
        response = asyncio.run(self._websocket_json(uri, request))
        if "error" in response:
            raise RuntimeError(response["error"])
        self.pending_observations.reset()
        return response

    def _validated_action_chunk(self, response: dict[str, Any]) -> list[list[float]]:
        if "action" not in response:
            raise KeyError("server response missing 'action'")
        action = response["action"]
        if not isinstance(action, list) or not action:
            raise ValueError(f"server returned empty or invalid action chunk: {action!r}")
        for index, item in enumerate(action):
            array = np.asarray(item, dtype=np.float32)
            if array.shape != (7,):
                raise ValueError(f"action[{index}] must be 7-D, got shape={array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"action[{index}] contains non-finite values: {array}")
        return action

    @staticmethod
    async def _websocket_json(uri: str, data: dict[str, Any]) -> dict[str, Any]:
        async with websockets.connect(uri, max_size=None, proxy=None) as websocket:
            await websocket.send(json.dumps(data))
            message = await websocket.recv()
        return json.loads(message)



def camera_matrices_from_env(env: Any, *, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    sim = env.env.sim if hasattr(env, "env") and hasattr(env.env, "sim") else env.sim
    height = getattr(env, "camera_heights", LIBERO_ENV_RESOLUTION)
    width = getattr(env, "camera_widths", LIBERO_ENV_RESOLUTION)
    if isinstance(height, (list, tuple, np.ndarray)):
        height = height[0]
    if isinstance(width, (list, tuple, np.ndarray)):
        width = width[0]
    height = int(height)
    width = int(width)

    intrinsic = get_camera_intrinsic_matrix(sim, camera_name, height, width)
    extrinsic = get_camera_extrinsic_matrix(sim, camera_name)
    return np.asarray(intrinsic, dtype=np.float64), np.asarray(extrinsic, dtype=np.float64)


def _get_libero_env(task: Any, resolution: int, seed: int, control_freq: int) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
        "control_delta": True,
        "control_freq": control_freq,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _prepare_observation(obs: dict[str, Any], env: Any) -> dict[str, Any]:
    sim = env.env.sim if hasattr(env, "env") and hasattr(env.env, "sim") else env.sim
    metric_depth = np.asarray(get_real_depth_map(sim, obs["agentview_depth"]), dtype=np.float32)
    if metric_depth.ndim == 3 and metric_depth.shape[-1] == 1:
        metric_depth = metric_depth[..., 0]
    if metric_depth.ndim != 2:
        raise ValueError(f"Expected agentview metric depth [H,W], got {metric_depth.shape}")
    return {
        "agentview_image": np.ascontiguousarray(obs["agentview_image"][::-1, :]),
        "agentview_metric_depth": np.ascontiguousarray(metric_depth[::-1, :]),
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }


def _dummy_action() -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[6] = -1.0
    return action


def _first_batch(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        return array[0]
    return array


def _draw_text_rgb(image: np.ndarray, text: str, xy: tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, xy, font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_text_rgb_right(
    image: np.ndarray,
    text: str,
    y: int,
    *,
    margin: int = 8,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(text, font, 0.48, 1)
    x = max(margin, image.shape[1] - margin - text_width)
    cv2.putText(image, text, (x, y), font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, 0.48, color, 1, cv2.LINE_AA)


def _current_score(response: dict[str, Any], name: str) -> float | None:
    value = response.get(name)
    if value is None:
        return None

    scores = _first_batch(value).astype(np.float32).reshape(-1)
    finite_scores = scores[np.isfinite(scores)]
    return None if finite_scores.size == 0 else float(np.max(finite_scores))


def _draw_response_scores(image: np.ndarray, response: dict[str, Any]) -> None:
    for name, y in (("is_complete", 18), ("is_contact", 38)):
        score = _current_score(response, name)
        text = "-" if score is None else f"{score:.2f}"
        color = (80, 255, 80) if score is not None and score >= 0.5 else (255, 255, 255)
        _draw_text_rgb_right(image, text, y, color=color)


def _draw_response_points(
    image_rgb: np.ndarray,
    response: dict[str, Any] | None,
    frame_id: int | None,
    *,
    mode: str,
    intrinsic: np.ndarray | None = None,
) -> np.ndarray:
    image = np.ascontiguousarray(image_rgb.copy())
    if response is None:
        return image
    if mode not in {"prediction", "tracking"}:
        raise ValueError(f"unknown visualization mode: {mode}")

    height, width = image.shape[:2]
    colors = {
        0: (255, 80, 40),
        1: (60, 60, 255),
        2: (255, 144, 30),
        3: (50, 205, 50),
        4: (0, 215, 255),
        5: (255, 0, 255),
        6: (255, 255, 0),
    }
    count = 0

    if mode == "tracking":
        if response.get("tracking_point") is not None and response.get("tracking_object_id") is not None:
            tracking_points = np.asarray(response["tracking_point"], dtype=np.float32).reshape(-1, 3)
            tracking_object_ids = np.asarray(response["tracking_object_id"], dtype=np.int64).reshape(-1)
            tracking_active = np.asarray(
                response.get("tracking_point_active", np.ones(len(tracking_points), dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            if not (len(tracking_points) == len(tracking_object_ids) == len(tracking_active)):
                raise ValueError("tracking visualization arrays must have equal lengths")
            draw_order = np.argsort(tracking_active, kind="stable")
            for point_index in draw_order:
                point = tracking_points[point_index]
                object_id = tracking_object_ids[point_index]
                is_active = tracking_active[point_index]
                if not np.isfinite(point[:2]).all():
                    continue
                x, y = np.rint(point[:2]).astype(int)
                if 0 <= x < width and 0 <= y < height:
                    color = colors.get(int(object_id), (160, 80, 160))
                    if not is_active or float(point[2]) <= 0.5:
                        color = (145, 145, 145)
                    cv2.circle(image, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
                    count += 1
        initial_count = 0
        if "initial_points" in response and response["initial_points"] is not None:
            initial_points = np.asarray(response["initial_points"], dtype=np.float32).reshape(-1, 2)
            initial_active = np.asarray(
                response.get("initial_point_active", np.ones(len(initial_points), dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            if len(initial_points) != len(initial_active):
                raise ValueError("initial-point visualization arrays must have equal lengths")
            draw_order = np.argsort(initial_active, kind="stable")
            for point_index in draw_order:
                point = initial_points[point_index]
                is_active = initial_active[point_index]
                if not np.isfinite(point).all():
                    continue
                x, y = np.rint(point).astype(int)
                if 0 <= x < width and 0 <= y < height:
                    cv2.drawMarker(
                        image,
                        (x, y),
                        (255, 230, 40) if is_active else (145, 145, 145),
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=10,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
                    initial_count += 1
        _draw_text_rgb(image, f"tracking in={count} initial={initial_count}", (8, 18))
        _draw_response_scores(image, response)
        return image

    if frame_id is not None and response.get("action_plan") is not None:
        action_plan = _first_batch(response["action_plan"]).astype(np.float32)
        future_index = int(frame_id) - 1
        if action_plan.ndim == 2 and action_plan.shape[1] == 7 and 0 <= future_index < len(action_plan):
            action = action_plan[future_index]
            _draw_text_rgb(
                image,
                f"camera delta xyz={np.round(action[:3], 3).tolist()}",
                (8, 38),
            )
            _draw_text_rgb(
                image,
                f"camera delta rot={np.round(action[3:6], 3).tolist()} grip={action[6]:.2f}",
                (8, 58),
            )

    label_frame = "-" if frame_id is None else str(frame_id)
    _draw_text_rgb(image, f"prediction action f={label_frame}", (8, 18))
    _draw_response_scores(image, response)
    return image


def _save_video_ffmpeg(frames_rgb: list[np.ndarray], save_path: Path, *, fps: float = 10.0) -> None:
    if not frames_rgb:
        return
    first = np.asarray(frames_rgb[0], dtype=np.uint8)
    h, w = first.shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
            "-pix_fmt", "yuv420p",
            str(save_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    pipe_broken = False
    try:
        for frame in frames_rgb:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.shape[:2] != (h, w):
                raise ValueError(f"video frame shape mismatch: expected {(h, w)}, got {frame.shape[:2]}")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"video frame must have shape HxWx3, got {frame.shape}")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pipe_broken = True
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pipe_broken = True

    stderr = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if pipe_broken or proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace") if stderr else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed while saving {save_path}: {message}")
    cs.print(f"saved video to: {save_path} ({len(frames_rgb)} frames, {fps:.1f} fps)")


def _goal_progress(env: Any) -> tuple[int, int, float, list[int], list[str]]:
    problem_env = env.env if hasattr(env, "env") else env
    goal_state = problem_env.parsed_problem["goal_state"]
    goal_results = [bool(problem_env._eval_predicate(state)) for state in goal_state]
    completed_subtasks = [index + 1 for index, complete in enumerate(goal_results) if complete]
    completed_goal_states = [str(state) for state, complete in zip(goal_state, goal_results) if complete]
    completed_goals = len(completed_subtasks)
    total_goals = len(goal_state)
    progress = float(completed_goals) / float(total_goals) if total_goals else 0.0
    return completed_goals, total_goals, progress, completed_subtasks, completed_goal_states


def _default_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name in {"libero_10", "libero_10_swap", "libero_custom", "libero_swap_test"}:
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat = quat / np.linalg.norm(quat)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3]) / den).astype(np.float32)


def _parse_tasks(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate GraphVLA through the observation-driven inference server on LIBERO.")
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--control-freq", type=int, default=Args.control_freq)
    parser.add_argument("--task-suite-name", default=Args.task_suite_name)
    parser.add_argument("--tasks", type=_parse_tasks, default=Args.tasks)
    parser.add_argument("--num-steps-wait", type=int, default=Args.num_steps_wait)
    parser.add_argument("--num-trials-per-task", type=int, default=Args.num_trials_per_task)
    parser.add_argument("--trials-init-state", type=int, nargs="+", default=Args.trials_init_state)
    parser.add_argument("--max-steps", type=int, default=Args.max_steps)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--no-save-video", action="store_true")
    ns = parser.parse_args()
    if ns.control_freq <= 0:
        parser.error("--control-freq must be positive")
    return Args(
        host=ns.host,
        control_freq=ns.control_freq,
        port=ns.port,
        task_suite_name=ns.task_suite_name,
        tasks=ns.tasks,
        num_steps_wait=ns.num_steps_wait,
        num_trials_per_task=ns.num_trials_per_task,
        trials_init_state=ns.trials_init_state,
        max_steps=ns.max_steps,
        seed=ns.seed,
        save_video=not ns.no_save_video,
    )


def main() -> None:
    from libero.libero import benchmark

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cs.print(args, markup=False)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = args.tasks if args.tasks is not None else list(range(num_tasks_in_suite))
    for task_id in task_ids:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(f"task id {task_id} out of range for {args.task_suite_name}: 0-{num_tasks_in_suite - 1}")
    max_steps = args.max_steps if args.max_steps is not None else _default_max_steps(args.task_suite_name)

    timestamp = datetime.now().strftime("%m%d-%H%M")
    suite_output_dir = DEFAULT_OUTPUT_DIR / f"{args.task_suite_name}-{timestamp}"
    video_dir = suite_output_dir / "videos"
    result_path = suite_output_dir / "result.json"
    video_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(host=args.host, port=args.port)
    total_episodes = 0
    total_successes = 0
    total_progress = 0.0
    task_results: list[dict[str, Any]] = []

    for task_order, task_id in enumerate(task_ids, start=1):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        init_state_ids = (
            args.trials_init_state
            if args.trials_init_state is not None
            else list(range(args.num_trials_per_task))
        )
        invalid_init_state_ids = [
            state_id for state_id in init_state_ids
            if state_id < 0 or state_id >= len(initial_states)
        ]
        if invalid_init_state_ids:
            raise ValueError(
                f"initial state indices {invalid_init_state_ids} out of range for task {task_id}; "
                f"available range is 0-{len(initial_states) - 1}"
            )
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            seed=args.seed,
            control_freq=args.control_freq,
        )
        try:
            total_goals = len(env.env.parsed_problem["goal_state"])
            task_episodes = 0
            task_successes = 0
            task_progress = 0.0
            episode_results: list[dict[str, Any]] = []

            for episode_idx, init_state_id in enumerate(init_state_ids):
                logging.info("Task %s episode %s: %s", task_id, episode_idx, task_description)
                cs.print(
                    f"[cyan]task {task_order}/{len(task_ids)}[/cyan] "
                    f"id={task_id} episode {episode_idx + 1}/{len(init_state_ids)} "
                    f"init_state={init_state_id}: "
                    f"{task_description}"
                )
                env.reset()
                obs = env.set_init_state(initial_states[init_state_id])
                client.reset_episode(env=env)
                prediction_images = []
                tracking_images = []
                done = False
                server_done = False
                interrupted = False

                try:
                    for step in range(max_steps + args.num_steps_wait):
                        if step < args.num_steps_wait:
                            cs.print(
                                f"[dim]task {task_order}/{len(task_ids)} id={task_id} "
                                f"episode {episode_idx + 1}/{len(init_state_ids)} "
                                f"wait_step {step + 1}/{args.num_steps_wait}[/dim]"
                            )
                            action = _dummy_action()
                        else:
                            policy_step = step - args.num_steps_wait + 1
                            cs.print(
                                f"[dim]task {task_order}/{len(task_ids)} id={task_id} "
                                f"episode {episode_idx + 1}/{len(init_state_ids)} "
                                f"step {policy_step}/{max_steps}[/dim]"
                            )
                            action = client.infer(_prepare_observation(obs, env), task_description)
                            if client.episode_done:
                                server_done = True
                                cs.print(
                                    f"[green]server completed episode at policy step {policy_step}[/green]"
                                )
                                break

                        frame = np.ascontiguousarray(obs["agentview_image"][::-1, :])
                        prediction_images.append(
                            _draw_response_points(
                                frame,
                                client.last_response,
                                client.last_action_frame_id,
                                mode="prediction",
                                intrinsic=client.intrinsic,
                            )
                        )
                        tracking_images.append(
                            _draw_response_points(
                                frame,
                                client.last_response,
                                client.last_action_frame_id,
                                mode="tracking",
                            )
                        )
                        obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
                        if done:
                            break

                except KeyboardInterrupt:
                    interrupted = True
                    cs.print(
                        f"[yellow]task={task_id} episode={episode_idx} interrupted; "
                        "saving current progress and continuing[/yellow]"
                    )

                task_episodes += 1
                total_episodes += 1
                env_success = bool(done) and not interrupted
                if env_success:
                    task_successes += 1
                    total_successes += 1

                (
                    completed_goals,
                    total_goals,
                    progress,
                    completed_subtasks,
                    completed_goal_states,
                ) = _goal_progress(env)
                task_progress += progress
                total_progress += progress

                if args.save_video:
                    suffix = "interrupted" if interrupted else ("success" if env_success else "failure")
                    video_stem = f"task_{task_id:03d}_ep_{episode_idx:03d}_{suffix}"
                    _save_video_ffmpeg(prediction_images, video_dir / f"{video_stem}_prediction.mp4", fps=float(args.control_freq))
                    _save_video_ffmpeg(tracking_images, video_dir / f"{video_stem}_tracking.mp4", fps=float(args.control_freq))

                episode_results.append({
                    "episode_id": episode_idx,
                    "init_state_id": init_state_id,
                    "success": env_success,
                    "interrupted": interrupted,
                    "progress": progress,
                    "completed_subtasks": completed_subtasks,
                    "completed_goals": completed_goal_states,
                })

                cs.print(
                    f"task={task_id} episode={episode_idx} success={env_success} env_done={done} server_done={server_done} "
                    f"progress={completed_goals}/{total_goals} "
                    f"completed_subtasks={completed_subtasks} "
                    f"completed_goals={completed_goal_states}"
                )

            task_result = {
                "task_id": task_id,
                "task_desc": task_description,
                "total_goals": total_goals,
                "success_rate": float(task_successes) / float(task_episodes),
                "progress_rate": float(task_progress) / float(task_episodes),
                "num_episodes": task_episodes,
                "episodes": episode_results,
            }
            task_results.append(task_result)
        finally:
            env.close()

    result = {
        "task_suite": args.task_suite_name,
        "success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
        "progress_rate": float(total_progress) / float(total_episodes) if total_episodes else 0.0,
        "total_episodes": total_episodes,
        "tasks": task_results,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cs.print(f"saved results to: {result_path}")


if __name__ == "__main__":
    main()
