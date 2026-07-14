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
from pathlib import Path
from typing import Any

LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from libero.libero import benchmark
import numpy as np
from rich.console import Console
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
import websockets

from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
from src.common.geom_utils import rot_transform

cs = Console()

LIBERO_DELTA_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
LIBERO_CAMERA_NAME = "agentview"


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8001
    task_suite_name: str = "libero_10"
    tasks: list[int] | None = None
    num_steps_wait: int = 30
    num_trials_per_task: int = 5
    max_steps: int | None = None
    video_out_path: str = "examples/libero/eval/output/videos"
    result_out_path: str = "examples/libero/eval/output/result.json"
    seed: int = 42
    save_video: bool = True
    control_delta: bool = False
    execute_chunk_len: int = 16


class ObservationDeltaBuffer:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.states: list[np.ndarray] = []

    def reset(self) -> None:
        self.images.clear()
        self.states.clear()

    def append(self, observation: dict[str, Any]) -> None:
        self.images.append(np.asarray(observation["agentview_image"], dtype=np.uint8))
        self.states.append(np.asarray(observation["state"], dtype=np.float64))

    def to_request_fields(self, *, intrinsic: np.ndarray, extrinsic: np.ndarray) -> dict[str, Any]:
        if not self.images:
            raise RuntimeError("observation delta buffer is empty")
        count = len(self.images)
        return {
            "observation.images.image": [image.tolist() for image in self.images],
            "observation.state": [state.tolist() for state in self.states],
            "camera.intrinsics": [np.asarray(intrinsic, dtype=np.float64).tolist()] * count,
            "camera.extrinsics": [np.asarray(extrinsic, dtype=np.float64).tolist()] * count,
        }


class InferenceClient:
    def __init__(self, *, host: str, port: int, execute_chunk_len: int = 1) -> None:
        self.host = host
        self.port = port
        self.future_horizon = int(LIBERO_MODEL_CONFIG.max_frame)
        if execute_chunk_len < 1 or execute_chunk_len > self.future_horizon:
            raise ValueError(f"execute_chunk_len must be in [1, {self.future_horizon}], got {execute_chunk_len}")
        self.execute_chunk_len = int(execute_chunk_len)
        self.pending_observations = ObservationDeltaBuffer()
        self.intrinsic: np.ndarray | None = None
        self.extrinsic: np.ndarray | None = None
        self.session_id = ""
        self.action_chunk: list[list[float]] = []
        self.action_frame_ids: list[int] = []
        self.first_request = True
        self.last_response: dict[str, Any] | None = None
        self.last_action_frame_id: int | None = None

    def reset_episode(self, *, task_id: int, episode_idx: int, env: Any) -> None:
        self.pending_observations.reset()
        self.action_chunk.clear()
        self.action_frame_ids.clear()
        self.intrinsic, self.extrinsic = camera_matrices_from_env(env, camera_name=LIBERO_CAMERA_NAME)
        self.session_id = f"libero-task{task_id}-ep{episode_idx}-{uuid.uuid4().hex[:8]}"
        self.first_request = True
        self.last_response = None
        self.last_action_frame_id = None

    def infer(self, observation: dict[str, Any], task_description: str) -> np.ndarray:
        self.pending_observations.append(observation)
        if not self.action_chunk:
            response = self._call_server(task_description)
            self.last_response = response
            action_chunk = self._validated_action_chunk(response)
            self.action_chunk = action_chunk[: self.execute_chunk_len]
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
        if len(action) != self.future_horizon:
            raise ValueError(f"expected action chunk length {self.future_horizon}, got {len(action)}")
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


def _get_libero_env(task: Any, resolution: int, seed: int, *, control_delta: bool) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "control_delta": control_delta,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _prepare_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "agentview_image": np.ascontiguousarray(obs["agentview_image"][::-1, :]),
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }


def transform_hand_to_gripper(hand_pos: np.ndarray, hand_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand_pose = np.eye(4, dtype=np.float64)
    hand_pose[:3, 3] = np.asarray(hand_pos, dtype=np.float64)
    hand_pose[:3, :3] = rot_transform(
        np.asarray(hand_quat, dtype=np.float64),
        input_format="quat",
        target_format="matrix",
    )
    local_rotation = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    gripper_pose = hand_pose @ local_rotation
    gripper_quat = rot_transform(gripper_pose[:3, :3], input_format="matrix", target_format="quat")
    return gripper_pose[:3, 3].astype(np.float32), gripper_quat.astype(np.float32)


def _dummy_action(obs: dict[str, Any], *, control_delta: bool) -> np.ndarray:
    if control_delta:
        return LIBERO_DELTA_DUMMY_ACTION.copy()
    gripper_pos, gripper_quat = transform_hand_to_gripper(
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
    )
    gripper_axis_angle = rot_transform(gripper_quat, input_format="quat", target_format="axis_angle").astype(np.float32)
    return np.concatenate((gripper_pos, gripper_axis_angle, np.asarray([-1.0], dtype=np.float32)))


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


def _complete_score_for_frame(response: dict[str, Any], frame_id: int | None) -> float | None:
    if frame_id is None:
        return None
    complete_value = response.get("is_complete", response.get("complete"))
    frame_ids_value = response.get("frame_query_frame_id")
    if complete_value is None or frame_ids_value is None:
        return None

    scores = _first_batch(complete_value).astype(np.float32).reshape(-1)
    frame_ids = _first_batch(frame_ids_value).astype(np.int64).reshape(-1)
    if scores.shape[0] != frame_ids.shape[0]:
        return None

    matched_scores = scores[frame_ids == int(frame_id)]
    matched_scores = matched_scores[np.isfinite(matched_scores)]
    if matched_scores.size == 0:
        return None
    return float(np.max(matched_scores))


def _light_color_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(channel * 0.35 + 255 * 0.65)) for channel in color)


def _draw_response_points(
    image_rgb: np.ndarray,
    response: dict[str, Any] | None,
    frame_id: int | None,
    *,
    mode: str,
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
        if all(key in response for key in ("input_point", "input_object_id", "input_frame_id")):
            input_points = _first_batch(response["input_point"]).astype(np.float32)
            input_object_id = _first_batch(response["input_object_id"]).astype(np.int64)
            input_frame_id = _first_batch(response["input_frame_id"]).astype(np.int64)
            if input_points.ndim == 2 and input_object_id.ndim == 1 and input_frame_id.ndim == 1:
                mask = input_frame_id == 0
                for point, obj_id in zip(input_points[mask], input_object_id[mask]):
                    if point.shape[0] < 2 or not np.isfinite(point[:2]).all():
                        continue
                    x, y = np.rint(point[:2]).astype(int)
                    if 0 <= x < width and 0 <= y < height:
                        color = colors.get(int(obj_id), (160, 80, 160))
                        if point.shape[0] > 3 and float(point[3]) <= 0.5:
                            color = (145, 145, 145)
                        cv2.circle(image, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
                        count += 1

        rb_count = 0
        if "robobrain_point" in response and response["robobrain_point"] is not None:
            rb_points = np.asarray(response["robobrain_point"], dtype=np.float32).reshape(-1, 2)
            for point in rb_points:
                if not np.isfinite(point).all():
                    continue
                x, y = np.rint(point).astype(int)
                if 0 <= x < width and 0 <= y < height:
                    cv2.drawMarker(
                        image,
                        (x, y),
                        (255, 230, 40),
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=10,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
                    rb_count += 1
        _draw_text_rgb(image, f"tracking in={count} rb={rb_count}", (8, 18))
        return image

    if frame_id is not None and all(key in response for key in ("point", "object_id", "frame_id")):
        points = _first_batch(response["point"]).astype(np.float32)
        object_id = _first_batch(response["object_id"]).astype(np.int64)
        point_frame_id = _first_batch(response["frame_id"]).astype(np.int64)
        if points.ndim == 2 and object_id.ndim == 1 and point_frame_id.ndim == 1:
            mask = point_frame_id == int(frame_id)
            for actor_layer in (False, True):
                for point, obj_id in zip(points[mask], object_id[mask]):
                    is_actor = int(obj_id) == 0
                    if is_actor != actor_layer or point.shape[0] < 2 or not np.isfinite(point[:2]).all():
                        continue
                    x, y = np.rint(point[:2]).astype(int)
                    if 0 <= x < width and 0 <= y < height:
                        color = colors.get(int(obj_id), (160, 80, 160))
                        if point.shape[0] > 3 and float(point[3]) <= 0.5:
                            color = _light_color_rgb(color)
                        cv2.circle(image, (x, y), 3, color, -1, lineType=cv2.LINE_AA)
                        count += 1

    label_frame = "-" if frame_id is None else str(frame_id)
    _draw_text_rgb(image, f"prediction f={label_frame} out={count}", (8, 18))
    complete_score = _complete_score_for_frame(response, frame_id)
    if complete_score is None:
        complete_text = "-"
        complete_color = (255, 255, 255)
    else:
        complete_text = f"{complete_score:.2f}"
        complete_color = (80, 255, 80) if complete_score >= 0.5 else (255, 255, 255)
    _draw_text_rgb_right(image, complete_text, 18, color=complete_color)
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


def _goal_progress(env: Any) -> tuple[int, int, float]:
    problem_env = env.env if hasattr(env, "env") else env
    goal_state = problem_env.parsed_problem["goal_state"]
    completed_goals = sum(bool(problem_env._eval_predicate(state)) for state in goal_state)
    total_goals = len(goal_state)
    progress = float(completed_goals) / float(total_goals) if total_goals else 0.0
    return completed_goals, total_goals, progress


def _default_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name in {"libero_10", "libero_10_swap"}:
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
    parser.add_argument("--task-suite-name", default=Args.task_suite_name)
    parser.add_argument("--tasks", type=_parse_tasks, default=Args.tasks)
    parser.add_argument("--num-steps-wait", type=int, default=Args.num_steps_wait)
    parser.add_argument("--num-trials-per-task", type=int, default=Args.num_trials_per_task)
    parser.add_argument("--max-steps", type=int, default=Args.max_steps)
    parser.add_argument("--video-out-path", default=Args.video_out_path)
    parser.add_argument("--result-out-path", default=Args.result_out_path)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--control-delta", action="store_true", default=Args.control_delta)
    parser.add_argument("--execute-chunk-len", type=int, default=Args.execute_chunk_len)
    ns = parser.parse_args()
    return Args(
        host=ns.host,
        port=ns.port,
        task_suite_name=ns.task_suite_name,
        tasks=ns.tasks,
        num_steps_wait=ns.num_steps_wait,
        num_trials_per_task=ns.num_trials_per_task,
        max_steps=ns.max_steps,
        video_out_path=ns.video_out_path,
        result_out_path=ns.result_out_path,
        seed=ns.seed,
        save_video=not ns.no_save_video,
        control_delta=ns.control_delta,
        execute_chunk_len=ns.execute_chunk_len,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cs.print(args, markup=False)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = args.tasks if args.tasks is not None else list(range(num_tasks_in_suite))
    for task_order, task_id in enumerate(task_ids, start=1):
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(f"task id {task_id} out of range for {args.task_suite_name}: 0-{num_tasks_in_suite - 1}")
    max_steps = args.max_steps if args.max_steps is not None else _default_max_steps(args.task_suite_name)

    video_dir = pathlib.Path(args.video_out_path)
    result_path = pathlib.Path(args.result_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(host=args.host, port=args.port, execute_chunk_len=args.execute_chunk_len)
    total_episodes = 0
    total_successes = 0
    total_progress = 0.0
    task_results: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            args.seed,
            control_delta=args.control_delta,
        )
        try:
            total_goals = len(env.env.parsed_problem["goal_state"])
            task_episodes = 0
            task_successes = 0
            task_progress = 0.0

            for episode_idx in range(args.num_trials_per_task):
                logging.info("Task %s episode %s: %s", task_id, episode_idx, task_description)
                cs.print(
                    f"[cyan]task {task_order}/{len(task_ids)}[/cyan] "
                    f"id={task_id} episode {episode_idx + 1}/{args.num_trials_per_task}: "
                    f"{task_description}"
                )
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                client.reset_episode(task_id=task_id, episode_idx=episode_idx, env=env)
                prediction_images = []
                tracking_images = []
                done = False

                for step in range(max_steps + args.num_steps_wait):
                    if step < args.num_steps_wait:
                        cs.print(
                            f"[dim]task {task_order}/{len(task_ids)} id={task_id} "
                            f"episode {episode_idx + 1}/{args.num_trials_per_task} "
                            f"wait_step {step + 1}/{args.num_steps_wait}[/dim]"
                        )
                        action = _dummy_action(obs, control_delta=args.control_delta)
                    else:
                        policy_step = step - args.num_steps_wait + 1
                        cs.print(
                            f"[dim]task {task_order}/{len(task_ids)} id={task_id} "
                            f"episode {episode_idx + 1}/{args.num_trials_per_task} "
                            f"step {policy_step}/{max_steps}[/dim]"
                        )
                        action = client.infer(_prepare_observation(obs), task_description)

                    frame = np.ascontiguousarray(obs["agentview_image"][::-1, :])
                    prediction_images.append(
                        _draw_response_points(
                            frame,
                            client.last_response,
                            client.last_action_frame_id,
                            mode="prediction",
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

                task_episodes += 1
                total_episodes += 1
                if done:
                    task_successes += 1
                    total_successes += 1

                completed_goals, total_goals, progress = _goal_progress(env)
                task_progress += progress
                total_progress += progress

                if args.save_video:
                    suffix = "success" if done else "failure"
                    video_stem = f"task_{task_id:03d}_ep_{episode_idx:03d}_{suffix}"
                    _save_video_ffmpeg(prediction_images, video_dir / f"{video_stem}_prediction.mp4", fps=10.0)
                    _save_video_ffmpeg(tracking_images, video_dir / f"{video_stem}_tracking.mp4", fps=10.0)

                cs.print(
                    f"task={task_id} episode={episode_idx} success={done} "
                    f"progress={completed_goals}/{total_goals}"
                )

            task_result = {
                "task_id": task_id,
                "task_desc": task_description,
                "total_goals": total_goals,
                "success_rate": float(task_successes) / float(task_episodes),
                "progress_rate": float(task_progress) / float(task_episodes),
                "num_episodes": task_episodes,
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
