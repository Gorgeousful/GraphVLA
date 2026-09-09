#!/usr/bin/env python3
"""Thin LIBERO evaluation client for the GraphVLA inference server.

The client owns benchmark rollout, observation history, and execution of
server-returned action chunks. Planning, perception, model inference, and action
recovery live in script/server.py.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import dataclasses
import json
import logging
import pathlib
import cv2
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

import numpy as np
from rich.console import Console
from scipy.spatial.transform import Rotation as R
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)
import websockets


cs = Console()


def _worker_prefix(worker_id: int) -> str:
    return f"\\[worker {worker_id}]"

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
    action_delta: bool = False
    num_workers: int = 1
    resume_dir: Path | None = None


@dataclasses.dataclass(frozen=True)
class EpisodeJob:
    task_order: int
    task_id: int
    episode_idx: int


class EpisodeScheduler:
    """Prefer idle tasks, then balance concurrent episodes across tasks."""

    def __init__(self, task_entries: list[tuple[int, int]], episodes_per_task: int) -> None:
        self.pending = {
            task_id: collections.deque(
                EpisodeJob(task_order, task_id, episode_idx)
                for episode_idx in range(episodes_per_task)
            )
            for task_order, task_id in task_entries
        }
        self.task_order = {task_id: task_order for task_order, task_id in task_entries}
        self.active = {task_id: 0 for _, task_id in task_entries}
        self.lock = threading.Lock()

    def acquire(self) -> EpisodeJob | None:
        with self.lock:
            available = [task_id for task_id, jobs in self.pending.items() if jobs]
            if not available:
                return None
            task_id = min(
                available,
                key=lambda item: (
                    self.active[item],
                    self.pending[item][0].episode_idx,
                    self.task_order[item],
                ),
            )
            self.active[task_id] += 1
            return self.pending[task_id].popleft()

    def release(self, job: EpisodeJob) -> None:
        with self.lock:
            if self.active[job.task_id] <= 0:
                raise RuntimeError(f"task {job.task_id} has no active episode to release")
            self.active[job.task_id] -= 1


class OverallProgress:
    def __init__(self, total_episodes: int, completed_episodes: int = 0) -> None:
        self.total_episodes = total_episodes
        self.completed_episodes = completed_episodes
        self.resumed_episodes = completed_episodes
        self.next_percent = int(100.0 * completed_episodes / total_episodes) + 1
        self.lock = threading.Lock()
        self.start_time = time.monotonic()

    def complete_episode(self) -> None:
        with self.lock:
            self.completed_episodes += 1
            if self.completed_episodes * 100 < self.next_percent * self.total_episodes:
                return
            percent = 100.0 * self.completed_episodes / self.total_episodes
            elapsed = time.monotonic() - self.start_time
            newly_completed = self.completed_episodes - self.resumed_episodes
            eta_seconds = elapsed / newly_completed * (self.total_episodes - self.completed_episodes)
            eta_hours, eta_remainder = divmod(int(eta_seconds), 3600)
            eta_minutes, eta_seconds = divmod(eta_remainder, 60)
            self.next_percent = min(int(percent) + 1, 101)
            cs.print(
                f"[bold magenta]Overall progress: {self.completed_episodes}/"
                f"{self.total_episodes} ({percent:.1f}%) "
                f"ETA {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}[/bold magenta]"
            )


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
    def __init__(self, *, host: str, port: int, worker_id: int = 0) -> None:
        self.host = host
        self.port = port
        self.worker_id = worker_id
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

    def set_camera(self, *, env: Any) -> None:
        self.intrinsic, self.extrinsic = camera_matrices_from_env(env, camera_name=LIBERO_CAMERA_NAME)

    def reset_episode(self) -> None:
        self.pending_observations.reset()
        self.action_chunk.clear()
        self.action_frame_ids.clear()
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
            "worker_id": self.worker_id,
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
        async with websockets.connect(
            uri, max_size=None, proxy=None, ping_interval=None, ping_timeout=None
        ) as websocket:
            await websocket.send(json.dumps(data))
            message = await websocket.recv()
        return json.loads(message)


def _server_info(*, host: str, port: int) -> tuple[str, float]:
    uri = f"ws://{host}:{port}"
    response = asyncio.run(InferenceClient._websocket_json(uri, {"type": "server_info"}))
    if "error" in response:
        raise RuntimeError(response["error"])
    ckpt_path = response.get("ckpt_path")
    if not isinstance(ckpt_path, str) or not ckpt_path:
        raise ValueError(f"server returned invalid ckpt_path: {ckpt_path!r}")
    progress_threshold = response.get("progress_threshold")
    if (
        isinstance(progress_threshold, bool)
        or not isinstance(progress_threshold, int | float)
        or not np.isfinite(progress_threshold)
    ):
        raise ValueError(f"server returned invalid progress_threshold: {progress_threshold!r}")
    return ckpt_path, float(progress_threshold)


def _ckpt_dir_name(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    run_dir = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    return f"{run_dir}-{path.stem}"


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


def _get_libero_env(
    task: Any, resolution: int, seed: int, control_freq: int, *, action_delta: bool = True,
) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
        "control_delta": action_delta,
        "control_freq": control_freq,
        "ignore_done": True,
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


def _dummy_action(
    observation: dict[str, Any] | None = None,
    *,
    action_delta: bool = True,
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    if not action_delta:
        if observation is None:
            raise ValueError("absolute wait action requires the current observation")
        position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError(
                "absolute wait action requires robot0_eef_pos [3] and robot0_eef_quat [4]"
            )
        action[:3] = position
        action[3:6] = _quat2axisangle(quaternion)
    action[6] = -1.0
    return action


def _to_libero_action(action: np.ndarray, *, action_delta: bool) -> np.ndarray:
    """Adapt a canonical world hand action to LIBERO's controller frame."""
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (7,):
        raise ValueError(f"LIBERO action must be 7-D, got {action.shape}")
    if action_delta:
        return action.astype(np.float32)
    hand_to_site = np.asarray([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    hand_rotation = R.from_rotvec(action[3:6]).as_matrix()
    action[3:6] = R.from_matrix(hand_rotation @ hand_to_site).as_rotvec()
    return action.astype(np.float32)


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


def _draw_response_scores(
    image: np.ndarray,
    response: dict[str, Any],
    progress_threshold: float,
) -> None:
    progress = _current_score(response, "subtask_progress")
    progress_text = "-" if progress is None else f"{progress:.2f}"
    progress_color = (
        (80, 255, 80)
        if progress is not None and progress >= progress_threshold
        else (255, 255, 255)
    )
    _draw_text_rgb_right(image, progress_text, 18, color=progress_color)


def _draw_response_points(
    image_rgb: np.ndarray,
    response: dict[str, Any] | None,
    frame_id: int | None,
    *,
    mode: str,
    progress_threshold: float,
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
        7: (255, 165, 0),
    }
    inactive_color = (145, 145, 145)
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
                    pale_color = tuple(
                        int(round(channel * 0.35 + 255 * 0.65)) for channel in color
                    )
                    if not is_active:
                        color = inactive_color
                    elif float(point[2]) <= 0.5:
                        color = pale_color
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
                        (255, 230, 40) if is_active else inactive_color,
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=10,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
                    initial_count += 1
        return image

    point_plan_value = response.get("point_plan")
    if frame_id is not None and intrinsic is not None and point_plan_value is not None:
        point_plan = _first_batch(point_plan_value).astype(np.float32)
        future_index = int(frame_id) - 1
        if point_plan.ndim == 3 and point_plan.shape[-1] == 3 and 0 <= future_index < len(point_plan):
            points = point_plan[future_index]
            mask_value = response.get("point_plan_mask")
            if mask_value is None:
                point_mask = np.ones(len(points), dtype=bool)
            else:
                point_masks = _first_batch(mask_value).astype(bool)
                if point_masks.shape != point_plan.shape[:-1]:
                    raise ValueError(
                        f"point_plan_mask shape {point_masks.shape} does not match "
                        f"point_plan {point_plan.shape[:-1]}"
                    )
                point_mask = point_masks[future_index]
            camera_matrix = np.asarray(intrinsic, dtype=np.float32)
            if camera_matrix.shape != (3, 3):
                raise ValueError(f"intrinsic must have shape (3, 3), got {camera_matrix.shape}")
            for point_index, (point, valid) in enumerate(zip(points, point_mask, strict=True)):
                if not valid or not np.isfinite(point).all() or point[2] <= 1e-6:
                    continue
                pixel = camera_matrix @ point
                x, y = np.rint(pixel[:2] / pixel[2]).astype(int)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                color = colors.get(point_index, inactive_color)
                cv2.circle(image, (x, y), 4, color, -1, lineType=cv2.LINE_AA)

    if frame_id is not None:
        _draw_text_rgb(image, f"f={frame_id}", (8, 18))
    _draw_response_scores(image, response, progress_threshold)
    return image


def _save_video_ffmpeg(
    frames_rgb: list[np.ndarray], save_path: Path, *, fps: float = 10.0, worker_id: int = 0,
) -> None:
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
    cs.print(f"{_worker_prefix(worker_id)} saved video to: {save_path} ({len(frames_rgb)} frames, {fps:.1f} fps)")


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


ARTIFACT_PATTERN = re.compile(
    r"task_(\d+)_ep_(\d+)_sec_\d+_(success|failure)_(success|failure)\.json$"
)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _load_episode_records(video_dir: Path) -> dict[tuple[int, int], Path]:
    records: dict[tuple[int, int], Path] = {}
    for path in sorted(video_dir.glob("*.json")):
        match = ARTIFACT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in records:
            raise ValueError(
                f"duplicate resume records for task={key[0]} episode={key[1]}: "
                f"{records[key].name}, {path.name}"
            )
        json.loads(path.read_text(encoding="utf-8"))
        records[key] = path
    return records


def _restore_episode_result(
    path: Path,
    *,
    task_id: int,
    episode_idx: int,
    init_state_id: int,
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    metadata = record.get("metadata", {})
    expected = (task_id, episode_idx, init_state_id)
    actual = (
        metadata.get("task_id"),
        metadata.get("episode_id"),
        metadata.get("init_state_id"),
    )
    if actual != expected:
        raise ValueError(f"resume metadata mismatch in {path}: expected {expected}, got {actual}")
    if "result" in record:
        return record["result"]

    match = ARTIFACT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid resume artifact name: {path.name}")
    env_success = match.group(3) == "success"
    episode_result = {
        "episode_id": episode_idx,
        "init_state_id": init_state_id,
        "success": env_success,
        "server_success": match.group(4) == "success",
        "interrupted": False,
        "progress": float(env_success),
        "completed_subtasks": [],
        "completed_goals": [],
    }
    record["result"] = episode_result
    _write_json_atomic(path, record)
    return episode_result


def _default_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name in {"libero_10", "libero_10_swap", "libero_custom", "libero_custom_0902", "libero_swap_test"}:
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


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate GraphVLA through the observation-driven inference server on LIBERO.")
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--control-freq", type=int, default=Args.control_freq)
    parser.add_argument("--task-suite-name", default=Args.task_suite_name)
    parser.add_argument("--tasks", type=int, nargs="+", default=Args.tasks)
    parser.add_argument("--num-steps-wait", type=int, default=Args.num_steps_wait)
    parser.add_argument("--num-trials-per-task", type=int, default=Args.num_trials_per_task)
    parser.add_argument("--trials-init-state", type=int, nargs="+", default=Args.trials_init_state)
    parser.add_argument("--max-steps", type=int, default=Args.max_steps)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=Args.resume_dir,
        help="Resume an interrupted evaluation from an existing suite output directory.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=Args.num_workers,
        help="Number of episodes evaluated concurrently against the same server.",
    )
    parser.add_argument("--no-save-video", action="store_true")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--absolute-action",
        dest="action_delta",
        action="store_false",
        help="Use LIBERO absolute OSC pose control (default).",
    )
    action_group.add_argument(
        "--delta-action",
        dest="action_delta",
        action="store_true",
        help="Use LIBERO delta OSC control; must match the checkpoint action mode.",
    )
    parser.set_defaults(action_delta=Args.action_delta)
    ns = parser.parse_args()
    if ns.control_freq <= 0:
        parser.error("--control-freq must be positive")
    if ns.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if ns.num_trials_per_task <= 0:
        parser.error("--num-trials-per-task must be positive")
    if ns.resume_dir is not None and ns.no_save_video:
        parser.error("--resume-dir cannot be combined with --no-save-video")
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
        action_delta=ns.action_delta,
        num_workers=ns.num_workers,
        resume_dir=ns.resume_dir,
    )


def _evaluate_task(
    *,
    args: Args,
    task_suite: Any,
    task_order: int,
    total_tasks: int,
    task_id: int,
    max_steps: int,
    progress_threshold: float,
    video_dir: Path,
    client: InferenceClient,
    overall_progress: OverallProgress,
    existing_records: dict[tuple[int, int], Path],
    episode_indices: list[int] | None = None,
) -> dict[str, Any]:
    worker_id = client.worker_id
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    init_state_ids = (
        args.trials_init_state if args.trials_init_state is not None else list(range(args.num_trials_per_task))
    )
    invalid_init_state_ids = [
        state_id for state_id in init_state_ids if state_id < 0 or state_id >= len(initial_states)
    ]
    if invalid_init_state_ids:
        raise ValueError(
            f"initial state indices {invalid_init_state_ids} out of range for task {task_id}; "
            f"available range is 0-{len(initial_states) - 1}"
        )
    selected_episode_indices = (
        list(range(len(init_state_ids))) if episode_indices is None else episode_indices
    )
    if len(set(selected_episode_indices)) != len(selected_episode_indices):
        raise ValueError(f"duplicate episode indices for task {task_id}: {selected_episode_indices}")
    invalid_episode_indices = [
        episode_idx
        for episode_idx in selected_episode_indices
        if episode_idx < 0 or episode_idx >= len(init_state_ids)
    ]
    if invalid_episode_indices:
        raise ValueError(
            f"episode indices {invalid_episode_indices} out of range for task {task_id}; "
            f"available range is 0-{len(init_state_ids) - 1}"
        )
    missing_episode_indices = [
        episode_idx
        for episode_idx in selected_episode_indices
        if (task_id, episode_idx) not in existing_records
    ]
    env = None
    task_description = task.language
    total_goals = 1
    if missing_episode_indices:
        env, task_description = _get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            seed=args.seed,
            control_freq=args.control_freq,
            action_delta=args.action_delta,
        )
        total_goals = len(env.env.parsed_problem["goal_state"])
        client.set_camera(env=env)
    try:
        task_episodes = 0
        task_successes = 0
        task_server_successes = 0
        task_progress = 0.0
        episode_results: list[dict[str, Any]] = []

        for episode_idx in selected_episode_indices:
            init_state_id = init_state_ids[episode_idx]
            existing_path = existing_records.get((task_id, episode_idx))
            if existing_path is not None:
                episode_result = _restore_episode_result(
                    existing_path,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    init_state_id=init_state_id,
                )
                episode_results.append(episode_result)
                task_episodes += 1
                task_successes += int(episode_result["success"])
                task_server_successes += int(episode_result["server_success"])
                task_progress += float(episode_result["progress"])
                cs.print(
                    f"{_worker_prefix(worker_id)} [magenta]resumed task={task_id} "
                    f"episode={episode_idx} from {existing_path.name}[/magenta]"
                )
                continue

            if env is None:
                raise RuntimeError(f"task {task_id} has a missing episode but no environment")
            logging.info("[worker %s] Task %s episode %s: %s", worker_id, task_id, episode_idx, task_description)
            cs.print(
                f"{_worker_prefix(worker_id)} [cyan]task {task_order}/{total_tasks}[/cyan] "
                f"id={task_id} episode {episode_idx + 1}/{len(init_state_ids)} "
                f"init_state={init_state_id}: "
                f"{task_description}"
            )
            env.reset()
            obs = env.set_init_state(initial_states[init_state_id])
            client.reset_episode()
            combined_frames = []
            step_records: list[dict[str, Any]] = []
            inference_records: list[dict[str, Any]] = []
            current_inference_id: int | None = None
            done = False
            server_done = False
            interrupted = False

            try:
                for step in range(max_steps + args.num_steps_wait):
                    if step < args.num_steps_wait:
                        cs.print(
                            f"{_worker_prefix(worker_id)} [dim]task {task_order}/{total_tasks} id={task_id} "
                            f"episode {episode_idx + 1}/{len(init_state_ids)} "
                            f"wait_step {step + 1}/{args.num_steps_wait}[/dim]"
                        )
                        action = _dummy_action(obs, action_delta=args.action_delta)
                        current_state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        policy_step = None
                    else:
                        policy_step = step - args.num_steps_wait + 1
                        cs.print(
                            f"{_worker_prefix(worker_id)} [dim]task {task_order}/{total_tasks} id={task_id} "
                            f"episode {episode_idx + 1}/{len(init_state_ids)} "
                            f"step {policy_step}/{max_steps}[/dim]"
                        )
                        prepared_observation = _prepare_observation(obs, env)
                        previous_response = client.last_response
                        action = client.infer(prepared_observation, task_description)
                        current_state = prepared_observation["state"]
                        if client.last_response is not previous_response and not client.episode_done:
                            current_inference_id = len(inference_records)
                            response = client.last_response or {}
                            inference_records.append(
                                {
                                    "inference_id": current_inference_id,
                                    "video_frame": len(combined_frames),
                                    "subtask": response.get("subtask"),
                                    "subtask_index": response.get("subtask_index"),
                                    "tracking_point": response.get("tracking_point"),
                                    "tracking_object_id": response.get("tracking_object_id"),
                                    "tracking_point_active": response.get("tracking_point_active"),
                                }
                            )
                        if client.episode_done:
                            server_done = True
                            cs.print(
                                f"{_worker_prefix(worker_id)} [green]server completed episode "
                                f"at policy step {policy_step}[/green]"
                            )
                            break

                    frame = np.ascontiguousarray(obs["agentview_image"][::-1, :])
                    prediction_frame = _draw_response_points(
                        frame,
                        client.last_response,
                        client.last_action_frame_id,
                        mode="prediction",
                        progress_threshold=progress_threshold,
                        intrinsic=client.intrinsic,
                    )
                    tracking_frame = _draw_response_points(
                        frame,
                        client.last_response,
                        client.last_action_frame_id,
                        mode="tracking",
                        progress_threshold=progress_threshold,
                    )
                    combined_frames.append(np.hstack([prediction_frame, tracking_frame]))
                    controller_action = _to_libero_action(
                        action,
                        action_delta=args.action_delta,
                    )
                    step_records.append(
                        {
                            "video_frame": len(combined_frames) - 1,
                            "policy_step": policy_step,
                            "state": np.asarray(current_state, dtype=np.float64).tolist(),
                            "action": np.asarray(action, dtype=np.float64).tolist(),
                            "controller_action": np.asarray(controller_action, dtype=np.float64).tolist(),
                            "inference_id": current_inference_id,
                            "action_frame_id": client.last_action_frame_id if policy_step is not None else None,
                        }
                    )
                    obs, _, done, _ = env.step(controller_action.tolist())
                    if done:
                        break

            except KeyboardInterrupt:
                interrupted = True
                cs.print(
                    f"{_worker_prefix(worker_id)} [yellow]task={task_id} episode={episode_idx} interrupted; "
                    "saving current progress and continuing[/yellow]"
                )

            task_episodes += 1
            env_success = bool(done) and not interrupted
            if env_success:
                task_successes += 1
            if server_done:
                task_server_successes += 1

            (
                completed_goals,
                total_goals,
                progress,
                completed_subtasks,
                completed_goal_states,
            ) = _goal_progress(env)
            task_progress += progress
            episode_result = {
                "episode_id": episode_idx,
                "init_state_id": init_state_id,
                "success": env_success,
                "server_success": server_done,
                "interrupted": interrupted,
                "progress": progress,
                "completed_subtasks": completed_subtasks,
                "completed_goals": completed_goal_states,
            }

            if args.save_video:
                env_result = "success" if env_success else "failure"
                server_result = "success" if server_done else "failure"
                duration_seconds = round(len(combined_frames) / args.control_freq)
                artifact_stem = (
                    f"task_{task_id:03d}_ep_{episode_idx:03d}_sec_{duration_seconds:03d}_{env_result}_{server_result}"
                )
                _save_video_ffmpeg(
                    combined_frames, video_dir / f"{artifact_stem}.mp4",
                    fps=float(args.control_freq), worker_id=worker_id,
                )
                record = {
                    "metadata": {
                        "task_id": task_id,
                        "episode_id": episode_idx,
                        "init_state_id": init_state_id,
                        "task_description": task_description,
                        "fps": args.control_freq,
                        "total_goals": total_goals,
                    },
                    "steps": step_records,
                    "inferences": inference_records,
                    "result": episode_result,
                }
                json_path = video_dir / f"{artifact_stem}.json"
                _write_json_atomic(json_path, record)
                cs.print(
                    f"{_worker_prefix(worker_id)} saved execution record to: "
                    f"{json_path} ({len(step_records)} frames)"
                )

            episode_results.append(episode_result)

            cs.print(
                f"{_worker_prefix(worker_id)} task={task_id} episode={episode_idx} "
                f"success={env_success} env_done={done} server_done={server_done} "
                f"progress={completed_goals}/{total_goals} "
                f"completed_subtasks={completed_subtasks} "
                f"completed_goals={completed_goal_states}"
            )
            overall_progress.complete_episode()

        return {
            "task_id": task_id,
            "task_desc": task_description,
            "total_goals": total_goals,
            "success_rate": float(task_successes) / float(task_episodes),
            "server_success_rate": float(task_server_successes) / float(task_episodes),
            "progress_rate": float(task_progress) / float(task_episodes),
            "num_episodes": task_episodes,
            "episodes": episode_results,
        }
    finally:
        if env is not None:
            env.close()


def _evaluate_task_group(
    args: Args,
    worker_id: int,
    scheduler: EpisodeScheduler,
    total_tasks: int,
    max_steps: int,
    progress_threshold: float,
    video_dir: Path,
    overall_progress: OverallProgress,
    existing_records: dict[tuple[int, int], Path],
) -> list[dict[str, Any]]:
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    client = InferenceClient(host=args.host, port=args.port, worker_id=worker_id)
    results = []
    while (job := scheduler.acquire()) is not None:
        try:
            results.append(
                _evaluate_task(
                    args=args,
                    task_suite=task_suite,
                    task_order=job.task_order,
                    total_tasks=total_tasks,
                    task_id=job.task_id,
                    max_steps=max_steps,
                    progress_threshold=progress_threshold,
                    video_dir=video_dir,
                    client=client,
                    overall_progress=overall_progress,
                    existing_records=existing_records,
                    episode_indices=[job.episode_idx],
                )
            )
        finally:
            scheduler.release(job)
    return results


def _merge_task_results(
    partial_results: list[dict[str, Any]], task_ids: list[int],
) -> list[dict[str, Any]]:
    by_task: dict[int, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    for result in partial_results:
        by_task[result["task_id"]].append(result)

    merged_results = []
    for task_id in task_ids:
        parts = by_task[task_id]
        episodes = sorted(
            (episode for part in parts for episode in part["episodes"]),
            key=lambda episode: episode["episode_id"],
        )
        episode_ids = [episode["episode_id"] for episode in episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError(f"duplicate episode results for task {task_id}: {episode_ids}")
        num_episodes = len(episodes)
        merged_results.append({
            "task_id": task_id,
            "task_desc": parts[0]["task_desc"],
            "total_goals": max(part["total_goals"] for part in parts),
            "success_rate": (
                sum(episode["success"] for episode in episodes) / num_episodes
            ),
            "server_success_rate": (
                sum(episode["server_success"] for episode in episodes) / num_episodes
            ),
            "progress_rate": (
                sum(episode["progress"] for episode in episodes) / num_episodes
            ),
            "num_episodes": num_episodes,
            "episodes": episodes,
        })
    return merged_results


def main() -> None:
    from libero.libero import benchmark

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cs.print(f"[worker 0] {args}", markup=False)
    np.random.seed(args.seed)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = args.tasks if args.tasks is not None else list(range(num_tasks_in_suite))
    for task_id in task_ids:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(
                f"task id {task_id} out of range for {args.task_suite_name}: "
                f"0-{num_tasks_in_suite - 1}"
            )
    max_steps = args.max_steps if args.max_steps is not None else _default_max_steps(args.task_suite_name)

    ckpt_path, progress_threshold = _server_info(host=args.host, port=args.port)
    ckpt_dir_name = _ckpt_dir_name(ckpt_path)
    if args.resume_dir is None:
        timestamp = datetime.now().strftime("%m%d-%H%M")
        suite_output_dir = DEFAULT_OUTPUT_DIR / ckpt_dir_name / f"{args.task_suite_name}-{timestamp}"
    else:
        suite_output_dir = args.resume_dir.resolve()
        if not suite_output_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {suite_output_dir}")
        if suite_output_dir.parent.name != ckpt_dir_name:
            raise ValueError(
                f"resume directory belongs to {suite_output_dir.parent.name!r}, "
                f"but server is running {ckpt_dir_name!r}"
            )
        if not suite_output_dir.name.startswith(f"{args.task_suite_name}-"):
            raise ValueError(
                f"resume directory {suite_output_dir.name!r} does not match "
                f"suite {args.task_suite_name!r}"
            )
    video_dir = suite_output_dir / "videos"
    result_path = suite_output_dir / "result.json"
    video_dir.mkdir(parents=True, exist_ok=True)

    task_entries = list(enumerate(task_ids, start=1))
    episodes_per_task = (
        len(args.trials_init_state)
        if args.trials_init_state is not None
        else args.num_trials_per_task
    )
    existing_records = _load_episode_records(video_dir) if args.resume_dir is not None else {}
    requested_keys = {
        (task_id, episode_idx)
        for task_id in task_ids
        for episode_idx in range(episodes_per_task)
    }
    resumed_episodes = len(requested_keys.intersection(existing_records))
    total_requested_episodes = len(task_entries) * episodes_per_task
    overall_progress = OverallProgress(total_requested_episodes, resumed_episodes)
    if args.resume_dir is not None:
        cs.print(
            f"{_worker_prefix(0)} [bold magenta]Resuming {resumed_episodes}/"
            f"{total_requested_episodes} completed episodes from {suite_output_dir}[/bold magenta]"
        )
    worker_count = min(args.num_workers, total_requested_episodes) if task_entries else 1
    scheduler = EpisodeScheduler(task_entries, episodes_per_task)
    if worker_count == 1:
        partial_results = _evaluate_task_group(
            args,
            0,
            scheduler,
            len(task_entries),
            max_steps,
            progress_threshold,
            video_dir,
            overall_progress,
            existing_records,
        )
    else:
        cs.print(
            f"{_worker_prefix(0)} running {total_requested_episodes} episodes from "
            f"{len(task_entries)} tasks across {worker_count} concurrent workers"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_task_group,
                    args,
                    worker_id,
                    scheduler,
                    len(task_entries),
                    max_steps,
                    progress_threshold,
                    video_dir,
                    overall_progress,
                    existing_records,
                )
                for worker_id in range(worker_count)
            ]
            partial_results = [result for future in futures for result in future.result()]
    task_results = _merge_task_results(partial_results, task_ids)

    total_episodes = sum(result["num_episodes"] for result in task_results)
    total_successes = sum(
        episode["success"] for result in task_results for episode in result["episodes"]
    )
    total_server_successes = sum(
        episode["server_success"] for result in task_results for episode in result["episodes"]
    )
    total_progress = sum(
        episode["progress"] for result in task_results for episode in result["episodes"]
    )

    result = {
        "task_suite": args.task_suite_name,
        "success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
        "server_success_rate": (
            float(total_server_successes) / float(total_episodes) if total_episodes else 0.0
        ),
        "progress_rate": float(total_progress) / float(total_episodes) if total_episodes else 0.0,
        "total_episodes": total_episodes,
        "tasks": task_results,
    }
    _write_json_atomic(result_path, result)
    cs.print(f"{_worker_prefix(0)} saved results to: {result_path}")


if __name__ == "__main__":
    main()
