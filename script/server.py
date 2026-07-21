#!/usr/bin/env python3
"""GraphVLA observation-driven inference server.

The server owns task planning, online preprocessing, model inference, and
benchmark-specific action recovery. Clients send flat observation messages.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import cv2
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import websockets
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console
from scipy.spatial.transform import Rotation as R

from src.model.model import PointQueryModel
from src.training.checkpoint import TrainingCheckpoint
from src.module.task_analyzer import TaskAnalyzer
from src.module.node_locator import NodeLocatorRobo
from src.module.depth_predictor import DepthPredictorSTream3R
from src.module.node_segmenter import NodeSegmenter
from src.module.point_tracker import PointTracker
from src.common.geom_utils import sample_points_from_mask
from src.common.schema import (
    LIBERO_ACTOR_POINT_INDICES,
    LIBERO_GRIPPER_MAX_WIDTH,
    POINT_FEATURE_DIM,
    POINT_GRIPPER_OPENNESS_INDEX,
    POINT_GRIPPER_OPENNESS_MASK_INDEX,
    POINT_METRIC_DEPTH_INDEX,
    POINT_METRIC_DEPTH_MASK_INDEX,
    POINT_RELATIVE_DEPTH_INDEX,
    POINT_VISIBILITY_INDEX,
    taskstructure_to_json,
)
from src.module.scale_estimator import RawDepthShiftCalibrator


cs = Console()

REQUIRED_REQUEST_FIELDS = (
    "benchmark",
    "session_id",
    "language",
    "execute_chunk_len",
    "observation.images.image",
    "observation.state",
    "camera.intrinsics",
    "camera.extrinsics",
)


@dataclass
class Args:
    example: str
    ckpt_path: str
    host: str
    port: int
    device: str
    bge_path: str
    task_analyzer_api_key: str | None
    complete_threshold: float = 0.5
    complete_window: int = 3
    devices: dict[str, str] = field(default_factory=dict)


@dataclass
class ObservationFrame:
    image: np.ndarray
    state: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray


@dataclass
class InferenceSession:
    session_id: str
    benchmark: str
    language: str
    current_subtask: str | None = None
    taskstructure: dict[str, Any] | None = None
    subtask_index: int = 0
    task_complete: bool = False
    complete_streak: int = 0
    feature_history: list[dict[str, np.ndarray]] = field(default_factory=list)
    frame_index: int = 0
    object_nodes: list[dict[str, Any]] = field(default_factory=list)
    active_object_indices: list[int] = field(default_factory=list)
    object_segmenter: Any = None
    table_segmenter: Any = None
    point_tracker: Any = None
    depth_predictor: Any = None
    depth_calibrator: Any = None
    tracked_points: np.ndarray | None = None
    robobrain_points: np.ndarray | None = None
    table_mask: np.ndarray | None = None

    def reset(self, *, benchmark: str, language: str) -> None:
        self.benchmark = benchmark
        self.language = language
        self.current_subtask = None
        self.taskstructure = None
        self.subtask_index = 0
        self.task_complete = False
        self.complete_streak = 0
        self.reset_preprocessor()

    def reset_preprocessor(self) -> None:
        self.feature_history.clear()
        self.frame_index = 0
        self.object_nodes.clear()
        self.active_object_indices.clear()
        self.object_segmenter = None
        self.table_segmenter = None
        self.point_tracker = None
        self.depth_predictor = None
        self.depth_calibrator = None
        self.tracked_points = None
        self.robobrain_points = None
        self.table_mask = None


#: =======================================================
class TopLevelTaskPlanner:
    """Session-level planner that stays in semantic space."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        task_analyzer_api_key: str | None = None,
        complete_threshold: float = 0.5,
        complete_window: int = 3,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.task_analyzer_api_key = task_analyzer_api_key
        self.complete_threshold = complete_threshold
        self.complete_window = max(1, complete_window)
        self.task_cache: dict[str, dict[str, Any]] = {}
        self.task_analyzer: TaskAnalyzer | None = None
        self._load_taskstructure_cache()

    def plan(self, request: Mapping[str, Any], session: InferenceSession) -> dict[str, Any]:
        if session.taskstructure is None:
            session.taskstructure = self._taskstructure(str(request["language"]))

        subtasks = session.taskstructure.get("subtasks", [])
        if not subtasks:
            raise ValueError(f"Task has no subtasks: {session.language!r}")

        subtask_index = min(session.subtask_index, len(subtasks) - 1)
        subtaskstructure = subtasks[subtask_index]
        session.current_subtask = str(subtaskstructure.get("subtask", ""))
        return subtaskstructure

    def update_after_inference(
        self,
        outputs: Mapping[str, Any],
        session: InferenceSession,
        model_input: Mapping[str, Any],
        gripper_widths: Sequence[float],
        actions: Sequence[Sequence[float]],
    ) -> bool:
        if session.task_complete or session.taskstructure is None:
            return False

        frame_scores = self._completion_frame_scores(outputs, model_input)
        subtasks = session.taskstructure.get("subtasks", [])
        subtask_label = f"subtask [{session.subtask_index + 1}/{len(subtasks)}]"
        gripper_width_text = ", ".join(f"{width:.4f}" for width in gripper_widths)
        gripper_action_text = ", ".join(f"{float(action[6]):.4f}" for action in actions)
        for _, score in frame_scores:
            score_text = (
                f"{subtask_label} complete_score={score:.4f} "
                f"chunk_len={len(actions)}\n"
                f"gripper_widths=[{gripper_width_text}] "
                f"gripper_actions=[{gripper_action_text}]"
            )
            if score >= self.complete_threshold:
                cs.print(f"[green]{score_text}[/green]", soft_wrap=True)
                session.complete_streak += 1
            else:
                cs.print(score_text, soft_wrap=True)
                session.complete_streak = 0

            if session.complete_streak >= self.complete_window:
                return self._advance_subtask(session)
        return False

    def _advance_subtask(self, session: InferenceSession) -> bool:
        subtasks = session.taskstructure.get("subtasks", [])
        session.complete_streak = 0
        if session.subtask_index + 1 >= len(subtasks):
            session.task_complete = True
            return False

        session.subtask_index += 1
        next_subtask = subtasks[session.subtask_index]
        session.current_subtask = str(next_subtask.get("subtask", ""))
        session.feature_history.clear()
        return True

    def _load_taskstructure_cache(self) -> None:
        path = self.dataset_dir / "meta" / "taskstructures.jsonl"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task = str(row.get("task", ""))
                if task:
                    self.task_cache[task] = row

    def _taskstructure(self, language: str) -> dict[str, Any]:
        if language not in self.task_cache:
            if not self.task_analyzer_api_key:
                raise RuntimeError(
                    "Taskstructure is missing from cache and --task-analyzer-api-key was not provided."
                )
            if self.task_analyzer is None:
                self.task_analyzer = TaskAnalyzer(api_key=self.task_analyzer_api_key)
            self.task_cache[language] = taskstructure_to_json(self.task_analyzer.analyze_task(language))
        return self.task_cache[language]

    def _completion_frame_scores(
        self,
        outputs: Mapping[str, Any],
        model_input: Mapping[str, Any],
    ) -> list[tuple[int, float]]:
        complete_value = outputs.get("is_complete")
        if complete_value is None:
            return []

        if isinstance(complete_value, torch.Tensor):
            complete_value = complete_value.detach().cpu().numpy()
        scores = np.asarray(complete_value, dtype=float)
        if scores.ndim >= 2 and scores.shape[0] == 1:
            scores = scores[0]
        if scores.ndim == 0:
            scores = scores.reshape(1)
        else:
            scores = scores.reshape(scores.shape[0], -1).max(axis=1)

        frame_ids = np.asarray(model_input.get("frame_query_frame_id", []), dtype=np.int64)
        if frame_ids.ndim >= 2 and frame_ids.shape[0] == 1:
            frame_ids = frame_ids[0]
        frame_ids = frame_ids.reshape(-1)
        if frame_ids.size != scores.size:
            raise ValueError(
                "is_complete and frame_query_frame_id must contain the same number of frames: "
                f"got {scores.size} scores and {frame_ids.size} frame ids"
            )
        return [
            (int(frame_id), float(score))
            for score, frame_id in zip(scores, frame_ids, strict=True)
            if frame_id == 0
        ]


class InputPreprocessor:
    """Build model inputs from flat observation requests."""

    def __init__(
        self,
        *,
        history_horizon: int,
        future_horizon: int,
        num_points: int,
        robot_cls: type[Any],
        dataset_dir: str | Path,
        devices: Mapping[str, str] | None = None,
    ) -> None:
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.num_points = num_points
        self.robot_cls = robot_cls
        self.devices = dict(devices or {})
        self.norm_stats = self._load_norm_stats(Path(dataset_dir))
        self.sam3_model = None
        self.robot: Any = None

    def build(
        self,
        request: Mapping[str, Any],
        session: InferenceSession,
        subtaskstructure: Mapping[str, Any],
    ) -> dict[str, Any]:
        frames = self._frames_from_request(request)
        for frame in frames:
            features = self._process_frame(session, frame)
            self._append_feature_history(session, features)
        return self._build_model_input(session, self._feature_window(session), subtaskstructure)

    def _device(self, name: str) -> str:
        return self.devices.get(name, self.devices.get("default", "cuda"))

    def _process_frame(self, session: InferenceSession, frame: ObservationFrame) -> dict[str, np.ndarray]:
        if session.tracked_points is None:
            self._initialize_perception(session, frame)
        else:
            self._update_perception(session, frame)

        depth = self._predict_depth(session, frame)
        gripper_uvd = self._state_to_gripper_uvd(frame)
        if session.tracked_points is None:
            raise RuntimeError("tracked_points is missing after perception update")
        features = {
            "tracks": np.asarray(session.tracked_points, dtype=np.float32),
            "depth": np.asarray(depth, dtype=np.float32),
            "gripper_uvd": np.asarray(gripper_uvd, dtype=np.float32),
            "gripper_openness": self._state_to_gripper_openness(frame),
        }
        session.frame_index += 1
        return features

    def _initialize_perception(self, session: InferenceSession, frame: ObservationFrame) -> None:
        if session.taskstructure is None:
            raise RuntimeError("taskstructure is missing before perception initialization")
        session.object_nodes = self._task_object_nodes(session.taskstructure)
        point_prompts = None
        if session.object_nodes:
            node_locator = NodeLocatorRobo(device_map=self._device("node_locator"))
            try:
                point_prompts = []
                for node in session.object_nodes:
                    points = self._locate_node_points(node_locator, frame.image, node["name"])
                    cs.print(
                        f"robobrain node={node['name']} pixel_xy={np.round(points, 1).tolist()}",
                        markup=False,
                    )
                    point_prompts.append(points)
            finally:
                del node_locator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            session.robobrain_points = np.asarray(point_prompts, dtype=np.float32)

        sam3_device = self._device("sam3")
        if self.sam3_model is None:
            self.sam3_model = NodeSegmenter(device=sam3_device).model
        session.object_segmenter = NodeSegmenter(device=sam3_device, model=self.sam3_model)
        session.table_segmenter = NodeSegmenter(device=sam3_device, model=self.sam3_model)

        if session.object_nodes:
            session.point_tracker = PointTracker(device=self._device("point_tracker"))
            object_masks = session.object_segmenter.predict(frame.image, points=point_prompts, anchor_frame=True)
            object_points = np.stack(
                [sample_points_from_mask(mask, num_points=self.num_points) for mask in object_masks],
                axis=0,
            ).astype(np.float32)
            track_result = session.point_tracker.track(frame.image, points=object_points, anchor_frame=True)
            session.tracked_points = self._pack_tracks(track_result)
        else:
            session.tracked_points = np.zeros((0, self.num_points, 3), dtype=np.float32)

        table_masks = session.table_segmenter.predict(frame.image, prompt="table", anchor_frame=True)
        session.table_mask = self._largest_mask(table_masks, frame.image.shape[:2])

    def _update_perception(self, session: InferenceSession, frame: ObservationFrame) -> None:
        if session.object_segmenter is not None and session.object_nodes:
            session.object_segmenter.predict(frame.image, anchor_frame=False)
        if session.table_segmenter is not None:
            table_masks = session.table_segmenter.predict(frame.image, anchor_frame=False)
            session.table_mask = self._largest_mask(table_masks, frame.image.shape[:2])
        if session.point_tracker is not None and session.object_nodes:
            session.tracked_points = self._pack_tracks(session.point_tracker.track(frame.image, anchor_frame=False))

    def _predict_depth(self, session: InferenceSession, frame: ObservationFrame) -> np.ndarray:
        if session.depth_predictor is None:
            session.depth_predictor = DepthPredictorSTream3R(device=self._device("depth_predictor"))
        output = session.depth_predictor.predict(frame.image, anchor_frame=(session.frame_index == 0))
        depth = output[0] if isinstance(output, tuple) else output
        depth = np.asarray(depth, dtype=np.float32)
        if session.table_mask is None:
            return self._normalize_field(depth, "depths.depth_rel")
        if session.depth_calibrator is None:
            session.depth_calibrator = RawDepthShiftCalibrator(depth, session.table_mask)
            return self._normalize_field(depth, "depths.depth_rel")
        try:
            depth = session.depth_calibrator.calibrate(depth, session.table_mask)
        except ValueError:
            pass
        return self._normalize_field(depth, "depths.depth_rel")

    def _state_to_gripper_uvd(self, frame: ObservationFrame) -> np.ndarray:
        state = frame.state
        if state.size < 8:
            raise ValueError(f"observation.state must contain at least 8 values, got {state.size}")
        gripper_width = abs(float(state[6])) + abs(float(state[7]))
        output = self._robot().project_gripper_to_uvd(
            tcp_state=state[:6],
            intrinsic=frame.intrinsic,
            extrinsic=frame.extrinsic,
            gripper_width=gripper_width,
        )
        gripper_points = np.stack([
            np.asarray(output[key], dtype=np.float32)
            for key in (
                "root_uvd",
                "left_base_uvd",
                "right_base_uvd",
                "left_fingertip_uvd",
                "right_fingertip_uvd",
                "tcp_uvd",
            )
        ])
        return gripper_points

    @staticmethod
    def _state_to_gripper_openness(frame: ObservationFrame) -> np.ndarray:
        if frame.state.size < 8:
            raise ValueError(f"observation.state must contain at least 8 values, got {frame.state.size}")
        width = abs(float(frame.state[6])) + abs(float(frame.state[7]))
        return np.asarray(
            np.clip(width / LIBERO_GRIPPER_MAX_WIDTH, 0.0, 1.0), dtype=np.float32
        )

    def _build_model_input(
        self,
        session: InferenceSession,
        frames: list[dict[str, np.ndarray]],
        subtaskstructure: Mapping[str, Any],
    ) -> dict[str, Any]:
        height, width = frames[-1]["depth"].shape
        session.active_object_indices = self._subtask_object_indices(session, subtaskstructure)
        object_points = np.stack(
            [
                self._object_feats(
                    self._active_tracks(item["tracks"], session.active_object_indices),
                    item["depth"],
                    height,
                    width,
                )
                for item in frames
            ],
            axis=0,
        )
        actor_points = np.stack(
            [
                self._actor_feats(
                    item["gripper_uvd"], item["gripper_openness"], item["depth"], height, width
                )
                for item in frames
            ],
            axis=0,
        )[:, None]
        future_frame_offsets = np.arange(1, self.future_horizon + 1, dtype=np.int64)
        input_frame_offsets = np.arange(-self.history_horizon, 1, dtype=np.int64)
        actor_num_points = actor_points.shape[2]
        object_id = np.zeros(future_frame_offsets.size * actor_num_points, dtype=np.int64)
        point_id = np.tile(np.arange(actor_num_points, dtype=np.int64), future_frame_offsets.size)
        frame_id = np.repeat(future_frame_offsets, actor_num_points)
        input_object_id, input_point_id, input_frame_id = self._build_input_point_ids(
            input_frame_offsets,
            object_points.shape[1],
            object_points.shape[2],
            actor_num_points,
        )
        input_point = self._input_points_from_feats(object_points, actor_points, height, width)
        action_type = str(subtaskstructure.get("action_type", ""))
        action_degree = subtaskstructure.get("action_degree")
        object_roles = self._object_roles(subtaskstructure, expected_count=object_points.shape[1])
        object_condition_texts = [
            {"role": role, "action_type": action_type, "action_degree": action_degree}
            for role in object_roles
        ]
        actor_condition_text = {
            "role": "actor",
            "action_type": action_type,
            "action_degree": action_degree,
        }
        return {
            "point_feats": object_points[None].tolist(),
            "actor_feats": actor_points[None].tolist(),
            "object_condition_texts": object_condition_texts,
            "actor_condition_text": actor_condition_text,
            "object_id": object_id[None].tolist(),
            "point_id": point_id[None].tolist(),
            "frame_id": frame_id[None].tolist(),
            "frame_query_frame_id": np.zeros((1, 1), dtype=np.int64).tolist(),
            "actor_query_frame_id": future_frame_offsets[None].tolist(),
            "input_point": input_point[None].tolist(),
            "input_object_id": input_object_id[None].tolist(),
            "input_point_id": input_point_id[None].tolist(),
            "input_frame_id": input_frame_id[None].tolist(),
            "head_names": ("point", "metric_depth"),
            "actor_head_names": ("gripper_openness", "gripper_action"),
            "frame_head_names": "is_complete",
        }

    def _frames_from_request(
        self,
        request: Mapping[str, Any],
    ) -> list[ObservationFrame]:
        images = self._as_frame_sequence(
            self._array(request["observation.images.image"], name="observation.images.image", dtype=np.uint8),
            name="observation.images.image",
            single_ndim={2, 3},
        )
        states = self._as_frame_sequence(
            self._array(request["observation.state"], name="observation.state", dtype=np.float64),
            name="observation.state",
            single_ndim={1},
        )
        intrinsics = self._as_frame_sequence(
            self._array(request["camera.intrinsics"], name="camera.intrinsics", dtype=np.float64),
            name="camera.intrinsics",
            single_shape=(3, 3),
        )
        extrinsics = self._as_frame_sequence(
            self._array(request["camera.extrinsics"], name="camera.extrinsics", dtype=np.float64),
            name="camera.extrinsics",
            single_shape=(4, 4),
        )

        target_len = max(len(images), len(states), len(intrinsics), len(extrinsics))
        images = self._broadcast_frames(images, target_len, name="observation.images.image")
        states = self._broadcast_frames(states, target_len, name="observation.state")
        intrinsics = self._broadcast_frames(intrinsics, target_len, name="camera.intrinsics")
        extrinsics = self._broadcast_frames(extrinsics, target_len, name="camera.extrinsics")

        return [
            ObservationFrame(
                image=image,
                state=state,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )
            for image, state, intrinsic, extrinsic in zip(images, states, intrinsics, extrinsics, strict=True)
        ]

    def _as_frame_sequence(
        self,
        array: np.ndarray,
        *,
        name: str,
        single_ndim: set[int] | None = None,
        single_shape: tuple[int, ...] | None = None,
    ) -> list[np.ndarray]:
        if single_shape is not None:
            if array.shape == single_shape:
                return [array]
            if array.ndim == len(single_shape) + 1 and array.shape[1:] == single_shape:
                return [array[index] for index in range(array.shape[0])]
            raise ValueError(f"{name} must have shape {single_shape} or Tx{single_shape}, got {array.shape}")

        if single_ndim is None:
            raise ValueError(f"{name} sequence parser requires single_ndim or single_shape")
        if array.ndim in single_ndim:
            return [array]
        if array.ndim - 1 in single_ndim:
            return [array[index] for index in range(array.shape[0])]
        expected = " or ".join(str(ndim) for ndim in sorted(single_ndim))
        raise ValueError(
            f"{name} must be a single frame with ndim {expected}, or a frame chunk with one leading T axis; got {array.shape}"
        )

    @staticmethod
    def _broadcast_frames(frames: list[np.ndarray], target_len: int, *, name: str) -> list[np.ndarray]:
        if len(frames) == target_len:
            return frames
        if len(frames) == 1:
            return frames * target_len
        raise ValueError(f"{name} length {len(frames)} does not match history chunk length {target_len}")

    def _append_feature_history(self, session: InferenceSession, features: dict[str, np.ndarray]) -> None:
        session.feature_history.append(features)
        max_history = self.history_horizon + 1
        if len(session.feature_history) > max_history:
            del session.feature_history[: len(session.feature_history) - max_history]

    def _feature_window(self, session: InferenceSession) -> list[dict[str, np.ndarray]]:
        if not session.feature_history:
            raise RuntimeError("feature history is empty")
        target_len = self.history_horizon + 1
        pad_count = max(0, target_len - len(session.feature_history))
        return [session.feature_history[0]] * pad_count + session.feature_history[-target_len:]

    def _locate_node_points(
        self,
        node_locator: NodeLocatorRobo,
        image: np.ndarray,
        node_name: str,
    ) -> list[list[float]]:
        result = node_locator.inference(text=node_name, image=Image.fromarray(image))
        points = result.get("points") or []
        if not points:
            raise RuntimeError(f"NodeLocatorRobo found no points for node={node_name!r}")
        return self._locator_points_to_pixels(points, image.shape[:2])

    @staticmethod
    def _load_norm_stats(dataset_dir: Path) -> dict[str, Any]:
        path = dataset_dir / "meta" / "norm_stats_suite.json"
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return dict(payload.get("norm_stats", payload))

    def _normalize_field(self, value: np.ndarray, field: str) -> np.ndarray:
        if field not in self.norm_stats:
            raise KeyError(f"Missing norm stats for field={field!r}")
        stats = self.norm_stats[field]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        return ((value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)

    def _robot(self) -> Any:
        if self.robot is None:
            self.robot = self.robot_cls(embodiment="franka_panda", with_fingers=True)
        return self.robot

    def _object_nodes(self, subtaskstructure: Mapping[str, Any]) -> list[dict[str, Any]]:
        nodes = []
        for node in subtaskstructure.get("nodes", []):
            if not isinstance(node, Mapping):
                continue
            role = str(node.get("role", ""))
            if bool(node.get("need_object", False)) and role != "actor":
                nodes.append(dict(node))
        return nodes

    def _task_object_nodes(self, taskstructure: Mapping[str, Any]) -> list[dict[str, Any]]:
        nodes = []
        seen_names = set()
        for subtask in taskstructure.get("subtasks", []):
            if not isinstance(subtask, Mapping):
                continue
            for node in self._object_nodes(subtask):
                name = str(node.get("name", ""))
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                nodes.append(node)
        return nodes

    def _subtask_object_indices(
        self,
        session: InferenceSession,
        subtaskstructure: Mapping[str, Any],
    ) -> list[int]:
        by_name = {str(node.get("name", "")): index for index, node in enumerate(session.object_nodes)}
        indices = []
        for node in self._object_nodes(subtaskstructure):
            name = str(node.get("name", ""))
            if name in by_name:
                indices.append(by_name[name])
        return indices[:2]

    @staticmethod
    def _active_tracks(tracks: np.ndarray, active_indices: list[int]) -> np.ndarray:
        if not active_indices:
            return tracks[:0]
        valid_indices = [index for index in active_indices if 0 <= index < tracks.shape[0]]
        return tracks[valid_indices]

    def _object_roles(self, subtaskstructure: Mapping[str, Any], *, expected_count: int) -> list[str]:
        roles = []
        for node in self._object_nodes(subtaskstructure):
            roles.append(str(node.get("role", "")))
        fallback = ["patient", "target"]
        while len(roles) < expected_count:
            roles.append(fallback[len(roles)] if len(roles) < len(fallback) else f"object_{len(roles)}")
        return roles[:expected_count]

    def _object_feats(self, tracks: np.ndarray, depth: np.ndarray, height: int, width: int) -> np.ndarray:
        points = np.zeros((2, self.num_points, POINT_FEATURE_DIM), dtype=np.float32)
        count = min(tracks.shape[0], 2)
        if count == 0:
            return points
        uv = tracks[:count, :, :2]
        vis = tracks[:count, :, 2:3]
        sampled_depth, in_bounds = self._sample_depth(depth, uv, height, width)
        points[:count, :, :2] = self._normalize_uv(uv, height, width)
        points[:count, :, POINT_RELATIVE_DEPTH_INDEX:POINT_RELATIVE_DEPTH_INDEX + 1] = sampled_depth
        points[:count, :, POINT_VISIBILITY_INDEX:POINT_VISIBILITY_INDEX + 1] = vis * in_bounds
        return points

    def _actor_feats(
        self,
        gripper_uvd: np.ndarray,
        gripper_openness: np.ndarray,
        depth: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        if gripper_uvd.shape != (6, 3):
            raise ValueError(f"Expected gripper_uvd [6, 3], got {gripper_uvd.shape}")
        uv = gripper_uvd[:, :2]
        sampled_depth, in_bounds = self._sample_depth(depth, uv, height, width)
        sampled_depth[5] = sampled_depth[3:5].mean(axis=0)
        gripper_uvd = gripper_uvd[list(LIBERO_ACTOR_POINT_INDICES)]
        sampled_depth = sampled_depth[list(LIBERO_ACTOR_POINT_INDICES)]
        in_bounds = in_bounds[list(LIBERO_ACTOR_POINT_INDICES)]
        uv = gripper_uvd[:, :2]
        points = np.zeros((gripper_uvd.shape[0], POINT_FEATURE_DIM), dtype=np.float32)
        points[:, :2] = self._normalize_uv(uv, height, width)
        points[:, POINT_RELATIVE_DEPTH_INDEX:POINT_RELATIVE_DEPTH_INDEX + 1] = sampled_depth
        points[:, POINT_VISIBILITY_INDEX:POINT_VISIBILITY_INDEX + 1] = in_bounds
        points[:, POINT_METRIC_DEPTH_INDEX:POINT_METRIC_DEPTH_INDEX + 1] = self._normalize_field(
            gripper_uvd[:, 2:3], "gripper_d"
        )
        points[:, POINT_METRIC_DEPTH_MASK_INDEX:POINT_METRIC_DEPTH_MASK_INDEX + 1] = 1.0
        points[:, POINT_GRIPPER_OPENNESS_INDEX:POINT_GRIPPER_OPENNESS_INDEX + 1] = float(
            gripper_openness
        )
        points[:, POINT_GRIPPER_OPENNESS_MASK_INDEX:POINT_GRIPPER_OPENNESS_MASK_INDEX + 1] = 1.0
        return points

    @staticmethod
    def _build_input_point_ids(frame_offsets: np.ndarray, object_count: int, object_points: int, actor_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        object_ids = []
        point_ids = []
        frame_ids = []
        actor_local_ids = np.arange(actor_points, dtype=np.int64)
        object_local_ids = np.arange(object_points, dtype=np.int64)
        for frame_offset in frame_offsets:
            object_ids.append(np.zeros(actor_points, dtype=np.int64))
            point_ids.append(actor_local_ids)
            frame_ids.append(np.full(actor_points, int(frame_offset), dtype=np.int64))
            for object_index in range(object_count):
                object_ids.append(np.full(object_points, object_index + 1, dtype=np.int64))
                point_ids.append(object_local_ids)
                frame_ids.append(np.full(object_points, int(frame_offset), dtype=np.int64))
        return np.concatenate(object_ids), np.concatenate(point_ids), np.concatenate(frame_ids)

    @staticmethod
    def _input_points_from_feats(object_points: np.ndarray, actor_points: np.ndarray, height: int, width: int) -> np.ndarray:
        rows = []
        for frame_index in range(object_points.shape[0]):
            actor = actor_points[frame_index, 0]
            actor_uv = InputPreprocessor._denormalize_uv(actor[:, :2], height, width)
            rows.append(np.concatenate([actor_uv, actor[:, 4:5], actor[:, 3:4]], axis=-1))
            for object_index in range(object_points.shape[1]):
                item = object_points[frame_index, object_index]
                uv = InputPreprocessor._denormalize_uv(item[:, :2], height, width)
                rows.append(np.concatenate([uv, item[:, 2:3], item[:, 3:4]], axis=-1))
        return np.concatenate(rows, axis=0).astype(np.float32)

    @staticmethod
    def _denormalize_uv(uv_norm: np.ndarray, height: int, width: int) -> np.ndarray:
        uv = np.asarray(uv_norm, dtype=np.float32).copy()
        uv[..., 0] = uv[..., 0] * (width / 2.0) + width / 2.0
        uv[..., 1] = uv[..., 1] * (height / 2.0) + height / 2.0
        return uv

    @staticmethod
    def _sample_depth(depth: np.ndarray, uv: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        u = np.rint(uv[..., 0]).astype(np.int64)
        v = np.rint(uv[..., 1]).astype(np.int64)
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        sampled = np.zeros(uv.shape[:-1] + (1,), dtype=np.float32)
        safe_u = np.clip(u, 0, width - 1)
        safe_v = np.clip(v, 0, height - 1)
        sampled[..., 0] = depth[safe_v, safe_u]
        sampled[..., 0] = np.where(in_bounds, sampled[..., 0], 0.0)
        return sampled, in_bounds[..., None].astype(np.float32)

    @staticmethod
    def _normalize_uv(uv: np.ndarray, height: int, width: int) -> np.ndarray:
        uv_norm = np.asarray(uv, dtype=np.float32).copy()
        uv_norm[..., 0] = (uv_norm[..., 0] - width / 2.0) / (width / 2.0)
        uv_norm[..., 1] = (uv_norm[..., 1] - height / 2.0) / (height / 2.0)
        return uv_norm

    @staticmethod
    def _pack_tracks(result: dict[str, np.ndarray]) -> np.ndarray:
        points = result["points"].astype(np.float32)
        visibles = result["visibles"].astype(np.float32)[..., None]
        return np.concatenate([points, visibles], axis=-1)

    @staticmethod
    def _locator_points_to_pixels(points: list[tuple[float, float]], image_size: tuple[int, int]) -> list[list[float]]:
        height, width = image_size
        converted = []
        for x, y in points:
            x = float(x) / 1000.0 * width
            y = float(y) / 1000.0 * height
            converted.append([float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))])
        return converted

    @staticmethod
    def _largest_mask(masks: list[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        valid_masks = [np.asarray(mask, dtype=bool) for mask in masks if np.asarray(mask).any()]
        if not valid_masks:
            return np.zeros((height, width), dtype=bool)
        return max(valid_masks, key=lambda mask: int(mask.sum()))

    @staticmethod
    def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return mask
        mask_u8 = mask.astype(np.uint8)
        flood = (1 - mask_u8).copy()
        height, width = mask_u8.shape
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 0)
        holes = flood.astype(bool)
        return mask | holes

    def _array(self, value: Any, *, name: str, dtype: Any) -> np.ndarray:
        if isinstance(value, str | bytes):
            raise TypeError(f"{name} must be a JSON array, got {type(value).__name__}")
        array = np.asarray(value, dtype=dtype)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        return array


class EmbodimentAdapter:
    """Recover benchmark executable actions from transformed model outputs."""

    def __init__(self, *, future_horizon: int, robot_cls: type[Any]) -> None:
        self.future_horizon = future_horizon
        self.robot_cls = robot_cls
        self.robot: Any = None

    def to_action(
        self,
        outputs: Mapping[str, Any],
        model_input: Mapping[str, Any],
        request: Mapping[str, Any],
        session: InferenceSession,
    ) -> tuple[list[list[float]], list[float]]:
        if session.benchmark != "libero":
            raise ValueError(f"Unsupported benchmark: {session.benchmark!r}")
        required_outputs = {"point", "metric_depth", "gripper_openness", "gripper_action"}
        missing_outputs = required_outputs.difference(outputs)
        if missing_outputs:
            raise KeyError(f"Model outputs are missing required heads: {sorted(missing_outputs)}")

        points = np.asarray(outputs["point"], dtype=np.float32)[0]
        metric_depth = np.asarray(outputs["metric_depth"], dtype=np.float32)[0].reshape(-1)
        gripper_openness = np.asarray(outputs["gripper_openness"], dtype=np.float32)[0].reshape(-1)
        gripper_action = np.asarray(outputs["gripper_action"], dtype=np.float32)[0].reshape(-1)
        actor_query_frame_id = np.asarray(model_input["actor_query_frame_id"], dtype=np.int64)[0]
        object_id = np.asarray(model_input["object_id"], dtype=np.int64)[0]
        point_id = np.asarray(model_input["point_id"], dtype=np.int64)[0]
        frame_id = np.asarray(model_input["frame_id"], dtype=np.int64)[0]
        intrinsic = self._current_camera_matrix(request, "camera.intrinsics", (3, 3))
        extrinsic = self._current_camera_matrix(request, "camera.extrinsics", (4, 4))

        actions = []
        gripper_widths = []
        for future_frame_id in range(1, self.future_horizon + 1):
            mask = (object_id == 0) & (frame_id == future_frame_id)
            actor_points = points[mask]
            actor_metric_depth = metric_depth[mask]
            actor_point_ids = point_id[mask]
            by_id = {int(pid): actor_points[index] for index, pid in enumerate(actor_point_ids)}
            metric_by_id = {int(pid): actor_metric_depth[index] for index, pid in enumerate(actor_point_ids)}
            missing = [index for index in (0, 1, 2) if index not in by_id]
            if missing:
                raise ValueError(f"Missing actor point ids {missing} for future_frame_id={future_frame_id}")
            uvd_dict = {
                "root_uvd": np.asarray([by_id[0][0], by_id[0][1], metric_by_id[0]]),
                "left_base_uvd": np.asarray([by_id[1][0], by_id[1][1], metric_by_id[1]]),
                "right_base_uvd": np.asarray([by_id[2][0], by_id[2][1], metric_by_id[2]]),
            }
            self._validate_uvd(uvd_dict, future_frame_id=future_frame_id)
            actor_mask = actor_query_frame_id == future_frame_id
            if int(actor_mask.sum()) != 1:
                raise ValueError(f"Expected one actor query for future_frame_id={future_frame_id}")
            predicted_openness = float(np.clip(gripper_openness[actor_mask][0], 0.0, 1.0))
            predicted_action = float(np.clip(gripper_action[actor_mask][0], -1.0, 1.0))
            action = self._robot().project_uvd_to_gripper(
                uvd_dict,
                intrinsic=intrinsic,
                gripper_width=predicted_openness * LIBERO_GRIPPER_MAX_WIDTH,
                extrinsic=extrinsic,
            )
            gripper_widths.append(float(action[6]))
            action = self._to_libero_pose(action)
            action[6] = predicted_action
            actions.append(action.astype(np.float32).tolist())

        if len(actions) != self.future_horizon:
            raise RuntimeError(f"Expected {self.future_horizon} actions, got {len(actions)}")
        return actions, gripper_widths

    def _robot(self) -> Any:
        if self.robot is None:
            self.robot = self.robot_cls(embodiment="franka_panda", with_fingers=True)
        return self.robot

    def _to_libero_pose(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).copy()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = action[:3]
        pose[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()
        local_rotation = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        action_pose = pose @ local_rotation
        action[:3] = action_pose[:3, 3]
        action[3:6] = R.from_matrix(action_pose[:3, :3]).as_rotvec()
        return action

    @staticmethod
    def _current_camera_matrix(request: Mapping[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
        value = np.asarray(request[key], dtype=np.float64)
        if value.shape == shape:
            return value
        if value.ndim == 3 and value.shape[1:] == shape:
            return value[-1]
        raise ValueError(f"{key} must have shape {shape} or Tx{shape}, got {value.shape}")

    @staticmethod
    def _validate_uvd(uvd_dict: Mapping[str, np.ndarray], *, future_frame_id: int) -> None:
        for name, value in uvd_dict.items():
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape} at future_frame_id={future_frame_id}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values at future_frame_id={future_frame_id}: {value}")
            if float(value[2]) <= 1e-6:
                raise ValueError(f"{name} depth must be positive at future_frame_id={future_frame_id}: {value}")


class InferenceModel:
    """Model wrapper with optional BGE text condition encoding."""

    def __init__(
        self,
        *,
        model_kwargs: Mapping[str, Any],
        data_kwargs: Mapping[str, Any] | None = None,
        ckpt_path: str | Path,
        bge_path: str | Path | None = None,
        device: torch.device,
    ) -> None:
        self.device = device
        self.out_transforms = tuple((data_kwargs or {}).get("out_transforms", ()))
        self.bge_path = Path(bge_path) if bge_path is not None else Path(
            "/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5"
        )
        self.bge_tokenizer = None
        self.bge_model = None
        self.embedding_cache: dict[str, torch.Tensor] = {}

        self.model = PointQueryModel(**dict(model_kwargs)).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(TrainingCheckpoint.unwrap_model_state(state), strict=True)
        cs.print(f"[green]loaded checkpoint from {ckpt_path}[/green]")
        self.model.eval()

    @torch.inference_mode()
    def infer(
        self,
        input_data: Mapping[str, Any],
        *,
        return_model_input: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(input_data[name], device=self.device).to(dtype=dtype)

        def optional_tensor(name: str, dtype: torch.dtype) -> torch.Tensor | None:
            if name not in input_data or input_data[name] is None:
                return None
            return torch.as_tensor(input_data[name], device=self.device).to(dtype=dtype)

        object_condition = optional_tensor("object_condition", torch.float32)
        if object_condition is None and input_data.get("object_condition_texts") is not None:
            text_conditions = input_data["object_condition_texts"]
            if not isinstance(text_conditions, Sequence) or isinstance(text_conditions, str | bytes):
                raise TypeError("object_condition_texts must be a list")
            object_condition = torch.stack([self.encode_condition(item) for item in text_conditions], dim=0).unsqueeze(0)
        elif object_condition is not None and object_condition.ndim == 2:
            object_condition = object_condition.unsqueeze(0)

        actor_condition = optional_tensor("actor_condition", torch.float32)
        if actor_condition is None and input_data.get("actor_condition_text") is not None:
            actor_condition = self.encode_condition(input_data["actor_condition_text"]).unsqueeze(0).unsqueeze(0)
        elif actor_condition is not None and actor_condition.ndim == 2:
            actor_condition = actor_condition.unsqueeze(0)

        infer_inputs = {
            "point_feats": tensor("point_feats", torch.float32),
            "actor_feats": tensor("actor_feats", torch.float32),
            "object_id": tensor("object_id", torch.long),
            "point_id": tensor("point_id", torch.long),
            "frame_id": tensor("frame_id", torch.long),
            "object_condition": object_condition,
            "actor_condition": actor_condition,
            "query_type": optional_tensor("query_type", torch.long),
            "head_names": input_data.get("head_names"),
            "frame_query_frame_id": optional_tensor("frame_query_frame_id", torch.long),
            "frame_query_type": optional_tensor("frame_query_type", torch.long),
            "frame_head_names": input_data.get("frame_head_names"),
            "actor_query_frame_id": optional_tensor("actor_query_frame_id", torch.long),
            "actor_query_type": optional_tensor("actor_query_type", torch.long),
            "actor_head_names": input_data.get("actor_head_names"),
        }
        outputs = self.model.infer(
            **infer_inputs,
            return_residual=bool(input_data.get("return_residual", False)),
        )
        output_data = {"outputs": outputs, "batch": infer_inputs}
        for transform in self.out_transforms:
            output_data = transform(output_data)
        json_outputs = self.to_json(output_data["outputs"])
        if not return_model_input:
            return json_outputs
        return json_outputs, self.to_json(infer_inputs)

    def encode_condition(self, value: Any) -> torch.Tensor:
        if isinstance(value, Mapping):
            parts = [value.get("role"), value.get("action_type"), value.get("action_degree")]
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) > 3:
                raise ValueError("text condition sequence must have at most 3 items")
            parts = list(value) + [None] * (3 - len(value))
        else:
            raise TypeError("text condition must be a dict or [role, action_type, action_degree]")
        return torch.cat([self.embed_text(None if part is None else str(part)) for part in parts], dim=0)

    def embed_text(self, text: str | None) -> torch.Tensor:
        if self.bge_model is None:
            from transformers import AutoModel, AutoTokenizer

            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_path)
            self.bge_model = AutoModel.from_pretrained(self.bge_path).to(self.device)
            self.bge_model.eval()
            cs.print(f"[green]loaded BGE from {self.bge_path}[/green]")

        assert self.bge_tokenizer is not None and self.bge_model is not None
        dim = int(self.bge_model.config.hidden_size)
        if text is None or text == "":
            return torch.zeros(dim, dtype=torch.float32, device=self.device)
        if text not in self.embedding_cache:
            batch = self.bge_tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.bge_model(**batch)
            embedding = F.normalize(output.last_hidden_state[:, 0], p=2, dim=1)[0]
            self.embedding_cache[text] = embedding.detach().float()
        return self.embedding_cache[text].to(self.device)

    def to_json(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {key: self.to_json(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [self.to_json(item) for item in value]
        if isinstance(value, list):
            return [self.to_json(item) for item in value]
        return value


class InferenceServer:
    """WebSocket orchestrator for observation-driven inference."""

    def __init__(
        self,
        host: str,
        port: int,
        planner: TopLevelTaskPlanner,
        preprocessor: InputPreprocessor,
        inference: InferenceModel,
        embodiment: EmbodimentAdapter,
    ) -> None:
        self.host = host
        self.port = port
        self.planner = planner
        self.preprocessor = preprocessor
        self.inference = inference
        self.embodiment = embodiment
        self.sessions: dict[str, InferenceSession] = {}
        self.idle_timeout = 180.0
        self.last_message_time = 0.0

    def serve_forever(self) -> None:
        asyncio.run(self.serve())

    async def serve(self) -> None:
        cs.print(f"[green]GraphVLA inference server listening on ws://{self.host}:{self.port}[/green]")
        cs.print("WebSocket messages must be flat observation requests; replies are model fields plus action.")
        self.last_message_time = asyncio.get_running_loop().time()
        async with websockets.serve(self.handle_connection, self.host, self.port, max_size=None):
            await self.wait_for_idle_timeout()

    async def wait_for_idle_timeout(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            idle_seconds = asyncio.get_running_loop().time() - self.last_message_time
            if idle_seconds >= self.idle_timeout:
                cs.print(f"[yellow]No WebSocket message received for {self.idle_timeout:.0f}s; shutting down.[/yellow]")
                return

    async def handle_connection(self, websocket: websockets.ServerConnection) -> None:
        peer = websocket.remote_address
        cs.print(f"{peer} - connected")
        async for message in websocket:
            self.last_message_time = asyncio.get_running_loop().time()
            try:
                request = json.loads(message)
                if not isinstance(request, Mapping):
                    raise TypeError("request message must be a JSON object")
                response = self.infer_from_observation(request)
            except Exception as exc:
                response = {"error": str(exc)}
            await websocket.send(json.dumps(response))

    def infer_from_observation(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_request(request)
        session = self._session_for(request)
        execute_chunk_len = int(request["execute_chunk_len"])
        subtaskstructure = self.planner.plan(request, session)
        model_input = self.preprocessor.build(request, session, subtaskstructure)
        response_robobrain_points, response_robobrain_object_id = self._active_robobrain_response(session)
        response_subtask = session.current_subtask
        response_subtask_index = session.subtask_index

        return_model_input = bool(request.get("return_model_input", False))
        inference_result = self.inference.infer(model_input, return_model_input=return_model_input)
        if return_model_input:
            outputs, captured_model_input = inference_result
            captured_model_input["object_condition_texts"] = model_input.get("object_condition_texts")
            captured_model_input["actor_condition_text"] = model_input.get("actor_condition_text")
        else:
            outputs = inference_result
            captured_model_input = None
        actions, gripper_widths = self.embodiment.to_action(outputs, model_input, request, session)
        action_chunk = actions[:execute_chunk_len]
        gripper_width_chunk = gripper_widths[:execute_chunk_len]
        subtask_switched = self.planner.update_after_inference(
            outputs,
            session,
            model_input,
            gripper_width_chunk,
            action_chunk,
        )
        response = {
            **outputs,
            "object_id": self.inference.to_json(model_input["object_id"]),
            "point_id": self.inference.to_json(model_input["point_id"]),
            "frame_id": self.inference.to_json(model_input["frame_id"]),
            "frame_query_frame_id": self.inference.to_json(model_input["frame_query_frame_id"]),
            "actor_query_frame_id": self.inference.to_json(model_input["actor_query_frame_id"]),
            "input_point": self.inference.to_json(model_input["input_point"]),
            "input_object_id": self.inference.to_json(model_input["input_object_id"]),
            "input_point_id": self.inference.to_json(model_input["input_point_id"]),
            "input_frame_id": self.inference.to_json(model_input["input_frame_id"]),
            "robobrain_point": self.inference.to_json(response_robobrain_points),
            "robobrain_object_id": self.inference.to_json(response_robobrain_object_id),
            "subtask": response_subtask,
            "subtask_index": response_subtask_index,
            "subtask_switched": subtask_switched,
            "action": action_chunk,
        }
        if captured_model_input is not None:
            response["model_input"] = captured_model_input
        return response

    @staticmethod
    def _active_robobrain_response(session: InferenceSession) -> tuple[np.ndarray | None, np.ndarray | None]:
        if session.robobrain_points is None or not session.active_object_indices:
            return None, None
        points_by_object = np.asarray(session.robobrain_points, dtype=np.float32)
        point_chunks = []
        id_chunks = []
        for local_id, object_index in enumerate(session.active_object_indices, start=1):
            if object_index < 0 or object_index >= len(points_by_object):
                continue
            points = points_by_object[object_index]
            point_chunks.append(points)
            id_chunks.append(np.full(len(points), local_id, dtype=np.int64))
        if not point_chunks:
            return None, None
        return np.concatenate(point_chunks, axis=0), np.concatenate(id_chunks, axis=0)

    def _validate_request(self, request: Mapping[str, Any]) -> None:
        missing = [key for key in REQUIRED_REQUEST_FIELDS if key not in request]
        if missing:
            raise KeyError(f"Missing required request fields: {missing}")
        if request["benchmark"] != "libero":
            raise ValueError(f"Unsupported benchmark: {request['benchmark']!r}")
        execute_chunk_len = request["execute_chunk_len"]
        if isinstance(execute_chunk_len, bool) or not isinstance(execute_chunk_len, int):
            raise TypeError("execute_chunk_len must be an integer")
        if not 1 <= execute_chunk_len <= self.embodiment.future_horizon:
            raise ValueError(
                f"execute_chunk_len must be in [1, {self.embodiment.future_horizon}], got {execute_chunk_len}"
            )

    def _session_for(self, request: Mapping[str, Any]) -> InferenceSession:
        session_id = str(request["session_id"])
        benchmark = str(request["benchmark"])
        language = str(request["language"])
        session = self.sessions.get(session_id)
        if session is None:
            session = InferenceSession(session_id=session_id, benchmark=benchmark, language=language)
            self.sessions[session_id] = session
            return session

        if bool(request.get("reset", False)) or session.language != language or session.benchmark != benchmark:
            session.reset(benchmark=benchmark, language=language)
        return session


def parse_devices(value: str | None, *, default_device: str) -> dict[str, str]:
    devices = {
        "default": default_device,
        "inference": default_device,
        "sam3": default_device,
        "point_tracker": default_device,
        "depth_predictor": default_device,
        "node_locator": default_device,
    }
    if value is None or value == "":
        return devices
    overrides = json.loads(value)
    if not isinstance(overrides, Mapping):
        raise TypeError("--devices must be a JSON object")
    devices.update({str(key): str(device) for key, device in overrides.items()})
    return devices


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Serve GraphVLA observation inference over WebSocket.")
    parser.add_argument("--example", default="libero", choices=("libero",))
    parser.add_argument("--ckpt-path", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--bge-path",
        default="/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5",
        help="Path to the BGE model used for text conditions.",
    )
    parser.add_argument(
        "--task-analyzer-api-key",
        default=None,
        help="API key used when a taskstructure is missing from the dataset cache.",
    )
    parser.add_argument(
        "--complete-threshold",
        type=float,
        default=0.5,
        help="Frame completion score threshold for subtask switching.",
    )
    parser.add_argument(
        "--complete-window",
        type=int,
        default=3,
        help="Number of consecutive completed frames required before switching subtasks.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help='JSON device map for server modules, e.g. {"inference":"cuda:0","sam3":"cuda:1"}.',
    )
    namespace = parser.parse_args()
    namespace.devices = parse_devices(namespace.devices, default_device=namespace.device)
    return Args(**vars(namespace))


def main() -> None:
    import os
    if os.environ.get("DEBUG", "0") == "1":
        import debugpy
        port = 10092  # 与launch.json中的一致
        debugpy.listen(("0.0.0.0", port))
        print(f"🔍 Rank 0 waiting for debugger attach on port {port}...")
        debugpy.wait_for_client()

    args = parse_args()
    cs.print(args)
    if args.example == "libero":
        from examples.libero.config.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
        from examples.libero.embodiment.robot import GeomRobot

        model_kwargs = LIBERO_MODEL_CONFIG.to_kwargs()
        data_kwargs = LIBERO_DATA_CONFIG.to_kwargs()
        history_horizon = abs(int(LIBERO_MODEL_CONFIG.min_frame))
        future_horizon = int(LIBERO_MODEL_CONFIG.max_frame)
    else:
        raise ValueError(f"Unsupported example: {args.example}")
        
    server = InferenceServer(
        host=args.host,
        port=args.port,
        planner=TopLevelTaskPlanner(
            dataset_dir=data_kwargs["dataset_dir"],
            task_analyzer_api_key=args.task_analyzer_api_key,
            complete_threshold=args.complete_threshold,
            complete_window=args.complete_window,
        ),
        preprocessor=InputPreprocessor(
            history_horizon=history_horizon,
            future_horizon=future_horizon,
            num_points=model_kwargs["num_points"],
            robot_cls=GeomRobot,
            dataset_dir=data_kwargs["dataset_dir"],
            devices=args.devices,
        ),
        inference = InferenceModel(
            model_kwargs=model_kwargs,
            data_kwargs=data_kwargs,
            ckpt_path=args.ckpt_path,
            device=torch.device(args.devices["inference"]),
            bge_path=args.bge_path,
        ),
        embodiment=EmbodimentAdapter(future_horizon=future_horizon, robot_cls=GeomRobot),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
