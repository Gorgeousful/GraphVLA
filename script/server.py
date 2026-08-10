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
import random
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

from src.model.model import FLOW_MODES, GraphFlowModel
from src.training.checkpoint import TrainingCheckpoint
from src.module.task_analyzer import TaskAnalyzer
from src.module.node_locator import NodeLocatorLA, NodeLocatorRobo
from src.module.node_segmenter import NodeSegmenter, NodeSegmenterSAM2
from src.module.point_tracker import PointTracker
from src.common.geom_utils import sample_points_from_mask
from src.common.schema import (
    ACTION_DIM,
    GRIPPER_NUM_POINTS,
    LIBERO_GRIPPER_MAX_WIDTH,
    POINT_FEATURE_DIM,
    taskstructure_to_json,
    validate_actor_point_indices,
)


cs = Console()

LIBERO_DELTA_POSITION_SCALE = 0.05
REQUIRED_REQUEST_FIELDS = (
    "benchmark",
    "session_id",
    "language",
    "observation.images.image",
    "observation.depth.metric",
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
    locator: str = "locateanything"
    locator_mode: str = "point"
    locator_scale: float = 1.0
    segmenter: str = "sam2"
    sam_only: bool = False
    execute_chunk_len: int = 5
    release_lift_height: float = 0.05
    seed: int = 42
    complete_threshold: float = 0.5
    complete_window: int = 3
    devices: dict[str, str] = field(default_factory=dict)


@dataclass
class ObservationFrame:
    image: np.ndarray
    metric_depth: np.ndarray
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
    release_pending: bool = False
    complete_streak: int = 0
    feature_history: list[dict[str, np.ndarray]] = field(default_factory=list)
    frame_index: int = 0
    object_nodes: list[dict[str, Any]] = field(default_factory=list)
    active_object_indices: list[int] = field(default_factory=list)
    object_segmenter: Any = None
    point_tracker: Any = None
    tracked_points: np.ndarray | None = None
    initial_points: np.ndarray | None = None
    object_to_unique_indices: list[int] = field(default_factory=list)

    def reset(self, *, benchmark: str, language: str) -> None:
        self.benchmark = benchmark
        self.language = language
        self.current_subtask = None
        self.taskstructure = None
        self.subtask_index = 0
        self.task_complete = False
        self.release_pending = False
        self.complete_streak = 0
        self.reset_preprocessor()

    def reset_preprocessor(self) -> None:
        self.feature_history.clear()
        self.frame_index = 0
        self.object_nodes.clear()
        self.active_object_indices.clear()
        self.object_segmenter = None
        self.point_tracker = None
        self.tracked_points = None
        self.initial_points = None
        self.object_to_unique_indices.clear()


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
    ) -> bool:
        if session.task_complete or session.release_pending or session.taskstructure is None:
            return False

        frame_scores = self._completion_frame_scores(outputs)
        subtasks = session.taskstructure.get("subtasks", [])
        subtask_label = f"subtask [{session.subtask_index + 1}/{len(subtasks)}]"
        for frame_id, score in frame_scores:
            score_text = (
                f"step={session.frame_index} {subtask_label} f={frame_id} "
                f"complete_score={score:.4f}"
            )
            if score >= self.complete_threshold:
                cs.print(f"[green]{score_text}[/green]")
                session.complete_streak += 1
            else:
                cs.print(score_text)
                session.complete_streak = 0
            if session.complete_streak >= self.complete_window:
                session.complete_streak = 0
                session.release_pending = True
                return True
        return False

    def advance_after_release(self, session: InferenceSession) -> bool:
        if not session.release_pending:
            return False

        subtasks = session.taskstructure.get("subtasks", [])
        session.release_pending = False
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
    ) -> list[tuple[int, float]]:
        score = self._output_score(outputs, "is_complete")
        return [] if score is None else [(0, score)]

    @staticmethod
    def _contact_score_text(outputs: Mapping[str, Any]) -> str:
        score = TopLevelTaskPlanner._output_score(outputs, "is_contact")
        return "contact_score=-" if score is None else f"contact_score={score:.3f}"

    @staticmethod
    def _output_score(outputs: Mapping[str, Any], name: str) -> float | None:
        value = outputs.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        scores = np.asarray(value, dtype=float)
        if scores.ndim >= 2 and scores.shape[0] == 1:
            scores = scores[0]
        if scores.ndim == 0:
            scores = scores.reshape(1)
        else:
            scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        return float(scores.max())


class InputPreprocessor:
    """Build model inputs from flat observation requests."""

    def __init__(
        self,
        *,
        history_horizon: int,
        future_horizon: int,
        num_points: int,
        actor_point_indices: tuple[int, ...],
        robot_cls: type[Any],
        dataset_dir: str | Path,
        locator: str = "locateanything",
        locator_mode: str = "point",
        locator_scale: float = 1.0,
        segmenter: str = "sam2",
        sam_only: bool = False,
        devices: Mapping[str, str] | None = None,
    ) -> None:
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.num_points = num_points
        self.actor_point_indices = validate_actor_point_indices(actor_point_indices)
        self.actor_num_points = len(self.actor_point_indices)
        if self.num_points < self.actor_num_points:
            raise ValueError(
                f"num_points must be at least {self.actor_num_points}, got {self.num_points}"
            )
        self.robot_cls = robot_cls
        self.locator = locator
        self.locator_mode = locator_mode
        self.locator_scale = locator_scale
        self.segmenter = segmenter
        self.sam_only = sam_only
        self.devices = dict(devices or {})
        self.norm_stats = self._load_norm_stats(Path(dataset_dir))
        self.node_segmenter_model = None
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

        gripper_points_xyz = self._state_to_gripper_points_xyz(frame)
        if session.tracked_points is None:
            raise RuntimeError("tracked_points is missing after perception update")
        features = {
            "tracks": np.asarray(session.tracked_points, dtype=np.float32),
            "metric_depth": np.asarray(frame.metric_depth, dtype=np.float32),
            "gripper_points_xyz": np.asarray(gripper_points_xyz, dtype=np.float32),
            "intrinsic": np.asarray(frame.intrinsic, dtype=np.float32),
        }
        session.frame_index += 1
        return features

    def _initialize_perception(self, session: InferenceSession, frame: ObservationFrame) -> None:
        if session.taskstructure is None:
            raise RuntimeError("taskstructure is missing before perception initialization")
        session.object_nodes = self._task_object_nodes(session.taskstructure)
        perception_nodes, session.object_to_unique_indices = self._unique_nodes_by_name(
            session.object_nodes
        )
        point_prompts = None
        box_prompts = None
        if perception_nodes:
            locator_cls = NodeLocatorLA if self.locator == "locateanything" else NodeLocatorRobo
            node_locator = locator_cls(device_map=self._device("node_locator"))
            try:
                if self.locator_mode == "box":
                    box_prompts = []
                    initial_points = []
                    for node in perception_nodes:
                        box = self._locate_node_box(node_locator, frame.image, node["name"])
                        cs.print(
                            f"node locator node={node['name']} pixel_xyxy={np.round(box, 1).tolist()}",
                            markup=False,
                        )
                        box_prompts.append(box)
                        initial_points.append([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
                    session.initial_points = np.asarray(initial_points, dtype=np.float32)
                else:
                    point_prompts = []
                    for node in perception_nodes:
                        points = self._locate_node_points(node_locator, frame.image, node["name"])
                        cs.print(
                            f"node locator node={node['name']} pixel_xy={np.round(points, 1).tolist()}",
                            markup=False,
                        )
                        point_prompts.append(points)
                    session.initial_points = np.asarray(point_prompts, dtype=np.float32)
                session.initial_points = session.initial_points[
                    session.object_to_unique_indices
                ]
            finally:
                del node_locator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        node_segmenter_device = self._device("node_segmenter")
        segmenter_cls = NodeSegmenterSAM2 if self.segmenter == "sam2" else NodeSegmenter
        if self.node_segmenter_model is None:
            self.node_segmenter_model = segmenter_cls(device=node_segmenter_device).model
        session.object_segmenter = segmenter_cls(
            device=node_segmenter_device,
            model=self.node_segmenter_model,
        )

        if perception_nodes:
            object_masks = session.object_segmenter.predict(
                frame.image,
                points=point_prompts,
                boxes=box_prompts,
                anchor_frame=True,
            )
            if self.sam_only:
                unique_tracks = self._tracks_from_masks(
                    object_masks, frame.image.shape[:2], len(perception_nodes)
                )
                session.tracked_points = unique_tracks[session.object_to_unique_indices]
                return

            session.point_tracker = PointTracker(device=self._device("point_tracker"))
            object_points = np.stack(
                [sample_points_from_mask(mask, num_points=self.num_points) for mask in object_masks],
                axis=0,
            ).astype(np.float32)
            track_result = session.point_tracker.track(frame.image, points=object_points, anchor_frame=True)
            unique_tracks = self._pack_tracks(track_result)
            session.tracked_points = unique_tracks[session.object_to_unique_indices]
        else:
            session.tracked_points = np.zeros((0, self.num_points, 3), dtype=np.float32)

    def _update_perception(self, session: InferenceSession, frame: ObservationFrame) -> None:
        if session.object_segmenter is not None and session.object_nodes:
            object_masks = session.object_segmenter.predict(frame.image, anchor_frame=False)
            if self.sam_only:
                unique_count = len(set(session.object_to_unique_indices))
                unique_tracks = self._tracks_from_masks(
                    object_masks, frame.image.shape[:2], unique_count
                )
                session.tracked_points = unique_tracks[session.object_to_unique_indices]
                return
        if session.point_tracker is not None and session.object_nodes:
            unique_tracks = self._pack_tracks(
                session.point_tracker.track(frame.image, anchor_frame=False)
            )
            session.tracked_points = unique_tracks[session.object_to_unique_indices]

    @staticmethod
    def _unique_nodes_by_name(
        nodes: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], list[int]]:
        unique_nodes = []
        unique_index_by_name = {}
        object_to_unique_indices = []
        for node in nodes:
            name = str(node.get("name", ""))
            if name not in unique_index_by_name:
                unique_index_by_name[name] = len(unique_nodes)
                unique_nodes.append(node)
            object_to_unique_indices.append(unique_index_by_name[name])
        return unique_nodes, object_to_unique_indices

    def _tracks_from_masks(
        self,
        masks: Sequence[np.ndarray],
        image_shape: tuple[int, int],
        expected_count: int,
    ) -> np.ndarray:
        if len(masks) != expected_count:
            raise RuntimeError(
                f"{self.segmenter} returned {len(masks)} masks for "
                f"{expected_count} unique node names"
            )
        tracks = np.zeros((expected_count, self.num_points, 3), dtype=np.float32)
        for index, mask in enumerate(masks):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != image_shape:
                raise ValueError(
                    f"mask {index} has shape {mask.shape}, expected {image_shape}"
                )
            if mask.any():
                tracks[index, :, :2] = sample_points_from_mask(
                    mask, num_points=self.num_points, erode_pixel=2
                )
                tracks[index, :, 2] = 1.0
        return tracks

    def _state_to_gripper_points_xyz(self, frame: ObservationFrame) -> np.ndarray:
        state = frame.state
        if state.size < 8:
            raise ValueError(f"observation.state must contain at least 8 values, got {state.size}")
        gripper_width = abs(float(state[6])) + abs(float(state[7]))
        return self._robot().project_gripper_to_xyz(
            tcp_state=state[:6],
            extrinsic=frame.extrinsic,
            gripper_width=gripper_width,
        )

    def _build_model_input(
        self,
        session: InferenceSession,
        frames: list[dict[str, np.ndarray]],
        subtaskstructure: Mapping[str, Any],
    ) -> dict[str, Any]:
        height, width = frames[-1]["metric_depth"].shape
        session.active_object_indices = self._subtask_object_indices(session, subtaskstructure)
        active_roles = [str(node.get("role", "")) for node in self._object_nodes(subtaskstructure)][:2]
        object_points = np.stack(
            [
                self._object_feats(
                    self._active_tracks(item["tracks"], session.active_object_indices),
                    active_roles, item["metric_depth"], frames[-1]["intrinsic"],
                    height,
                    width,
                )
                for item in frames
            ],
            axis=0,
        )
        actor_xyz = np.stack([
            self._normalize_field(item["gripper_points_xyz"], "camera_xyz")[
                list(self.actor_point_indices)
            ]
            for item in frames
        ], axis=0)
        full_gripper_xyz = np.stack(
            [item["gripper_points_xyz"] for item in frames], axis=0,
        )
        entity_points = np.zeros(
            (len(frames), 3, self.num_points, POINT_FEATURE_DIM), dtype=np.float32
        )
        entity_mask = np.zeros((len(frames), 3, self.num_points), dtype=bool)
        entity_points[:, 0, :self.actor_num_points] = actor_xyz
        entity_mask[:, 0, :self.actor_num_points] = True
        entity_points[:, 1:3] = object_points
        object_valid = np.any(object_points != 0, axis=(-1, -2))
        for object_index in range(2):
            entity_mask[:, object_index + 1] = object_valid[:, object_index, None]
        action_type = str(subtaskstructure.get("action_type", ""))
        action_degree = subtaskstructure.get("action_degree")
        scene_condition_texts = [action_type, action_degree]
        gripper_width = np.linalg.norm(
            full_gripper_xyz[:, 3] - full_gripper_xyz[:, 4], axis=-1,
        )
        openness = np.clip(gripper_width / LIBERO_GRIPPER_MAX_WIDTH, 0.0, 1.0)
        closedness = (1.0 - 2.0 * openness)[:, None].astype(np.float32)
        return {
            "entity_points": entity_points[None].tolist(),
            "entity_point_mask": entity_mask[None].tolist(),
            "gripper_closedness_history": closedness[None].tolist(),
            "scene_condition_texts": scene_condition_texts,
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
        metric_depths = self._as_frame_sequence(
            self._array(request["observation.depth.metric"], name="observation.depth.metric", dtype=np.float32),
            name="observation.depth.metric",
            single_ndim={2},
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

        target_len = max(len(images), len(metric_depths), len(states), len(intrinsics), len(extrinsics))
        images = self._broadcast_frames(images, target_len, name="observation.images.image")
        metric_depths = self._broadcast_frames(metric_depths, target_len, name="observation.depth.metric")
        states = self._broadcast_frames(states, target_len, name="observation.state")
        intrinsics = self._broadcast_frames(intrinsics, target_len, name="camera.intrinsics")
        extrinsics = self._broadcast_frames(extrinsics, target_len, name="camera.extrinsics")

        return [
            ObservationFrame(
                image=np.ascontiguousarray(image),
                metric_depth=np.ascontiguousarray(metric_depth),
                state=state,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )
            for image, metric_depth, state, intrinsic, extrinsic in zip(
                images, metric_depths, states, intrinsics, extrinsics, strict=True
            )
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
        node_locator: Any,
        image: np.ndarray,
        node_name: str,
    ) -> list[list[float]]:
        result = node_locator.inference(
            text=node_name,
            image=Image.fromarray(image),
            resize_scale=self.locator_scale,
        )
        points = result.get("points") or []
        if not points:
            raise RuntimeError(f"Node locator found no points for node={node_name!r}")
        return self._locator_points_to_pixels(points, image.shape[:2])

    def _locate_node_box(
        self,
        node_locator: Any,
        image: np.ndarray,
        node_name: str,
    ) -> list[float]:
        result = node_locator.inference(
            text=node_name,
            image=Image.fromarray(image),
            task="grounding",
            resize_scale=self.locator_scale,
        )
        boxes = result.get("boxes") or []
        if not boxes:
            raise RuntimeError(f"Node locator found no boxes for node={node_name!r}")
        return self._locator_boxes_to_pixels(boxes, image.shape[:2])[0]

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
            if role != "actor":
                nodes.append(dict(node))
        return nodes

    def _task_object_nodes(self, taskstructure: Mapping[str, Any]) -> list[dict[str, Any]]:
        nodes = []
        for subtask in taskstructure.get("subtasks", []):
            if isinstance(subtask, Mapping):
                nodes.extend(self._object_nodes(subtask))
        return nodes

    def _subtask_object_indices(
        self,
        session: InferenceSession,
        subtaskstructure: Mapping[str, Any],
    ) -> list[int]:
        if session.taskstructure is None:
            raise RuntimeError("taskstructure is required to resolve subtask object indices")
        subtasks = session.taskstructure.get("subtasks", [])
        object_start = sum(
            len(self._object_nodes(subtask))
            for subtask in subtasks[:session.subtask_index]
            if isinstance(subtask, Mapping)
        )
        object_count = len(self._object_nodes(subtaskstructure))
        return list(range(object_start, object_start + object_count))[:2]

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

    def _object_feats(
        self,
        tracks: np.ndarray,
        roles: Sequence[str],
        depth: np.ndarray,
        intrinsic: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        points = np.zeros((2, self.num_points, POINT_FEATURE_DIM), dtype=np.float32)
        count = min(tracks.shape[0], 2)
        if count == 0:
            return points
        for source_index, role in enumerate(roles[:count]):
            target_index = 0 if role == "patient" else (1 if role == "target" else source_index)
            if target_index >= 2:
                continue
            uv = tracks[source_index, :, :2]
            sampled_depth, in_bounds = self._sample_depth(depth, uv, height, width)
            valid = (
                (tracks[source_index, :, 2] > 0.5)
                & in_bounds[:, 0].astype(bool)
                & np.isfinite(sampled_depth[:, 0])
                & (sampled_depth[:, 0] > 0.0)
            )
            valid_indices = np.flatnonzero(valid)
            if valid_indices.size == 0:
                continue
            selected = np.resize(valid_indices, self.num_points)
            selected_uv = uv[selected]
            z = sampled_depth[selected, 0]
            xyz = np.stack([
                (selected_uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                (selected_uv[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                z,
            ], axis=-1)
            points[target_index] = self._normalize_field(xyz, "camera_xyz")
        return points

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
    def _locator_boxes_to_pixels(boxes: list[list[float]], image_size: tuple[int, int]) -> list[list[float]]:
        height, width = image_size
        converted = []
        for x1, y1, x2, y2 in boxes:
            xs = sorted((float(x1) / 1000.0 * width, float(x2) / 1000.0 * width))
            ys = sorted((float(y1) / 1000.0 * height, float(y2) / 1000.0 * height))
            converted.append([
                float(np.clip(xs[0], 0, width - 1)),
                float(np.clip(ys[0], 0, height - 1)),
                float(np.clip(xs[1], 0, width - 1)),
                float(np.clip(ys[1], 0, height - 1)),
            ])
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
    """Convert routed model plans into executable LIBERO world-frame actions."""

    def __init__(
        self,
        *,
        future_horizon: int,
        actor_point_indices: tuple[int, ...],
        action_delta: bool = True,
        flow_mode: str = "joint",
        robot_cls: type[Any] | None = None,
        release_lift_height: float = 0.05,
    ) -> None:
        if flow_mode not in FLOW_MODES:
            raise ValueError(f"Unsupported flow mode: {flow_mode!r}")
        if flow_mode == "point_only" and action_delta:
            raise ValueError("point_only server execution requires action_delta=False")
        if flow_mode == "point_only" and robot_cls is None:
            raise ValueError("point_only server execution requires robot_cls")
        self.future_horizon = future_horizon
        self.actor_point_indices = validate_actor_point_indices(actor_point_indices)
        self.actor_num_points = len(self.actor_point_indices)
        self.action_delta = bool(action_delta)
        self.flow_mode = flow_mode
        self.robot_cls = robot_cls
        if release_lift_height < 0:
            raise ValueError("release_lift_height must be non-negative")
        self.release_lift_height = float(release_lift_height)
        self.robot: Any = None

    def to_action(
        self,
        outputs: Mapping[str, Any],
        request: Mapping[str, Any],
        session: InferenceSession,
    ) -> list[list[float]]:
        if session.benchmark != "libero":
            raise ValueError(f"Unsupported benchmark: {session.benchmark!r}")
        if self.flow_mode == "point_only":
            return self._point_plan_to_action(outputs, request, session)
        if "action_plan" not in outputs:
            raise KeyError("Model outputs are missing required head: action_plan")

        camera_actions = np.asarray(outputs["action_plan"], dtype=np.float64)
        expected_shape = (1, self.future_horizon, ACTION_DIM)
        if camera_actions.shape != expected_shape:
            raise ValueError(
                f"action_plan must have shape {expected_shape}, got {camera_actions.shape}"
            )
        camera_actions = camera_actions[0]
        if not np.isfinite(camera_actions).all():
            raise ValueError("action_plan contains non-finite values")

        extrinsic = self._current_camera_matrix(request, "camera.extrinsics", (4, 4))
        camera_to_world_rotation = extrinsic[:3, :3]
        world_actions = camera_actions.copy()
        if self.action_delta:
            world_actions[:, :3] = camera_actions[:, :3] @ camera_to_world_rotation.T
            world_actions[:, 3:6] = camera_actions[:, 3:6] @ camera_to_world_rotation.T
            world_actions = np.clip(world_actions, -1.0, 1.0)
        else:
            world_actions[:, :3] = (
                camera_actions[:, :3] @ camera_to_world_rotation.T + extrinsic[:3, 3]
            )
            for index, camera_rotvec in enumerate(camera_actions[:, 3:6]):
                world_rotation = camera_to_world_rotation @ R.from_rotvec(camera_rotvec).as_matrix()
                world_actions[index, 3:6] = R.from_matrix(world_rotation).as_rotvec()

        for action in world_actions:
            action[6] = self._gripper_command(float(action[6]))
        return world_actions.astype(np.float32).tolist()

    def _point_plan_to_action(
        self,
        outputs: Mapping[str, Any],
        request: Mapping[str, Any],
        session: InferenceSession,
    ) -> list[list[float]]:
        required = {"point_plan", "point_plan_mask", "gripper_plan"}
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"Model outputs are missing required heads: {sorted(missing)}")

        point_plan = np.asarray(outputs["point_plan"], dtype=np.float64)
        if (
            point_plan.ndim != 4
            or point_plan.shape[0] != 1
            or point_plan.shape[1] != self.future_horizon
            or point_plan.shape[2] != self.actor_num_points
            or point_plan.shape[3] != POINT_FEATURE_DIM
        ):
            raise ValueError(
                "point_plan must have shape "
                f"[1,{self.future_horizon},{self.actor_num_points},{POINT_FEATURE_DIM}], "
                f"got {point_plan.shape}"
            )
        actor_plan = point_plan[0]
        if not np.isfinite(actor_plan).all():
            raise ValueError("point_plan actor keypoints contain non-finite values")

        point_mask = np.asarray(outputs["point_plan_mask"], dtype=bool)
        if point_mask.shape != point_plan.shape[:-1]:
            raise ValueError(
                f"point_plan_mask shape {point_mask.shape} does not match {point_plan.shape[:-1]}"
            )
        if not point_mask.all():
            raise ValueError("point_plan contains invalid actor keypoints")

        gripper_plan = np.asarray(outputs["gripper_plan"], dtype=np.float64)
        expected_gripper_shape = (1, self.future_horizon)
        if gripper_plan.shape != expected_gripper_shape:
            raise ValueError(
                f"gripper_plan must have shape {expected_gripper_shape}, got {gripper_plan.shape}"
            )
        if not np.isfinite(gripper_plan).all():
            raise ValueError("gripper_plan contains non-finite values")

        state = np.asarray(request["observation.state"], dtype=np.float64)
        if state.ndim == 2:
            state = state[-1]
        if state.size < 8:
            raise ValueError(
                f"point_only execution requires observation.state with gripper fingers, got {state.shape}"
            )
        current_width = abs(float(state[6])) + abs(float(state[7]))
        extrinsic = self._current_camera_matrix(request, "camera.extrinsics", (4, 4))
        robot = self._robot()

        actions = []
        for future_index, points in enumerate(actor_plan):
            pose = robot.project_actor_xyz_to_gripper(
                points,
                gripper_width=current_width,
                extrinsic=extrinsic,
                actor_point_indices=self.actor_point_indices,
            )
            action = np.asarray(pose, dtype=np.float64)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError(
                    f"Invalid point-only recovered action at index {future_index}: {action}"
                )
            predicted_command = float(gripper_plan[0, future_index])
            action[6] = self._gripper_command(predicted_command)
            actions.append(action.astype(np.float32).tolist())

        return actions

    def _robot(self) -> Any:
        if self.robot is None:
            assert self.robot_cls is not None
            self.robot = self.robot_cls(embodiment="franka_panda", with_fingers=True)
        return self.robot

    @staticmethod
    def _gripper_command(predicted_command: float) -> float:
        return float(np.clip(predicted_command, -1.0, 1.0))

    def release_actions(
        self,
        session: InferenceSession,
        chunk_len: int,
        request: Mapping[str, Any] | None = None,
    ) -> list[list[float]]:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        if not self.action_delta:
            if request is None:
                raise ValueError("absolute release action requires the current observation request")
            state = np.asarray(request["observation.state"], dtype=np.float32)
            if state.ndim == 2:
                state = state[-1]
            if state.size < 6:
                raise ValueError(f"observation.state must contain a 6-D TCP pose, got {state.shape}")
            action[:6] = state[:6]
        action[6] = -1.0
        actions = np.repeat(action[None], chunk_len, axis=0)
        if chunk_len > 0:
            if self.action_delta:
                actions[:, 2] = np.clip(
                    self.release_lift_height / chunk_len / LIBERO_DELTA_POSITION_SCALE,
                    -1.0,
                    1.0,
                )
            else:
                actions[:, 2] += np.linspace(
                    self.release_lift_height / chunk_len,
                    self.release_lift_height,
                    chunk_len,
                    dtype=np.float32,
                )
        return actions.tolist()

    @staticmethod
    def _current_camera_matrix(
        request: Mapping[str, Any],
        key: str,
        shape: tuple[int, int],
    ) -> np.ndarray:
        value = np.asarray(request[key], dtype=np.float64)
        if value.shape == shape:
            return value
        if value.ndim == 3 and value.shape[1:] == shape:
            return value[-1]
        raise ValueError(f"{key} must have shape {shape} or Tx{shape}, got {value.shape}")


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

        model_kwargs = dict(model_kwargs)
        model_kwargs.pop("include_future_object_point", None)
        self.model = GraphFlowModel(**model_kwargs).to(device)
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

        scene_condition = input_data.get("scene_condition")
        if scene_condition is None:
            text_conditions = input_data.get("scene_condition_texts")
            if not isinstance(text_conditions, Sequence) or isinstance(text_conditions, str | bytes):
                raise TypeError("scene_condition_texts must be [action_type, action_degree]")
            if len(text_conditions) != 2:
                raise ValueError("scene_condition_texts must contain action_type and action_degree")
            scene_condition = torch.stack([
                self.embed_text(None if item is None else str(item)) for item in text_conditions
            ]).unsqueeze(0)
        else:
            scene_condition = torch.as_tensor(scene_condition, device=self.device, dtype=torch.float32)
            if scene_condition.ndim == 2:
                scene_condition = scene_condition.unsqueeze(0)

        infer_inputs = {
            "entity_points": tensor("entity_points", torch.float32),
            "entity_point_mask": tensor("entity_point_mask", torch.bool),
            "scene_condition": scene_condition,
        }
        if "gripper_closedness_history" in input_data:
            infer_inputs["gripper_closedness_history"] = tensor(
                "gripper_closedness_history", torch.float32,
            )
        outputs = self.model.sample(infer_inputs)
        output_data = {"outputs": outputs, "batch": infer_inputs}
        for transform in self.out_transforms:
            output_data = transform(output_data)
        json_outputs = self.to_json(output_data["outputs"])
        if not return_model_input:
            return json_outputs
        return json_outputs, self.to_json(infer_inputs)

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
        execute_chunk_len: int,
        preprocessor: InputPreprocessor,
        inference: InferenceModel,
        embodiment: EmbodimentAdapter,
        ckpt_path: str | Path,
    ) -> None:
        self.host = host
        self.port = port
        self.planner = planner
        if not 1 <= execute_chunk_len <= embodiment.future_horizon:
            raise ValueError(
                f"execute_chunk_len must be in [1, {embodiment.future_horizon}], "
                f"got {execute_chunk_len}"
            )
        self.execute_chunk_len = int(execute_chunk_len)
        self.preprocessor = preprocessor
        self.inference = inference
        self.embodiment = embodiment
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.sessions: dict[str, InferenceSession] = {}
        self.idle_timeout = 180.0
        self.last_message_time = 0.0

    def serve_forever(self) -> None:
        asyncio.run(self.serve())

    async def serve(self) -> None:
        cs.print(f"[green]GraphVLA inference server listening on ws://{self.host}:{self.port}[/green]")
        cs.print("WebSocket messages must be flat observation requests; replies are model fields plus action.")
        self.last_message_time = asyncio.get_running_loop().time()
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        ):
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
                if request.get("type") == "server_info":
                    response = {"ckpt_path": self.ckpt_path}
                else:
                    response = self.infer_from_observation(request)
            except Exception as exc:
                response = {"error": str(exc)}
            await websocket.send(json.dumps(response))

    def infer_from_observation(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_request(request)
        session = self._session_for(request)
        subtaskstructure = self.planner.plan(request, session)
        model_input = self.preprocessor.build(request, session, subtaskstructure)
        current_features = session.feature_history[-1]

        subtask_switched = False
        if session.release_pending:
            subtask_switched = self.planner.advance_after_release(session)
            if subtask_switched:
                self.preprocessor._append_feature_history(session, current_features)
                subtaskstructure = self.planner.plan(request, session)
                model_input = self.preprocessor._build_model_input(
                    session,
                    self.preprocessor._feature_window(session),
                    subtaskstructure,
                )

        return_model_input = bool(request.get("return_model_input", False))
        release_requested = False
        if session.task_complete:
            outputs = {}
            captured_model_input = None
        else:
            outputs, captured_model_input = self._infer_model(
                model_input,
                return_model_input=return_model_input,
            )
            release_requested = self.planner.update_after_inference(outputs, session)

        if release_requested or session.task_complete:
            executed_actions = self.embodiment.release_actions(
                session,
                self.execute_chunk_len,
                request,
            )
        else:
            actions = self.embodiment.to_action(outputs, request, session)
            executed_actions = actions[:self.execute_chunk_len]
            cs.print(
                f"step={session.frame_index} "
                f"{self.planner._contact_score_text(outputs)}",
                markup=False,
            )
        gripper_actions = [float(action[-1]) for action in executed_actions]
        gripper_text = ", ".join(f"{g:.3f}" for g in gripper_actions)
        cs.print(
            f"step={session.frame_index} gripper_action[{len(gripper_actions)}]=[{gripper_text}]",
            markup=False,
        )
        response_initial_points, response_initial_point_object_ids, response_initial_point_active = (
            self._initial_points_response(session)
        )
        response_tracking_points, response_tracking_object_id, response_tracking_point_active = (
            self._tracking_response(session, current_features)
        )
        response = {
            **outputs,
            "entity_points": self.inference.to_json(model_input["entity_points"]),
            "entity_point_mask": self.inference.to_json(model_input["entity_point_mask"]),
            "initial_points": self.inference.to_json(response_initial_points),
            "initial_point_object_ids": self.inference.to_json(response_initial_point_object_ids),
            "initial_point_active": self.inference.to_json(response_initial_point_active),
            "tracking_point": self.inference.to_json(response_tracking_points),
            "tracking_object_id": self.inference.to_json(response_tracking_object_id),
            "tracking_point_active": self.inference.to_json(response_tracking_point_active),
            "subtask": session.current_subtask,
            "subtask_index": session.subtask_index,
            "subtask_switched": subtask_switched,
            "episode_done": session.task_complete,
            "action": executed_actions,
        }
        if captured_model_input is not None:
            response["model_input"] = captured_model_input
        return response

    def _infer_model(
        self,
        model_input: Mapping[str, Any],
        *,
        return_model_input: bool,
    ) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
        inference_result = self.inference.infer(
            model_input,
            return_model_input=return_model_input,
        )
        if not return_model_input:
            return inference_result, None

        outputs, captured_model_input = inference_result
        captured_model_input["scene_condition_texts"] = model_input.get("scene_condition_texts")
        return outputs, captured_model_input

    @staticmethod
    def _initial_points_response(
        session: InferenceSession,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if session.initial_points is None:
            return None, None, None
        points_by_object = np.asarray(session.initial_points, dtype=np.float32)
        point_chunks = []
        id_chunks = []
        active_chunks = []
        active_ids = {
            object_index: local_id
            for local_id, object_index in enumerate(session.active_object_indices, start=1)
        }
        for object_index, points in enumerate(points_by_object):
            local_id = active_ids.get(object_index, object_index + 1)
            point_chunks.append(points)
            id_chunks.append(np.full(len(points), local_id, dtype=np.int64))
            active_chunks.append(np.full(len(points), object_index in active_ids, dtype=bool))
        if not point_chunks:
            return None, None, None
        return (
            np.concatenate(point_chunks, axis=0),
            np.concatenate(id_chunks, axis=0),
            np.concatenate(active_chunks, axis=0),
        )

    def _tracking_response(
        self,
        session: InferenceSession,
        current_features: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        point_chunks = []
        id_chunks = []
        active_chunks = []

        actor_xyz = np.asarray(current_features.get("gripper_points_xyz", []), dtype=np.float32)
        if actor_xyz.shape == (GRIPPER_NUM_POINTS, 3):
            actor_xyz = actor_xyz[list(self.embodiment.actor_point_indices)]
        intrinsic = np.asarray(current_features.get("intrinsic", []), dtype=np.float32)
        if actor_xyz.shape == (self.embodiment.actor_num_points, 3) and intrinsic.shape == (3, 3):
            valid = np.isfinite(actor_xyz).all(axis=1) & (actor_xyz[:, 2] > 1e-6)
            actor_points = np.full((self.embodiment.actor_num_points, 3), np.nan, dtype=np.float32)
            actor_points[:, 2] = 0.0
            actor_points[valid, 0] = actor_xyz[valid, 0] / actor_xyz[valid, 2] * intrinsic[0, 0] + intrinsic[0, 2]
            actor_points[valid, 1] = actor_xyz[valid, 1] / actor_xyz[valid, 2] * intrinsic[1, 1] + intrinsic[1, 2]
            actor_points[valid, 2] = 1.0
            point_chunks.append(actor_points)
            id_chunks.append(np.zeros(self.embodiment.actor_num_points, dtype=np.int64))
            active_chunks.append(np.ones(self.embodiment.actor_num_points, dtype=bool))

        if session.tracked_points is not None:
            points_by_object = np.asarray(session.tracked_points, dtype=np.float32)
            active_ids = {
                object_index: local_id
                for local_id, object_index in enumerate(session.active_object_indices, start=1)
            }
            for object_index, points in enumerate(points_by_object):
                local_id = active_ids.get(object_index, object_index + 1)
                point_chunks.append(points)
                id_chunks.append(np.full(len(points), local_id, dtype=np.int64))
                active_chunks.append(np.full(len(points), object_index in active_ids, dtype=bool))
        if not point_chunks:
            return None, None, None
        return (
            np.concatenate(point_chunks, axis=0),
            np.concatenate(id_chunks, axis=0),
            np.concatenate(active_chunks, axis=0),
        )

    def _validate_request(self, request: Mapping[str, Any]) -> None:
        missing = [key for key in REQUIRED_REQUEST_FIELDS if key not in request]
        if missing:
            raise KeyError(f"Missing required request fields: {missing}")
        if request["benchmark"] != "libero":
            raise ValueError(f"Unsupported benchmark: {request['benchmark']!r}")

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
        "node_segmenter": default_device,
        "point_tracker": default_device,
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
    parser.add_argument(
        "--locator",
        choices=("locateanything", "robobrain"),
        default=Args.locator,
        help="Node locator used to generate point prompts.",
    )
    parser.add_argument(
        "--locator-scale",
        type=float,
        default=Args.locator_scale,
        help="Scale factor applied to the image before node locator inference.",
    )
    parser.add_argument(
        "--locator-mode",
        choices=("point", "box"),
        default=Args.locator_mode,
        help="Prompt type generated by the node locator.",
    )
    parser.add_argument(
        "--segmenter",
        choices=("sam2", "sam3"),
        default=Args.segmenter,
        help="Node segmenter used to initialize object masks.",
    )
    parser.add_argument(
        "--sam-only",
        action="store_true",
        help="Track masks with SAM and resample object points every frame without PointTracker.",
    )
    parser.add_argument("--execute-chunk-len", type=int, default=Args.execute_chunk_len)
    parser.add_argument(
        "--release-lift-height",
        type=float,
        default=Args.release_lift_height,
        help="Total world-Z lift in meters while releasing in absolute-action mode.",
    )
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
    parser.add_argument("--seed", type=int, default=Args.seed, help="Random seed for server inference.")
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
        help='JSON device map for server modules, e.g. {"inference":"cuda:0","node_segmenter":"cuda:1"}.',
    )
    namespace = parser.parse_args()
    if namespace.locator_scale <= 0:
        parser.error("--locator-scale must be positive")
    if namespace.release_lift_height < 0:
        parser.error("--release-lift-height must be non-negative")
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.example == "libero":
        from examples.libero.embodiment.robot import GeomRobot

        data_config, model_config, _ = TrainingCheckpoint.load_config_snapshots(args.ckpt_path)
        model_kwargs = model_config.to_kwargs()
        data_kwargs = data_config.to_kwargs()
        history_horizon = int(model_config.history_horizon)
        future_horizon = int(model_config.future_horizon)
        actor_point_indices = validate_actor_point_indices(model_config.actor_point_indices)
    else:
        raise ValueError(f"Unsupported example: {args.example}")
        
    server = InferenceServer(
        host=args.host,
        port=args.port,
        execute_chunk_len=args.execute_chunk_len,
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
            actor_point_indices=actor_point_indices,
            robot_cls=GeomRobot,
            dataset_dir=data_kwargs["dataset_dir"],
            locator=args.locator,
            locator_mode=args.locator_mode,
            locator_scale=args.locator_scale,
            segmenter=args.segmenter,
            sam_only=args.sam_only,
            devices=args.devices,
        ),
        inference = InferenceModel(
            model_kwargs=model_kwargs,
            data_kwargs=data_kwargs,
            ckpt_path=args.ckpt_path,
            device=torch.device(args.devices["inference"]),
            bge_path=args.bge_path,
        ),
        embodiment=EmbodimentAdapter(
            future_horizon=future_horizon,
            actor_point_indices=actor_point_indices,
            action_delta=bool(getattr(model_config, "action_delta", True)),
            flow_mode=str(getattr(model_config, "flow_mode", "joint")),
            robot_cls=GeomRobot,
            release_lift_height=args.release_lift_height,
        ),
        ckpt_path=args.ckpt_path,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
