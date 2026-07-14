#!/usr/bin/env python3
"""LIBERO evaluation client skeleton for GraphVLA.

This file owns benchmark rollout and metric collection. Model-specific inference
is isolated in InferenceClient; for now server outputs are cached and the
executable action remains a dummy action.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import json
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Any

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
LIBERO_PACKAGE_ROOT = LIBERO_ROOT / "libero"
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/graphvla_libero_config")
libero_config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
libero_config_dir.mkdir(parents=True, exist_ok=True)
libero_config_file = libero_config_dir / "config.yaml"
if not libero_config_file.exists():
    benchmark_root = LIBERO_PACKAGE_ROOT / "libero"
    libero_config_file.write_text(
        "benchmark_root: {0}\n"
        "bddl_files: {0}/bddl_files\n"
        "init_states: {0}/init_files\n"
        "datasets: {1}/datasets\n"
        "assets: {0}/assets\n".format(benchmark_root, LIBERO_PACKAGE_ROOT),
        encoding="utf-8",
    )

from libero.libero import benchmark
import imageio
import numpy as np
from PIL import Image
from rich.console import Console
import websockets

from examples.libero.config.data_config import LIBERO_DATA_CONFIG
from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
from examples.libero.embodiment.robot import GeomFrankaPanda
from src.common.geom_utils import sample_points_from_mask
from src.common.schema import NodeRole, taskstructure_to_json
from src.module.depth_predictor import DepthPredictorSTream3R
from src.module.node_locator import NodeLocatorRobo
from src.module.node_segmenter import NodeSegmenter
from src.module.point_tracker import PointTracker
from src.module.scale_estimator import RawDepthShiftCalibrator
from src.module.task_analyzer import TaskAnalyzer

cs = Console()

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
LIBERO_CAMERA_DATASET_DIR = Path("/data0/luokang/dataset/luokang/lerobot/libero/libero_31_no_noops_1.0.0_lerobot_10hz")
LIBERO_CAMERA_TASK_INDEX = 0
LIBERO_CAMERA_NAME = "agentview"


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10092
    task_suite_name: str = "libero_10"
    tasks: list[int] | None = None
    api_key: str | None = None
    complete_score_threshold: float = 0.5
    complete_streak_threshold: int = 5
    num_steps_wait: int = 10
    num_trials_per_task: int = 5
    max_steps: int | None = None
    video_out_path: str = "examples/libero/eval/output/videos"
    result_out_path: str = "examples/libero/eval/output/result.json"
    seed: int = 42
    save_video: bool = True


class TopLevelPlanner:
    """Subtask planner driven by model is_complete outputs."""

    def __init__(self, *, score_threshold: float, streak_threshold: int) -> None:
        self.score_threshold = score_threshold
        self.streak_threshold = streak_threshold
        self.subtask_index = 0
        self.complete_streak = 0
        self.task_complete = False

    def reset(self) -> None:
        self.subtask_index = 0
        self.complete_streak = 0
        self.task_complete = False

    def current_subtask(self, taskstructure: Any) -> Any:
        return taskstructure.subtask_list[self.subtask_index]

    def update(self, outputs: dict[str, Any], request_data: dict[str, Any], taskstructure: Any) -> bool:
        if "is_complete" not in outputs:
            self.complete_streak = 0
            return False
        scores = np.asarray(outputs["is_complete"], dtype=np.float32)[0].reshape(-1)
        frame_ids = np.asarray(request_data["frame_query_frame_id"], dtype=np.int64)[0].reshape(-1)
        future_scores = scores[frame_ids >= 0]
        complete_now = bool(future_scores.size and np.nanmax(future_scores) >= self.score_threshold)
        self.complete_streak = self.complete_streak + 1 if complete_now else 0
        if self.complete_streak < self.streak_threshold:
            return False

        self.complete_streak = 0
        if self.subtask_index + 1 < len(taskstructure.subtask_list):
            self.subtask_index += 1
            cs.print(f"[cyan]switch to subtask {self.subtask_index}: {self.current_subtask(taskstructure).subtask}[/cyan]")
        else:
            self.task_complete = True
            cs.print("[cyan]all subtasks complete according to is_complete[/cyan]")
        return True


class ModelInputBuilder:
    """Build GraphVLA websocket request payloads from online perception state."""

    def __init__(self, *, model_config: Any, history_horizon: int = 15, future_horizon: int = 16) -> None:
        self.model_config = model_config
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.history: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        self.history = []

    def append(self, *, tracks: np.ndarray, depth: np.ndarray, gripper_uvd: np.ndarray) -> None:
        self.history.append({
            "tracks": np.asarray(tracks, dtype=np.float32),
            "depth": np.asarray(depth, dtype=np.float32),
            "gripper_uvd": np.asarray(gripper_uvd, dtype=np.float32),
        })
        max_len = self.history_horizon + 1
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]

    def build(self, subtask: Any) -> dict[str, Any]:
        frames = self._history_window()
        height, width = frames[-1]["depth"].shape
        object_points = np.stack([self._object_feats(item["tracks"], item["depth"], height, width) for item in frames], axis=0)
        actor_points = np.stack([self._actor_feats(item["gripper_uvd"], item["depth"], height, width) for item in frames], axis=0)[:, None]
        frame_offsets = np.arange(-self.history_horizon, self.future_horizon + 1, dtype=np.int64)
        object_id, point_id, frame_id = self._build_query_ids(
            frame_offsets,
            object_points.shape[1],
            object_points.shape[2],
            actor_points.shape[2],
        )
        object_roles = ["patient", "target"]
        object_condition_texts = [
            {"role": role, "action_type": subtask.action_type.value[0], "action_degree": subtask.action_degree}
            for role in object_roles
        ]
        actor_condition_text = {
            "role": "actor",
            "action_type": subtask.action_type.value[0],
            "action_degree": subtask.action_degree,
        }
        return {
            "point_feats": object_points[None].tolist(),
            "actor_feats": actor_points[None].tolist(),
            "object_condition_texts": object_condition_texts,
            "actor_condition_text": actor_condition_text,
            "object_id": object_id[None].tolist(),
            "point_id": point_id[None].tolist(),
            "frame_id": frame_id[None].tolist(),
            "frame_query_frame_id": frame_offsets[None].tolist(),
            "head_names": "point",
            "frame_head_names": "is_complete",
        }

    def _history_window(self) -> list[dict[str, np.ndarray]]:
        if not self.history:
            raise RuntimeError("history is empty")
        target_len = self.history_horizon + 1
        pad_count = max(0, target_len - len(self.history))
        return [self.history[0]] * pad_count + self.history[-target_len:]

    def _object_feats(self, tracks: np.ndarray, depth: np.ndarray, height: int, width: int) -> np.ndarray:
        points = np.zeros((2, self.model_config.num_points, 6), dtype=np.float32)
        count = min(tracks.shape[0], 2)
        if count == 0:
            return points
        uv = tracks[:count, :, :2]
        vis = tracks[:count, :, 2:3]
        sampled_depth, in_bounds = self._sample_depth(depth, uv, height, width)
        points[:count, :, :2] = self._normalize_uv(uv, height, width)
        points[:count, :, 2:3] = sampled_depth
        points[:count, :, 3:4] = vis * in_bounds
        return points

    def _actor_feats(self, gripper_uvd: np.ndarray, depth: np.ndarray, height: int, width: int) -> np.ndarray:
        uv = gripper_uvd[:, :2]
        sampled_depth, in_bounds = self._sample_depth(depth, uv, height, width)
        points = np.zeros((gripper_uvd.shape[0], 6), dtype=np.float32)
        points[:, :2] = self._normalize_uv(uv, height, width)
        points[:, 2:3] = sampled_depth
        points[:, 3:4] = in_bounds
        points[:, 4:5] = gripper_uvd[:, 2:3]
        points[:, 5:6] = 1.0
        return points

    @staticmethod
    def _build_query_ids(frame_offsets: np.ndarray, object_count: int, object_points: int, actor_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class ModelOutputParser:
    """Convert GraphVLA model outputs into executable LIBERO action chunks."""

    def __init__(self, *, robot: GeomFrankaPanda, intrinsic: np.ndarray, extrinsic: np.ndarray, future_horizon: int) -> None:
        self.robot = robot
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.future_horizon = future_horizon

    def to_action_chunk(self, outputs: dict[str, Any], request_data: dict[str, Any]) -> list[np.ndarray]:
        if "point" not in outputs:
            return []
        points = np.asarray(outputs["point"], dtype=np.float32)[0]
        object_id = np.asarray(request_data["object_id"], dtype=np.int64)[0]
        point_id = np.asarray(request_data["point_id"], dtype=np.int64)[0]
        frame_id = np.asarray(request_data["frame_id"], dtype=np.int64)[0]
        actions = []
        for future_frame_id in range(1, self.future_horizon + 1):
            mask = (object_id == 0) & (frame_id == future_frame_id)
            actor_points = points[mask]
            actor_point_ids = point_id[mask]
            by_id = {int(pid): actor_points[index] for index, pid in enumerate(actor_point_ids)}
            if not all(index in by_id for index in (0, 1, 2)):
                continue
            uvd_dict = {
                "root_uvd": by_id[0][:3],
                "left_uvd": by_id[1][:3],
                "right_uvd": by_id[2][:3],
            }
            if not all(np.isfinite(value).all() and float(value[2]) > 1e-6 for value in uvd_dict.values()):
                continue
            action = self.robot.project_uvd_to_gripper(
                uvd_dict,
                intrinsic=self.intrinsic,
                extrinsic=self.extrinsic,
            )
            actions.append(np.asarray(action, dtype=np.float32))
        return actions


class InferenceClient:
    """Thin online client that coordinates perception, planning, websocket inference, and actions."""

    def __init__(self, *, host: str, port: int, api_key: str | None, complete_score_threshold: float, complete_streak_threshold: int) -> None:
        self.host = host
        self.port = port
        self.api_key = api_key
        self.model_config = LIBERO_MODEL_CONFIG
        self.data_config = LIBERO_DATA_CONFIG
        self.taskstructure_cache = {}

        self.task_analyzer = None
        self.node_locator = None
        self.sam3_model = None
        self.object_segmenter = None
        self.table_segmenter = None
        self.point_tracker = None
        self.depth_predictor = None
        self.depth_calibrator = None
        self.robot = GeomFrankaPanda()
        self.intrinsic, self.extrinsic = self._load_camera()

        self.planner = TopLevelPlanner(
            score_threshold=complete_score_threshold,
            streak_threshold=complete_streak_threshold,
        )
        self.input_builder = ModelInputBuilder(model_config=self.model_config)
        self.output_parser = ModelOutputParser(
            robot=self.robot,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic,
            future_horizon=self.input_builder.future_horizon,
        )
        self.server_output = None
        self.action_plan = []
        self.frame_index = 0
        self.object_nodes = []
        self.object_points = None
        self.object_masks = None
        self.table_points = None
        self.table_mask = None
        self.tracked_points = None
        self.depth = None
        self.gripper_uvd = None

    def reset_episode(self) -> None:
        self.server_output = None
        self.action_plan = []
        self.frame_index = 0
        self.object_nodes = []
        self.object_points = None
        self.object_masks = None
        self.table_points = None
        self.table_mask = None
        self.tracked_points = None
        self.depth = None
        self.gripper_uvd = None
        self.object_segmenter = None
        self.table_segmenter = None
        self.point_tracker = None
        self.depth_predictor = None
        self.depth_calibrator = None
        self.planner.reset()
        self.input_builder.reset()

    def infer(self, observation: dict[str, Any], task_description: str) -> np.ndarray:
        frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
        state = np.asarray(observation["state"], dtype=np.float64)
        taskstructure = self._taskstructure(task_description)
        if self.planner.task_complete:
            return LIBERO_DUMMY_ACTION.copy()
        if self.frame_index == 0:
            self._initialize_episode(frame, taskstructure)
        else:
            self._update_episode(frame)
        self.depth = self._predict_depth(frame)
        self.gripper_uvd = self._state_to_gripper_uvd(state, frame.shape[:2])
        self.input_builder.append(
            tracks=self.tracked_points,
            depth=self.depth,
            gripper_uvd=self.gripper_uvd,
        )
        if not self.action_plan:
            request_data = self.input_builder.build(self.planner.current_subtask(taskstructure))
            self.server_output = self._call_model(request_data)
            switched = self.planner.update(self.server_output, request_data, taskstructure)
            if not switched and not self.planner.task_complete:
                self.action_plan = self.output_parser.to_action_chunk(self.server_output, request_data)
        self.frame_index += 1
        if not self.action_plan:
            return LIBERO_DUMMY_ACTION.copy()
        return self.action_plan.pop(0)

    def _initialize_episode(self, frame: np.ndarray, taskstructure: Any) -> None:
        debug_frame_path = PROJECT_ROOT / "examples/libero/test/first_eval_frame.png"
        debug_frame_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(debug_frame_path, frame)
        cs.print(f"saved first eval frame to: {debug_frame_path}")

        self.object_nodes = self._object_nodes(taskstructure)
        if self.sam3_model is None:
            self.sam3_model = NodeSegmenter().model
        self.object_segmenter = NodeSegmenter(model=self.sam3_model)
        self.table_segmenter = NodeSegmenter(model=self.sam3_model)
        self.point_tracker = PointTracker()
        self.depth_predictor = DepthPredictorSTream3R()

        if self.object_nodes:
            try:
                point_prompts = [self._locate_node_points(frame, node.name) for node in self.object_nodes]
                self._save_node_locator_debug(frame, self.object_nodes, point_prompts)
            finally:
                self._release_node_locator()
            self.object_masks = self.object_segmenter.predict(frame, points=point_prompts, anchor_frame=True)
            self.object_points = np.stack(
                [
                    sample_points_from_mask(mask, num_points=self.model_config.num_points)
                    for mask in self.object_masks
                ],
                axis=0,
            ).astype(np.float32)
            track_result = self.point_tracker.track(frame, points=self.object_points, anchor_frame=True)
            self.tracked_points = self._pack_tracks(track_result)
        else:
            self.object_masks = []
            self.object_points = np.zeros((0, self.model_config.num_points, 2), dtype=np.float32)
            self.tracked_points = np.zeros((0, self.model_config.num_points, 3), dtype=np.float32)

        table_prompt_masks = self.table_segmenter.segment_prompt_video([frame], prompt="table")[0]
        table_mask = self._fill_mask_holes(self._largest_mask(table_prompt_masks, frame.shape[:2]))
        self.table_points = sample_points_from_mask(table_mask, num_points=self.model_config.num_points).astype(np.float32)
        tracked_table_masks = self.table_segmenter.predict(frame, points=[self.table_points], anchor_frame=True)
        self.table_mask = self._fill_mask_holes(self._largest_mask(tracked_table_masks, frame.shape[:2]))

    def _update_episode(self, frame: np.ndarray) -> None:
        if self.object_segmenter is not None and self.object_nodes:
            self.object_masks = self.object_segmenter.predict(frame, anchor_frame=False)
        if self.table_segmenter is not None:
            table_masks = self.table_segmenter.predict(frame, anchor_frame=False)
            self.table_mask = self._fill_mask_holes(self._largest_mask(table_masks, frame.shape[:2]))
        if self.point_tracker is not None and self.object_nodes:
            self.tracked_points = self._pack_tracks(self.point_tracker.track(frame, anchor_frame=False))

    def _predict_depth(self, frame: np.ndarray) -> np.ndarray:
        if self.depth_predictor is None:
            self.depth_predictor = DepthPredictorSTream3R()
        output = self.depth_predictor.predict(frame, anchor_frame=(self.frame_index == 0))
        depth = output[0] if isinstance(output, tuple) else output
        depth = np.asarray(depth, dtype=np.float32)
        if self.table_mask is None:
            return depth
        if self.depth_calibrator is None:
            self.depth_calibrator = RawDepthShiftCalibrator(depth, self.table_mask)
            return depth
        try:
            return self.depth_calibrator.calibrate(depth, self.table_mask)
        except ValueError:
            return depth

    def _call_model(self, request_data: dict[str, Any]) -> dict[str, Any]:
        uri = f"ws://{self.host}:{self.port}"
        response = asyncio.run(self._websocket_json(uri, request_data))
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["outputs"]

    @staticmethod
    async def _websocket_json(uri: str, data: dict[str, Any]) -> dict[str, Any]:
        async with websockets.connect(uri, max_size=None, proxy=None) as websocket:
            await websocket.send(json.dumps(data))
            message = await websocket.recv()
        return json.loads(message)

    def _taskstructure(self, task_description: str) -> Any:
        if task_description not in self.taskstructure_cache:
            if self.task_analyzer is None:
                if not self.api_key:
                    raise RuntimeError("api_key is required for TaskAnalyzer")
                self.task_analyzer = TaskAnalyzer(api_key=self.api_key)
            taskstructure = self.task_analyzer.analyze_task(task_description)
            self.taskstructure_cache[task_description] = taskstructure
            cs.print("[cyan]taskstructure:[/cyan]")
            cs.print(json.dumps(taskstructure_to_json(taskstructure), indent=2, ensure_ascii=False))
        return self.taskstructure_cache[task_description]

    def _locate_node_points(self, frame: np.ndarray, node_name: str) -> list[list[float]]:
        if self.node_locator is None:
            self.node_locator = NodeLocatorRobo()
        result = self.node_locator.inference(text=node_name, image=Image.fromarray(frame))
        points = result.get("points") or []
        if not points:
            raise RuntimeError(f"NodeLocatorRobo found no points for node={node_name!r}")
        return self._locator_points_to_pixels(points, frame.shape[:2])

    def _release_node_locator(self) -> None:
        if self.node_locator is None:
            return
        del self.node_locator
        self.node_locator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_node_locator_debug(self, frame: np.ndarray, nodes: list[Any], point_prompts: list[list[list[float]]]) -> None:
        debug = np.ascontiguousarray(frame.copy())
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 128, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]
        for node_idx, (node, points) in enumerate(zip(nodes, point_prompts)):
            color = colors[node_idx % len(colors)]
            for point_idx, point in enumerate(points):
                x, y = int(round(point[0])), int(round(point[1]))
                cv2.circle(debug, (x, y), 3, color, thickness=-1)
                cv2.putText(
                    debug,
                    str(point_idx),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            label = f"{node_idx}: {node.name} ({len(points)} pts)"
            cv2.putText(
                debug,
                label,
                (6, 18 + 18 * node_idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            cs.print(f"node locator: {node.name} -> {len(points)} points")

        debug_path = PROJECT_ROOT / "examples/libero/test/node_locator_points.png"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(debug_path, debug)
        cs.print(f"saved node locator points to: {debug_path}")

    def _state_to_gripper_uvd(self, state: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        gripper_state = abs(float(state[6])) + abs(float(state[7])) if state.size >= 8 else float(state[6])
        output = self.robot.project_gripper_to_uvd(
            tcp_state=state[:6],
            gripper_state=gripper_state,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic,
            image_size=image_size,
            mode="3P",
        )
        return np.stack([np.asarray(output[key], dtype=np.float32) for key in ("root_uvd", "left_uvd", "right_uvd")])

    def _load_camera(self) -> tuple[np.ndarray, np.ndarray]:
        camera_file = LIBERO_CAMERA_DATASET_DIR / "meta" / "cameras.json"
        rows = json.loads(camera_file.read_text(encoding="utf-8"))
        record = next(row for row in rows if int(row["task_index"]) == LIBERO_CAMERA_TASK_INDEX)
        camera = record["cameras"][LIBERO_CAMERA_NAME]
        return np.asarray(camera["intrinsic"], dtype=np.float64), np.asarray(camera["extrinsic"], dtype=np.float64)

    @staticmethod
    def _object_nodes(taskstructure: Any) -> list[Any]:
        nodes = []
        for subtask in taskstructure.subtask_list:
            for node in subtask.node_list or []:
                if bool(node.need_object) and node.role != NodeRole.ACTOR:
                    nodes.append(node)
        return nodes

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


def eval_libero(args: Args) -> None:
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

    video_dir = pathlib.Path(args.video_out_path)
    result_path = pathlib.Path(args.result_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        complete_score_threshold=args.complete_score_threshold,
        complete_streak_threshold=args.complete_streak_threshold,
    )
    total_episodes = 0
    total_successes = 0
    total_completion = 0.0
    task_results: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        total_goals = len(env.env.parsed_problem["goal_state"])
        task_episodes = 0
        task_successes = 0
        task_completion = 0.0

        for episode_idx in range(args.num_trials_per_task):
            logging.info("Task %s episode %s: %s", task_id, episode_idx, task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            client.reset_episode()
            replay_images = []
            done = False

            for step in range(max_steps + args.num_steps_wait):
                if step < args.num_steps_wait:
                    action = LIBERO_DUMMY_ACTION
                else:
                    action = client.infer(_prepare_observation(obs), task_description)

                replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, :]))
                obs, _, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
                if done:
                    break

            task_episodes += 1
            total_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1

            goal_state = env.env.parsed_problem["goal_state"]
            completed_goals = sum(bool(env._eval_predicate(state)) for state in goal_state)
            total_goals = len(goal_state)
            completion = float(completed_goals) / float(total_goals) if total_goals else 0.0
            task_completion += completion
            total_completion += completion

            if args.save_video:
                suffix = "success" if done else "failure"
                video_name = f"task_{task_id:03d}_ep_{episode_idx:03d}_{suffix}.mp4"
                imageio.mimwrite(video_dir / video_name, replay_images, fps=10)

            cs.print(
                f"task={task_id} episode={episode_idx} success={done} "
                f"completion={completed_goals}/{total_goals}"
            )

        task_result = {
            "task_id": task_id,
            "task_desc": task_description,
            "total_goals": total_goals,
            "success_rate": float(task_successes) / float(task_episodes),
            "completion_rate": float(task_completion) / float(task_episodes),
            "num_episodes": task_episodes,
        }
        task_results.append(task_result)

    result = {
        "task_suite": args.task_suite_name,
        "success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
        "completion_rate": float(total_completion) / float(total_episodes) if total_episodes else 0.0,
        "total_episodes": total_episodes,
        "tasks": task_results,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cs.print(f"saved results to: {result_path}")


def _get_libero_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _prepare_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "agentview_image": np.ascontiguousarray(obs["agentview_image"][::-1, :]),
        "eye_in_hand_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, :]),
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }


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
    parser = argparse.ArgumentParser(description="Evaluate a GraphVLA policy skeleton on LIBERO.")
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--task-suite-name", default=Args.task_suite_name)
    parser.add_argument("--tasks", type=_parse_tasks, default=Args.tasks)
    parser.add_argument("--api-key", default=Args.api_key)
    parser.add_argument("--complete-score-threshold", type=float, default=Args.complete_score_threshold)
    parser.add_argument("--complete-streak-threshold", type=int, default=Args.complete_streak_threshold)
    parser.add_argument("--num-steps-wait", type=int, default=Args.num_steps_wait)
    parser.add_argument("--num-trials-per-task", type=int, default=Args.num_trials_per_task)
    parser.add_argument("--max-steps", type=int, default=Args.max_steps)
    parser.add_argument("--video-out-path", default=Args.video_out_path)
    parser.add_argument("--result-out-path", default=Args.result_out_path)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--no-save-video", action="store_true")
    ns = parser.parse_args()
    return Args(
        host=ns.host,
        port=ns.port,
        task_suite_name=ns.task_suite_name,
        tasks=ns.tasks,
        api_key=ns.api_key,
        complete_score_threshold=ns.complete_score_threshold,
        complete_streak_threshold=ns.complete_streak_threshold,
        num_steps_wait=ns.num_steps_wait,
        num_trials_per_task=ns.num_trials_per_task,
        max_steps=ns.max_steps,
        video_out_path=ns.video_out_path,
        result_out_path=ns.result_out_path,
        seed=ns.seed,
        save_video=not ns.no_save_video,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    eval_libero(parse_args())


if __name__ == "__main__":
    main()
