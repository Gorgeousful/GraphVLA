from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rich.console import Console
from tqdm import tqdm

from src.common.geom_utils import sample_points_from_mask
from src.common.schema import LIBERO_GRIPPER_MAX_WIDTH, NodeRole, TaskStructure, json_to_taskstructure, taskstructure_to_json
from src.module.binary_segmenter import BinarySegmenter
from src.module.depth_predictor import DepthPredictorSTream3R
from src.module.node_locator import NodeLocatorRobo
from src.module.node_segmenter import NodeSegmenter
from src.module.point_tracker import PointTracker
from src.module.scale_estimator import RawDepthShiftCalibrator
from src.module.task_analyzer import TaskAnalyzer

cs = Console()


@dataclass
class PipelineConfig:
    dataset_dir: str | os.PathLike
    episode_selector: Mapping[str | int, Sequence[int | str]]
    dataset_type: str = "libero"
    task_analyzer_api_key: str | None = None
    points_per_node: int = 128
    max_nodes: int = 10
    erode_pixel: int = 2
    overwrite: Mapping[str, bool] | None = None
    debug: bool = False
    debug_dir: str | os.PathLike = "/data0/luokang/research/GraphVLA/__tmp__/pipeline"


class OfflinePipeline:
    """Precompute task structures, node tracks, and relative depth for episodes."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        self.meta_dir = self.dataset_dir / "meta"
        self.taskstructures_jsonl_path = self.meta_dir / "taskstructures.jsonl"
        self.tasks_jsonl_path = self.meta_dir / "tasks.jsonl"
        self.debug_dir = Path(config.debug_dir)
        self.node_locator_vis_dir = self.debug_dir / "node_locator"
        self.point_tracker_vis_dir = self.debug_dir / "point_tracker"
        self.depth_vis_dir = self.debug_dir / "depth"
        self.far_background_vis_dir = self.debug_dir / "far_background"
        self.gripper_uvd_vis_dir = self.debug_dir / "gripper_uvd"
        self.pointcloud_dir = self.debug_dir / "pointcloud"
        overwrite_defaults = {
            "taskstructure": False,
            "node_points_track": True,
            "node_points_mask": True,
            "depths_rel": True,
            "far_background_mask": True,
            "is_complete": True,
            "gripper_uvd": True,
            "gripper_openness": True,
            "subtask_id": True,
        }
        unknown_overwrite = set(config.overwrite or {}) - set(overwrite_defaults)
        if unknown_overwrite:
            raise KeyError(f"Unknown overwrite keys: {sorted(unknown_overwrite)}")
        self.overwrite = {**overwrite_defaults, **(config.overwrite or {})}
        self.info = self._load_json(self.meta_dir / "info.json")
        self.data_path_template = self.info["data_path"]
        self.video_path_template = self.info["video_path"]
        self.chunk_size = int(self.info.get("chunks_size", 1000))
        self.video_key = self._infer_video_key()
        self.task_index_to_desc = self._load_task_index_to_desc()

        self.task_analyzer = None
        self.node_locator = None
        self.node_segmenter = None
        self.point_tracker = None
        self.depth_predictor = None
        self.binary_segmenter = None
        self.complete_intervals = None
        self.libero_cameras = None
        self.libero_subtask_id_map = None
        self.gripper_geometry = None

    def run(self):
        if not self.config.debug:
            self._ensure_output_features()
        else:
            cs.print("[yellow]debug dry-run: skip meta/info.json update[/yellow]")
        task_indices = [int(task_index) for task_index in self.config.episode_selector]
        taskstructures = self.build_taskstructures(task_indices)
        episode_indices = self.resolve_episode_indices(self.config.episode_selector)
        self._print_save_plan()
        self.process_episodes(episode_indices, taskstructures)

    def build_taskstructures(self, task_indices: Sequence[int]) -> dict[int, TaskStructure]:
        cached = self._load_taskstructures_jsonl()
        cached_by_task = {taskstructure.task: taskstructure for taskstructure in cached.values()}
        taskstructures = {}
        for task_index in task_indices:
            if task_index not in self.task_index_to_desc:
                raise KeyError(f"Missing task description for task_index={task_index}")

            task_desc = self.task_index_to_desc[task_index]
            if not self.overwrite["taskstructure"]:
                if task_index in cached:
                    taskstructures[task_index] = cached[task_index]
                    continue
                if task_desc in cached_by_task:
                    taskstructures[task_index] = cached_by_task[task_desc]
                    continue

            taskstructure = self._task_analyzer().analyze_task(task_desc)
            self._append_taskstructure_jsonl(task_index, taskstructure)
            taskstructures[task_index] = taskstructure
        return taskstructures

    def resolve_episode_indices(self, selector: Mapping[str | int, Sequence[int | str]]) -> list[int]:
        task_to_episodes = self._scan_task_episodes()
        selected = []
        for raw_task_index, local_indices in selector.items():
            task_index = int(raw_task_index)
            episodes = task_to_episodes.get(task_index, [])
            if not episodes:
                raise KeyError(f"No episodes found for task_index={task_index}")

            local_indices = list(local_indices)
            if local_indices == ["*"]:
                selected.extend(episodes)
                continue

            for local_index in local_indices:
                if isinstance(local_index, str):
                    local_index = int(local_index)
                if local_index < 0 or local_index >= len(episodes):
                    raise IndexError(
                        f"task_index={task_index} local_episode_index={local_index} "
                        f"out of range [0, {len(episodes)})"
                    )
                selected.append(episodes[int(local_index)])

        selected = sorted(dict.fromkeys(selected))
        cs.print(f"selected {len(selected)} episodes: {selected[:10]}{'...' if len(selected) > 10 else ''}")
        return selected

    def process_episodes(
        self,
        episode_indices: Sequence[int],
        taskstructures: Mapping[int, TaskStructure],
    ):
        task_to_episodes = {}
        for episode_index in episode_indices:
            parquet_path = self._episode_parquet_path(episode_index)
            meta = pd.read_parquet(parquet_path, columns=["task_index"])
            task_index = int(meta["task_index"].iloc[0])
            task_to_episodes.setdefault(task_index, []).append(episode_index)

        for task_index in sorted(task_to_episodes):
            episodes = task_to_episodes[task_index]
            desc = f"task {task_index}"
            for episode_index in tqdm(episodes, desc=desc, unit="episode"):
                self._process_episode(episode_index, taskstructures)

    def _process_episode(
        self,
        episode_index: int,
        taskstructures: Mapping[int, TaskStructure],
    ):
        parquet_path = self._episode_parquet_path(episode_index)
        df = pd.read_parquet(parquet_path)
        task_index = int(df["task_index"].iloc[0])
        if task_index not in taskstructures:
            raise KeyError(f"Missing taskstructure for episode={episode_index}, task_index={task_index}")

        need_node_track = (
            self.overwrite["node_points_track"]
            or "node_points_track" not in df.columns
        )
        need_node_points_mask = self.overwrite["node_points_mask"] or "node_points_mask" not in df.columns
        need_depth = self.overwrite["depths_rel"] or "depths_rel" not in df.columns
        need_far_background = (
            self.overwrite["far_background_mask"]
            or "far_background_mask" not in df.columns
        )
        need_is_complete = self.overwrite["is_complete"] or "is_complete" not in df.columns
        need_gripper_uvd = self.overwrite["gripper_uvd"] or "gripper_uvd" not in df.columns
        need_gripper_openness = (
            self.overwrite["gripper_openness"]
            or "gripper_openness" not in df.columns
        )
        need_subtask_id = (
            self.overwrite["subtask_id"]
            or "subtask_id" not in df.columns
            or (need_node_points_mask and "subtask_id" not in df.columns)
        )
        if (
            not need_node_track
            and not need_node_points_mask
            and not need_depth
            and not need_far_background
            and not need_is_complete
            and not need_gripper_uvd
            and not need_gripper_openness
            and not need_subtask_id
        ):
            return

        frames = None
        if need_node_track or need_depth or need_far_background or (need_gripper_uvd and self.config.debug):
            frames = self._read_video_frames(self._episode_video_path(episode_index))
            if len(frames) != len(df):
                raise RuntimeError(
                    f"episode={episode_index} has {len(df)} parquet rows but {len(frames)} video frames"
                )

        if need_is_complete:
            df["is_complete"] = self._build_is_complete(df, episode_index).tolist()

        if need_subtask_id:
            df["subtask_id"] = self._build_subtask_id(df, task_index).tolist()

        if need_node_points_mask:
            node_points_mask = self._build_node_points_mask(df, taskstructures[task_index])
            df["node_points_mask"] = [mask.tolist() for mask in node_points_mask]

        if need_gripper_openness:
            df["gripper_openness"] = self._build_gripper_openness(df)

        if need_gripper_uvd:
            gripper_uvd = self._build_gripper_uvd(df, task_index)
            df["gripper_uvd"] = gripper_uvd
            if self.config.debug:
                self._save_gripper_uvd_vis_video(
                    frames,
                    gripper_uvd,
                    df["gripper_openness"].to_numpy(),
                    episode_index=episode_index,
                    task_index=task_index,
                )

        if need_node_track:
            debug_node_points_masks = None
            if self.config.debug and "node_points_mask" in df.columns:
                debug_node_points_masks = np.asarray(df["node_points_mask"].tolist(), dtype=bool)
            tracks = self._build_node_points_track(
                frames,
                taskstructures[task_index],
                episode_index=episode_index,
                task_index=task_index,
                node_points_masks=debug_node_points_masks,
            )
            df["node_points_track"] = [frame_track.tolist() for frame_track in tracks]

        raw_table_masks, filled_table_masks = None, None
        if need_depth or need_far_background:
            raw_table_masks, filled_table_masks = self._build_table_masks(frames)

        if need_depth:
            depths = self._build_depths_rel(
                frames,
                table_masks=raw_table_masks,
                episode_index=episode_index,
                task_index=task_index,
            )
            df["depths_rel"] = [depth.tolist() for depth in depths]

        if need_far_background:
            far_background_masks = self._build_far_background_masks(
                frames,
                table_masks=filled_table_masks,
                episode_index=episode_index,
                task_index=task_index,
            )
            df["far_background_mask"] = [mask.tolist() for mask in far_background_masks]

        if self.config.debug:
            cs.print(f"[yellow]debug dry-run: skip writing parquet {parquet_path}[/yellow]")
            return
        self._write_episode_parquet(df, parquet_path)

    @staticmethod
    def _write_episode_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        target_types = {
            "node_points_track": pa.list_(pa.list_(pa.list_(pa.float32()))),
            "node_points_mask": pa.list_(pa.bool_()),
            "depths_rel": pa.list_(pa.list_(pa.float32())),
            "far_background_mask": pa.list_(pa.list_(pa.bool_())),
            "gripper_uvd": pa.list_(pa.list_(pa.float32())),
            "gripper_openness": pa.float32(),
            "is_complete": pa.bool_(),
            "subtask_id": pa.int64(),
        }
        for name, target_type in target_types.items():
            if name not in table.column_names:
                continue
            index = table.column_names.index(name)
            table = table.set_column(index, name, table[name].cast(target_type))
        pq.write_table(table, parquet_path)

    def _build_node_points_track(
        self,
        frames: Sequence[np.ndarray],
        taskstructure: TaskStructure,
        episode_index: int,
        task_index: int,
        node_points_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        nodes = self._object_nodes(taskstructure)
        if len(nodes) > self.config.max_nodes:
            raise ValueError(
                f"task_index={task_index} has {len(nodes)} object nodes, "
                f"which exceeds max_nodes={self.config.max_nodes}"
            )
        if not nodes:
            return np.zeros(
                (len(frames), self.config.max_nodes, self.config.points_per_node, 3),
                dtype=np.float32,
            )

        self._ensure_node_modules()
        point_prompts = [
            self._locate_node_points(frames[0], node.name)
            for node in nodes
        ]
        if self.config.debug:
            self._save_node_locator_vis(
                frames[0],
                nodes,
                point_prompts,
                episode_index=episode_index,
                task_index=task_index,
            )
        masks = self.node_segmenter.predict(frames[0], points=point_prompts, anchor_frame=True)
        sampled_points = np.stack(
            [
                sample_points_from_mask(
                    mask,
                    num_points=self.config.points_per_node,
                    erode_pixel=self.config.erode_pixel,
                )
                for mask in masks
            ],
            axis=0,
        ).astype(np.float32)

        frame_tracks = []
        tracker_results = []
        for frame_index, frame in enumerate(frames):
            result = self.point_tracker.track(
                frame,
                points=sampled_points if frame_index == 0 else None,
                anchor_frame=(frame_index == 0),
            )
            points = result["points"].astype(np.float32)
            visibles = result["visibles"].astype(np.float32)[..., None]
            frame_tracks.append(np.concatenate([points, visibles], axis=-1))
            if self.config.debug:
                tracker_results.append(result)

        if self.config.debug:
            self._save_point_tracker_vis_video(
                frames,
                tracker_results,
                episode_index=episode_index,
                task_index=task_index,
                node_points_masks=node_points_masks,
            )
        tracks = np.stack(frame_tracks, axis=0).astype(np.float32)
        padded_tracks = np.zeros(
            (len(frames), self.config.max_nodes, self.config.points_per_node, 3),
            dtype=np.float32,
        )
        padded_tracks[:, :tracks.shape[1]] = tracks
        return padded_tracks

    def _build_depths_rel(
        self,
        frames: Sequence[np.ndarray],
        table_masks: np.ndarray,
        episode_index: int,
        task_index: int,
    ) -> np.ndarray:
        self._ensure_depth_modules()
        depths = []
        intrinsic = None
        for frame_index, frame in enumerate(frames):
            output = self.depth_predictor.predict(frame, anchor_frame=(frame_index == 0))
            if frame_index == 0 and isinstance(output, tuple):
                depth, intrinsic = output
            else:
                depth = output
            depths.append(np.asarray(depth, dtype=np.float32))

        calibrator = RawDepthShiftCalibrator(depths[0], table_masks[0])
        depths_rel = [depths[0]]
        for depth, mask in zip(depths[1:], table_masks[1:]):
            try:
                depths_rel.append(calibrator.calibrate(depth, mask))
            except ValueError:
                depths_rel.append(depth)
        depths_rel = np.stack(depths_rel, axis=0).astype(np.float32)
        if self.config.debug:
            self._save_depth_vis_video(
                depths_rel,
                episode_index=episode_index,
                task_index=task_index,
            )
        if self.config.debug and intrinsic is not None:
            self._save_pointcloud_npy(
                depths_rel,
                frames,
                intrinsic,
                episode_index=episode_index,
                task_index=task_index,
            )
        return depths_rel

    def _build_far_background_masks(
        self,
        frames: Sequence[np.ndarray],
        table_masks: np.ndarray,
        episode_index: int,
        task_index: int,
    ) -> np.ndarray:
        self._ensure_far_background_modules()
        far_background_masks = []
        for frame, table_mask in zip(frames, table_masks):
            foreground_mask = self.binary_segmenter.predict(frame, mode="foreground").astype(bool)
            near_mask = foreground_mask | table_mask
            near_mask = cv2.dilate(
                near_mask.astype(np.uint8),
                np.ones((6, 6), dtype=np.uint8),
                iterations=1,
            )
            near_mask = cv2.erode(
                near_mask,
                np.ones((6, 6), dtype=np.uint8),
                iterations=1,
            ).astype(bool)
            far_background_masks.append(np.logical_not(near_mask))

        far_background_masks = np.stack(far_background_masks, axis=0).astype(bool)
        if self.config.debug:
            self._save_far_background_overlay_video(
                frames,
                far_background_masks,
                episode_index=episode_index,
                task_index=task_index,
            )
        return far_background_masks

    def _build_table_masks(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.node_segmenter is None:
            self.node_segmenter = NodeSegmenter()
        prompt_masks = self.node_segmenter.segment_prompt_video(frames, prompt="table")
        raw_table_masks = [
            self._largest_mask(masks, frame.shape[:2])
            for frame, masks in zip(frames, prompt_masks)
        ]
        filled_table_masks = [self._fill_mask_holes(mask) for mask in raw_table_masks]
        return (
            np.stack(raw_table_masks, axis=0).astype(bool),
            np.stack(filled_table_masks, axis=0).astype(bool),
        )

    def _build_is_complete(self, df: pd.DataFrame, episode_index: int) -> np.ndarray:
        dataset_type = str(self.config.dataset_type).lower()
        if dataset_type == "libero":
            return self._build_libero_is_complete(df, episode_index)
        raise KeyError(f"Unsupported dataset_type for is_complete: {self.config.dataset_type}")

    def _build_libero_is_complete(self, df: pd.DataFrame, episode_index: int) -> np.ndarray:
        intervals = self._load_libero_complete_intervals().get(int(episode_index), [])
        frame_indices = df["frame_index"].to_numpy(dtype=np.int64)
        is_complete = np.zeros(len(frame_indices), dtype=bool)
        for start_idx, end_idx in intervals:
            is_complete |= (start_idx <= frame_indices) & (frame_indices <= end_idx)
        return is_complete

    def _load_libero_complete_intervals(self) -> dict[int, list[tuple[int, int]]]:
        if self.complete_intervals is not None:
            return self.complete_intervals

        extra_file = self.meta_dir / "episodes_phase_segment.jsonl"
        if not extra_file.exists():
            raise FileNotFoundError(f"Missing is_complete source file: {extra_file}")

        with extra_file.open("r", encoding="utf-8") as f:
            if extra_file.suffix == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            elif extra_file.suffix == ".json":
                rows = json.load(f)
            else:
                raise ValueError(f"Unsupported complete segment file extension: {extra_file.suffix}")

        complete_intervals = {}
        for row in rows:
            episode_index = int(row["episode_index"])
            intervals = complete_intervals.setdefault(episode_index, [])
            for segment in row.get("segments", []):
                if str(segment.get("phase_id")) == "4":
                    intervals.append((int(segment["start_idx"]), int(segment["end_idx"])))
        self.complete_intervals = complete_intervals
        return complete_intervals

    def _build_subtask_id(self, df: pd.DataFrame, task_index: int) -> np.ndarray:
        dataset_type = str(self.config.dataset_type).lower()
        if dataset_type == "libero":
            return self._build_libero_subtask_id(df, task_index)
        raise KeyError(f"Unsupported dataset_type for subtask_id: {self.config.dataset_type}")

    def _build_libero_subtask_id(self, df: pd.DataFrame, task_index: int) -> np.ndarray:
        mapping: dict[str, int] = {}
        ids = []
        for subtask in df["subtask"]:
            key = str(subtask).strip().lower()
            if key not in mapping:
                mapping[key] = len(mapping) + 1
            ids.append(mapping[key])
        return np.asarray(ids, dtype=np.int32)

    def _build_node_points_mask(self, df: pd.DataFrame, taskstructure: TaskStructure) -> np.ndarray:
        dataset_type = str(self.config.dataset_type).lower()
        if dataset_type != "libero":
            raise KeyError(f"Unsupported dataset_type for node_points_mask: {self.config.dataset_type}")
        if "subtask_id" not in df.columns:
            raise KeyError("subtask_id is required to build node_points_mask")

        spans = []
        cursor = 0
        for subtask in taskstructure.subtask_list:
            count = sum(
                1
                for node in (subtask.node_list or [])
                if bool(node.need_object) and node.role != NodeRole.ACTOR
            )
            spans.append((cursor, cursor + count))
            cursor += count

        masks = np.zeros((len(df), self.config.max_nodes), dtype=bool)
        for row_index, subtask_id in enumerate(df["subtask_id"]):
            subtask_index = int(subtask_id) - 1
            if 0 <= subtask_index < len(spans):
                start, end = spans[subtask_index]
                masks[row_index, start:min(end, self.config.max_nodes)] = True
        return masks

    def _build_gripper_uvd(self, df: pd.DataFrame, task_index: int) -> list[list[list[float]]]:
        dataset_type = str(self.config.dataset_type).lower()
        if dataset_type == "libero":
            return self._build_libero_gripper_uvd(df, task_index)
        raise KeyError(f"Unsupported dataset_type for gripper_uvd: {self.config.dataset_type}")

    def _build_gripper_openness(self, df: pd.DataFrame) -> np.ndarray:
        if str(self.config.dataset_type).lower() != "libero":
            raise KeyError(
                f"Unsupported dataset_type for gripper_openness: {self.config.dataset_type}"
            )
        states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] < 8:
            raise ValueError(
                "observation.state must have shape [frames, >=8] to build gripper_openness"
            )
        widths = np.abs(states[:, 6]) + np.abs(states[:, 7])
        return np.clip(widths / LIBERO_GRIPPER_MAX_WIDTH, 0.0, 1.0).astype(np.float32)

    def _build_libero_gripper_uvd(self, df: pd.DataFrame, task_index: int) -> list[list[list[float]]]:
        camera = self._load_libero_cameras()[int(task_index)]["agentview"]
        intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
        geometry = self._ensure_gripper_geometry()

        results = []
        for state in df["observation.state"]:
            state = np.asarray(state, dtype=np.float64)
            if state.size < 8:
                raise ValueError(
                    "observation.state must contain at least 8 values to build gripper_uvd"
                )
            output = geometry.project_gripper_to_uvd(
                tcp_state=state[:6],
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                gripper_width=abs(float(state[6])) + abs(float(state[7])),
            )
            results.append(self._flatten_gripper_uvd(output))
        return results

    @staticmethod
    def _flatten_gripper_uvd(output: Mapping[str, object]) -> list[list[float]]:
        return [
            np.asarray(output[key], dtype=np.float64).astype(float).tolist()
            for key in (
                "root_uvd",
                "left_base_uvd",
                "right_base_uvd",
                "left_fingertip_uvd",
                "right_fingertip_uvd",
                "tcp_uvd",
            )
        ]

    def _load_libero_cameras(self) -> dict[int, dict]:
        if self.libero_cameras is not None:
            return self.libero_cameras

        camera_file = self.meta_dir / "cameras.json"
        with camera_file.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        self.libero_cameras = {int(row["task_index"]): row["cameras"] for row in rows}
        return self.libero_cameras

    def _ensure_gripper_geometry(self):
        if self.gripper_geometry is None:
            from examples.libero.embodiment.robot import GeomFrankaPanda
            self.gripper_geometry = GeomFrankaPanda()
        return self.gripper_geometry

    def _video_image_size(self) -> tuple[int, int]:
        feature = self.info["features"][self.video_key]
        info = feature.get("info", {})
        if "video.height" in info and "video.width" in info:
            return int(info["video.height"]), int(info["video.width"])

        shape = feature.get("shape")
        names = feature.get("names") or []
        if shape and "height" in names and "width" in names:
            return int(shape[names.index("height")]), int(shape[names.index("width")])
        if shape and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        raise RuntimeError(f"Cannot infer image size from video feature: {feature}")

    @staticmethod
    def _largest_mask(masks: Sequence[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
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

    def _scan_task_episodes(self) -> dict[int, list[int]]:
        task_to_episodes = {}
        for parquet_path in sorted((self.dataset_dir / "data").glob("chunk-*/*.parquet")):
            meta = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
            episode_index = int(meta["episode_index"].iloc[0])
            task_index = int(meta["task_index"].iloc[0])
            task_to_episodes.setdefault(task_index, []).append(episode_index)
        for episodes in task_to_episodes.values():
            episodes.sort()
        return task_to_episodes

    def _print_save_plan(self):
        cs.rule()
        cs.print("[bold cyan]Pipeline outputs[/bold cyan]")
        if self.config.debug:
            cs.print("taskstructures jsonl: debug dry-run prints generated taskstructures instead of appending")
            cs.print("parquet fields: debug dry-run does not write original parquet files")
        else:
            cs.print(f"taskstructures jsonl: {self.taskstructures_jsonl_path}")
            cs.print(
                "parquet fields: node_points_track, node_points_mask, depths_rel, "
                "far_background_mask, is_complete, gripper_uvd, gripper_openness, subtask_id"
            )
        if self.config.debug:
            cs.print(f"debug node locator images: {self.node_locator_vis_dir}")
            cs.print(f"debug point tracker videos: {self.point_tracker_vis_dir}")
            cs.print(f"debug depth videos: {self.depth_vis_dir}")
            cs.print(f"debug far background overlays: {self.far_background_vis_dir}")
            cs.print(f"debug gripper uvd videos: {self.gripper_uvd_vis_dir}")
            cs.print(f"debug pointcloud npy: {self.pointcloud_dir}")
        else:
            cs.print("debug outputs: disabled (use --debug to enable)")
        cs.rule()

    def _object_nodes(self, taskstructure: TaskStructure):
        nodes = []
        for subtask in taskstructure.subtask_list:
            for node in subtask.node_list or []:
                if bool(node.need_object) and node.role != NodeRole.ACTOR:
                    nodes.append(node)
        return nodes

    def _locate_node_points(self, frame_rgb: np.ndarray, node_name: str) -> list[list[float]]:
        result = self.node_locator.inference(
            text=node_name,
            image=Image.fromarray(frame_rgb),
        )
        points = result.get("points") or []
        if not points:
            raise RuntimeError(f"NodeLocatorRobo found no points for node={node_name!r}")
        return self._locator_points_to_pixels(points, frame_rgb.shape[:2])

    def _locator_points_to_pixels(self, points, image_size: tuple[int, int]) -> list[list[float]]:
        height, width = image_size
        converted = []
        for x, y in points:
            x, y = float(x), float(y)
            if x > width or y > height:
                x = x / 1000.0 * width
                y = y / 1000.0 * height
            converted.append([
                float(np.clip(x, 0, width - 1)),
                float(np.clip(y, 0, height - 1)),
            ])
        return converted

    def _save_node_locator_vis(
        self,
        frame_rgb: np.ndarray,
        nodes,
        point_prompts: Sequence[Sequence[Sequence[float]]],
        episode_index: int,
        task_index: int,
    ):
        self.node_locator_vis_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
        colors = [
            (60, 60, 255),
            (255, 144, 30),
            (50, 205, 50),
            (0, 215, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        for node_index, (node, points) in enumerate(zip(nodes, point_prompts)):
            color = colors[node_index % len(colors)]
            for point in points:
                x, y = np.round(point).astype(int)
                cv2.circle(image, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(image, (int(x), int(y)), 7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            if points:
                x, y = np.round(points[0]).astype(int)
                label = f"{node_index}:{node.name}"
                cv2.putText(
                    image,
                    label,
                    (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        save_path = self.node_locator_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.png"
        cv2.imwrite(os.fspath(save_path), image)

    def _save_point_tracker_vis_video(
        self,
        frames: Sequence[np.ndarray],
        tracker_results: Sequence[dict],
        episode_index: int,
        task_index: int,
        node_points_masks: np.ndarray | None = None,
    ):
        if not frames:
            return
        self.point_tracker_vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.point_tracker_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.mp4"
        height, width = frames[0].shape[:2]
        fps = float(self.info.get("fps", 10) or 10)
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
                "-pix_fmt", "yuv420p",
                os.fspath(save_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame_index, (frame, result) in enumerate(zip(frames, tracker_results)):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mask = None if node_points_masks is None else node_points_masks[frame_index]
                drawn = self._draw_point_tracker_result(frame_bgr, result, mask)
                proc.stdin.write(drawn.tobytes())
            proc.stdin.close()
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed for {save_path}: {stderr}")


    @staticmethod
    def _draw_point_tracker_result(
        image_bgr: np.ndarray,
        result: Mapping[str, np.ndarray],
        node_points_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        colors = [
            (60, 60, 255),
            (255, 144, 30),
            (50, 205, 50),
            (0, 215, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        drawn = image_bgr.copy()
        points = np.asarray(result["points"], dtype=np.float32)
        visibles = np.asarray(result.get("visibles", np.ones(points.shape[:2], dtype=bool)), dtype=bool)
        if node_points_mask is None:
            node_points_mask = np.ones(points.shape[0], dtype=bool)
        else:
            node_points_mask = np.asarray(node_points_mask, dtype=bool)[:points.shape[0]]

        active = np.zeros(points.shape[0], dtype=bool)
        active[:len(node_points_mask)] = node_points_mask[:points.shape[0]]
        draw_order = list(np.where(~active)[0]) + list(np.where(active)[0])
        for node_index in draw_order:
            color = colors[node_index % len(colors)] if active[node_index] else (145, 145, 145)
            pale_color = tuple(int(round(c * 0.35 + 255 * 0.65)) for c in color)
            for point, visible in zip(points[node_index], visibles[node_index]):
                x, y = np.round(point).astype(int)
                draw_color = color if visible else pale_color
                cv2.circle(drawn, (int(x), int(y)), 2, draw_color, -1, lineType=cv2.LINE_AA)
        return drawn

    def _save_pointcloud_npy(
        self,
        depths: np.ndarray,
        frames: Sequence[np.ndarray],
        intrinsic: np.ndarray,
        episode_index: int,
        task_index: int,
    ):
        self.pointcloud_dir.mkdir(parents=True, exist_ok=True)
        pointcloud = self._depths_to_pointcloud_thw6(depths, frames, intrinsic)
        save_path = self.pointcloud_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.npy"
        np.save(save_path, pointcloud)

    def _save_depth_vis_video(
        self,
        depths: np.ndarray,
        episode_index: int,
        task_index: int,
    ):
        self.depth_vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.depth_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.mp4"
        self.depth_predictor.draw_on_image(depths, save_path=os.fspath(save_path))

    def _save_far_background_overlay_video(
        self,
        frames: Sequence[np.ndarray],
        masks: np.ndarray,
        episode_index: int,
        task_index: int,
    ):
        if not frames:
            return
        self.far_background_vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.far_background_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.mp4"
        height, width = frames[0].shape[:2]
        fps = float(self.info.get("fps", 10) or 10)
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
                "-pix_fmt", "yuv420p",
                os.fspath(save_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame, mask in zip(frames, masks):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                overlay = np.zeros_like(frame_bgr)
                overlay[mask.astype(bool)] = (255, 80, 0)
                drawn = np.where(mask[..., None], (frame_bgr * 0.55 + overlay * 0.45).astype(np.uint8), frame_bgr)
                proc.stdin.write(drawn.tobytes())
            proc.stdin.close()
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed for {save_path}: {stderr}")

    def _save_gripper_uvd_vis_video(
        self,
        frames: Sequence[np.ndarray],
        gripper_uvds: Sequence[Sequence[Sequence[float]]],
        gripper_opennesses: Sequence[float],
        episode_index: int,
        task_index: int,
    ):
        if not frames:
            return
        self.gripper_uvd_vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.gripper_uvd_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.mp4"
        height, width = frames[0].shape[:2]
        fps = float(self.info.get("fps", 10) or 10)
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
                "-pix_fmt", "yuv420p",
                os.fspath(save_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame, uvd, openness in zip(frames, gripper_uvds, gripper_opennesses):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                openness_value = float(np.asarray(openness).reshape(-1)[0])
                drawn = self._draw_gripper_uvd(
                    frame_bgr, uvd, openness_value,
                )
                proc.stdin.write(drawn.tobytes())
            proc.stdin.close()
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed for {save_path}: {stderr}")

    def _draw_gripper_uvd(
        self,
        image_bgr: np.ndarray,
        uvd: Sequence[Sequence[float]],
        openness: float,
    ) -> np.ndarray:
        image = image_bgr.copy()
        root, left_base, right_base, left_tip, right_tip, tcp = np.asarray(
            uvd, dtype=np.float64,
        )

        for point, color, radius in (
            (root, (255, 0, 0), 5),
            (left_base, (0, 255, 0), 5),
            (right_base, (0, 255, 255), 5),
            (left_tip, (255, 0, 255), 5),
            (right_tip, (255, 255, 0), 5),
            (tcp, (0, 0, 255), 6),
        ):
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, (x, y), radius + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        for left, right in ((left_base, right_base), (left_tip, right_tip)):
            cv2.line(
                image,
                tuple(np.round(left[:2]).astype(int)),
                tuple(np.round(right[:2]).astype(int)),
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        label = f"open={openness:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1,
        )
        cv2.rectangle(image, (0, 0), (text_width + 16, text_height + 14), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (8, text_height + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def _depths_to_pointcloud_thw6(
        self,
        depths: np.ndarray,
        frames: Sequence[np.ndarray],
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        depths = np.asarray(depths, dtype=np.float32)
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        frame_array = np.asarray(frames, dtype=np.float32) / 255.0
        time, height, width = depths.shape
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        z = depths
        x = (xs[None] - cx) / max(fx, 1e-6) * z
        y = (ys[None] - cy) / max(fy, 1e-6) * z
        xyz = np.stack([x, y, z], axis=-1)
        valid = np.isfinite(z) & (z > 0)
        pointcloud = np.concatenate([xyz, frame_array], axis=-1).astype(np.float32)
        pointcloud[~valid] = np.nan
        return pointcloud.reshape(time, height, width, 6)

    def _ensure_node_modules(self):
        if self.node_locator is None:
            self.node_locator = NodeLocatorRobo()
        if self.node_segmenter is None:
            self.node_segmenter = NodeSegmenter()
        if self.point_tracker is None:
            self.point_tracker = PointTracker()

    def _ensure_depth_modules(self):
        if self.depth_predictor is None:
            self.depth_predictor = DepthPredictorSTream3R()

    def _ensure_far_background_modules(self):
        if self.binary_segmenter is None:
            self.binary_segmenter = BinarySegmenter(fp32=True)
        if self.node_segmenter is None:
            self.node_segmenter = NodeSegmenter()

    def _infer_video_key(self) -> str:
        video_keys = [
            key for key, spec in self.info["features"].items()
            if spec.get("dtype") == "video"
        ]
        for key in ("observation.images.rgb_static", "observation.images.image"):
            if key in video_keys:
                return key
        for key in video_keys:
            lowered = key.lower()
            if "wrist" not in lowered and "gripper" not in lowered:
                return key
        if not video_keys:
            raise RuntimeError("No video feature found in meta/info.json")
        return video_keys[0]

    def _episode_parquet_path(self, episode_index: int) -> Path:
        episode_chunk = int(episode_index) // self.chunk_size
        return self.dataset_dir / self.data_path_template.format(
            episode_chunk=episode_chunk,
            episode_index=int(episode_index),
        )

    def _episode_video_path(self, episode_index: int) -> Path:
        episode_chunk = int(episode_index) // self.chunk_size
        return self.dataset_dir / self.video_path_template.format(
            episode_chunk=episode_chunk,
            video_key=self.video_key,
            episode_index=int(episode_index),
        )

    def _read_video_frames(self, video_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(os.fspath(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        frames = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if str(self.config.dataset_type).lower() == "libero":
                    frame_rgb = np.ascontiguousarray(np.fliplr(frame_rgb))
                frames.append(frame_rgb)
        finally:
            cap.release()
        return frames

    def _task_analyzer(self) -> TaskAnalyzer:
        if self.task_analyzer is None:
            if not self.config.task_analyzer_api_key:
                raise RuntimeError(
                    "task_analyzer_api_key is required when taskstructure is missing from "
                    f"{self.taskstructures_jsonl_path}"
                )
            self.task_analyzer = TaskAnalyzer(api_key=self.config.task_analyzer_api_key)
        return self.task_analyzer


    def _ensure_output_features(self) -> None:
        features = dict(self.info.get("features", {}))
        output_features = {
            "node_points_track": {
                "dtype": "float32",
                "shape": [self.config.max_nodes, self.config.points_per_node, 3],
                "names": ["node", "point", "xyv"],
            },
            "node_points_mask": {
                "dtype": "bool",
                "shape": [self.config.max_nodes],
                "names": ["node"],
            },
            "depths_rel": {
                "dtype": "float32",
                "shape": [256, 256],
                "names": ["height", "width"],
            },
            "far_background_mask": {
                "dtype": "bool",
                "shape": [256, 256],
                "names": ["height", "width"],
            },
            "gripper_uvd": {
                "dtype": "float32",
                "shape": [6, 3],
                "names": ["point", "uvd"],
            },
            "gripper_openness": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "is_complete": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
            "subtask_id": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        }

        changed = False
        for key, spec in output_features.items():
            if features.get(key) != spec:
                features[key] = spec
                changed = True

        if not changed:
            return

        self.info["features"] = features
        info_path = self.meta_dir / "info.json"
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _load_taskstructures_jsonl(self) -> dict[int, TaskStructure]:
        taskstructures = {}
        if not self.taskstructures_jsonl_path.exists():
            return taskstructures

        task_to_index = self._load_task_to_index()
        with self.taskstructures_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                taskstructure = json_to_taskstructure(data)
                if taskstructure.task not in task_to_index:
                    continue
                taskstructures[task_to_index[taskstructure.task]] = taskstructure
        return taskstructures

    def _append_taskstructure_jsonl(self, task_index: int, taskstructure: TaskStructure):
        data = taskstructure_to_json(taskstructure)
        if self.config.debug:
            cs.print(f"[yellow]debug dry-run: skip appending taskstructure for task_index={task_index}[/yellow]")
            cs.print_json(data=data)
            return

        self.taskstructures_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.taskstructures_jsonl_path.open("a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")

    def _load_task_index_to_desc(self) -> dict[int, str]:
        task_index_to_desc = {}
        with self.tasks_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_index_to_desc[int(data["task_index"])] = str(data["task"])
        return task_index_to_desc

    def _load_task_to_index(self) -> dict[str, int]:
        return {task_desc: task_index for task_index, task_desc in self.task_index_to_desc.items()}

    def _load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", default="/data0/luokang/research/GraphVLA/__tmp__/pipeline")
    args = parser.parse_args()
    api_key = "sk-UZpG2yYwDE5itw7s57eIJA"

    config = PipelineConfig(
        dataset_dir="/data0/luokang/dataset/luokang/lerobot/libero/libero_31_no_noops_1.0.0_lerobot_10hz",
        dataset_type="libero",
        episode_selector={
            # 30: [0],
            # 31: [0],
            # 32: [0],
            # 33: [0],
            # 34: [0],
            # 30: ["*"],
            # 31: ["*"],
            # 32: ["*"],
            # 33: ["*"],
            # 34: ["*"],
            0: ["*"],
            # 0: [0]
        },
        points_per_node=32,
        overwrite={
            "taskstructure": False,
            "node_points_track": True,
            "depths_rel": True,
            "far_background_mask": True,
            "is_complete": True,
            "gripper_uvd": True,
            "gripper_openness": True,
            "subtask_id": True,
        },
        task_analyzer_api_key=api_key,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    OfflinePipeline(config).run()
