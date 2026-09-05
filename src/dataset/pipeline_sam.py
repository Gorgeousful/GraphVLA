from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
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
from src.common.schema import NodeRole, TaskStructure, json_to_taskstructure, taskstructure_to_json
from src.module.node_segmenter import NodeSegmenter, NodeSegmenterSAM2
from src.module.task_analyzer import TaskAnalyzer

cs = Console()


def _run_worker(
    config: "PipelineConfig",
    episode_indices: Sequence[int],
    worker_id: int,
) -> None:
    pipeline = OfflineSAMPipeline(config)
    task_indices = sorted(
        {
            int(
                pd.read_parquet(
                    pipeline._episode_parquet_path(episode_index),
                    columns=["task_index"],
                )["task_index"].iloc[0]
            )
            for episode_index in episode_indices
        }
    )
    taskstructures = pipeline.build_taskstructures(task_indices)
    started = time.monotonic()
    cs.print(
        f"worker {worker_id} starting: device={config.sam_device}, "
        f"episodes={len(episode_indices)}"
    )
    pipeline.process_episodes(episode_indices, taskstructures)
    cs.print(
        f"worker {worker_id} finished: device={config.sam_device}, "
        f"episodes={len(episode_indices)}, elapsed={time.monotonic() - started:.1f}s"
    )


@dataclass
class PipelineConfig:
    dataset_dir: str | os.PathLike
    episode_selector: Mapping[str | int, Sequence[int | str]]
    dataset_type: str = "libero"
    task_analyzer_api_key: str | None = None
    points_per_node: int = 32
    max_nodes: int = 10
    erode_pixel: int = 2
    sam_version: str = "sam3"
    sam_device: str = "cuda"
    overwrite: Mapping[str, bool] | None = None
    debug: bool = False
    debug_dir: str | os.PathLike = "/data0/luokang/research/GraphVLA/__tmp__/pipeline_sam"


class OfflineSAMPipeline:
    """Precompute LIBERO entity points with SAM mask tracking."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        if config.sam_version not in {"sam2", "sam3"}:
            raise ValueError(f"Unsupported sam_version={config.sam_version!r}; expected 'sam2' or 'sam3'")
        self.dataset_dir = Path(config.dataset_dir)
        self.meta_dir = self.dataset_dir / "meta"
        self.taskstructures_jsonl_path = self.meta_dir / "taskstructures.jsonl"
        self.tasks_jsonl_path = self.meta_dir / "tasks.jsonl"
        self.node_initialization_cache_dir = self.meta_dir / "node_initialization_cache"
        debug_root = Path(config.debug_dir)
        if config.debug and debug_root.resolve().is_relative_to(self.dataset_dir.resolve()):
            raise ValueError("debug_dir must be outside dataset_dir in debug mode")
        self.debug_dir = (
            debug_root / datetime.now().strftime("%Y%m%d_%H%M%S")
            if config.debug
            else debug_root
        )
        self.sam_initialization_vis_dir = self.debug_dir / "sam_initialization"
        self.sam_tracker_vis_dir = self.debug_dir / "sam_tracker"
        overwrite_defaults = {
            "taskstructure": False,
            "node_points_xyz": True,
            "valid_node_mask": True,
            "subtask_node_mask": True,
            "gripper_points_xyz": False,
        }
        unknown_overwrite = set(config.overwrite or {}) - set(overwrite_defaults)
        if unknown_overwrite:
            raise KeyError(f"Unknown overwrite keys: {sorted(unknown_overwrite)}")
        self.overwrite = {**overwrite_defaults, **(config.overwrite or {})}
        self.info = self._load_json(self.meta_dir / "info.json")
        self.data_path_template = self.info["data_path"]
        self.chunk_size = int(self.info.get("chunks_size", 1000))
        self.image_key = self._infer_image_key()
        self.task_index_to_desc = self._load_task_index_to_desc()

        self.task_analyzer = None
        self.node_segmenter = None
        self.libero_cameras = None
        self.gripper_geometry = None

    def run(self):
        selector = self.config.episode_selector
        if selector is None:
            raise TypeError("episode_selector must be a mapping; use {} to select all episodes")

        if selector:
            task_indices = [int(task_index) for task_index in selector]
            episode_indices = self.resolve_episode_indices(selector)
        else:
            task_to_episodes = self._scan_task_episodes()
            task_indices = sorted(task_to_episodes)
            episode_indices = sorted(
                episode
                for episodes in task_to_episodes.values()
                for episode in episodes
            )
            cs.print(f"selected all {len(episode_indices)} episodes")

        self._preflight_node_initialization_caches(episode_indices)
        self._ensure_output_features()
        taskstructures = self.build_taskstructures(task_indices)
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

        need_node_tracking = (
            self.overwrite["node_points_xyz"]
            or "node_points_xyz" not in df.columns
            or self.overwrite["valid_node_mask"]
            or "valid_node_mask" not in df.columns
        )
        need_subtask_node_mask = (
            self.overwrite["subtask_node_mask"]
            or "subtask_node_mask" not in df.columns
        )
        need_gripper_points_xyz = self.overwrite["gripper_points_xyz"] or "gripper_points_xyz" not in df.columns
        if "subtask_id" not in df.columns:
            raise KeyError(f"subtask_id is required in {parquet_path}")
        if (
            not need_node_tracking
            and not need_subtask_node_mask
            and not need_gripper_points_xyz
        ):
            return

        subtask_node_mask = self._build_subtask_node_mask(df, taskstructures[task_index])
        if need_node_tracking:
            frame_flipped = self._node_cache_frame_flipped(episode_index)
            frames = self._read_episode_frames(
                df, episode_index=episode_index, horizontal_flip=frame_flipped
            )
            tracks = self._build_node_points_track(
                frames,
                taskstructures[task_index],
                episode_index=episode_index,
                task_index=task_index,
                subtask_node_masks=subtask_node_mask if self.config.debug else None,
            )
            intrinsic = np.asarray(
                self._load_libero_cameras()[task_index]["agentview"]["intrinsic"], dtype=np.float64
            )
            node_xyz, _, valid_node_mask = self._build_node_points_xyz(
                tracks, self._read_metric_depths(df, horizontal_flip=frame_flipped), intrinsic,
            )
            df["node_points_xyz"] = [points.tolist() for points in node_xyz]
            df["valid_node_mask"] = [mask.tolist() for mask in valid_node_mask]
        if need_subtask_node_mask:
            df["subtask_node_mask"] = [mask.tolist() for mask in subtask_node_mask]

        if need_gripper_points_xyz:
            df["gripper_points_xyz"] = self._build_gripper_points_xyz(df, task_index)

        if self.config.debug:
            cs.print(f"[yellow]debug dry-run: skip writing parquet {parquet_path}[/yellow]")
            return
        self._write_episode_parquet(df, parquet_path)

    def _write_episode_parquet(self, df: pd.DataFrame, parquet_path: Path) -> None:
        if self.config.debug:
            raise RuntimeError("debug mode cannot write dataset parquet files")
        table = pa.Table.from_pandas(df, preserve_index=False)
        target_types = {
            "node_points_xyz": pa.list_(pa.list_(pa.list_(pa.float32()))),
            "valid_node_mask": pa.list_(pa.bool_()),
            "subtask_node_mask": pa.list_(pa.bool_()),
            "gripper_points_xyz": pa.list_(pa.list_(pa.float32())),
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
        subtask_node_masks: np.ndarray | None = None,
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

        initial_masks = self._load_node_initialization_cache(
            episode_index, task_index, nodes, frames[0].shape[:2]
        )
        first_node_by_name = {}
        unique_node_indices = []
        unique_index_by_node = []
        for node_index, node in enumerate(nodes):
            if node.name not in first_node_by_name:
                first_node_by_name[node.name] = len(unique_node_indices)
                unique_node_indices.append(node_index)
            unique_index_by_node.append(first_node_by_name[node.name])

        unique_initial_points = np.stack(
            [
                sample_points_from_mask(
                    initial_masks[node_index],
                    num_points=self.config.points_per_node,
                    erode_pixel=self.config.erode_pixel,
                )
                for node_index in unique_node_indices
            ],
            axis=0,
        ).astype(np.float32)
        initial_points = unique_initial_points[unique_index_by_node]
        self._ensure_node_segmenter()

        if self.config.debug:
            self._save_sam_initialization_vis(
                frames[0],
                nodes,
                initial_points,
                episode_index=episode_index,
                task_index=task_index,
            )

        frame_tracks = []
        sam_results = []
        for frame_index, frame in enumerate(frames):
            masks = self.node_segmenter.predict(
                frame,
                points=unique_initial_points if frame_index == 0 else None,
                anchor_frame=(frame_index == 0),
            )
            if len(masks) != len(unique_node_indices):
                raise RuntimeError(
                    f"episode={episode_index} frame={frame_index}: "
                    f"{self.config.sam_version} returned {len(masks)} masks "
                    f"for {len(unique_node_indices)} unique node names"
                )

            unique_points = np.zeros(
                (len(unique_node_indices), self.config.points_per_node, 2),
                dtype=np.float32,
            )
            unique_visibles = np.zeros(
                (len(unique_node_indices), self.config.points_per_node), dtype=bool
            )
            for unique_index, mask in enumerate(masks):
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != frame.shape[:2]:
                    raise ValueError(
                        f"episode={episode_index} frame={frame_index} "
                        f"unique_node={unique_index}: mask shape {mask.shape} "
                        f"does not match frame shape {frame.shape[:2]}"
                    )
                if mask.any():
                    unique_points[unique_index] = sample_points_from_mask(
                        mask,
                        num_points=self.config.points_per_node,
                        erode_pixel=self.config.erode_pixel,
                    )
                    unique_visibles[unique_index] = True

            points = unique_points[unique_index_by_node]
            visibles = unique_visibles[unique_index_by_node]

            frame_tracks.append(
                np.concatenate([points, visibles[..., None]], axis=-1)
            )
            if self.config.debug:
                sam_results.append({"points": points, "visibles": visibles})

        if self.config.debug:
            self._save_sam_tracker_vis_video(
                frames,
                sam_results,
                episode_index=episode_index,
                task_index=task_index,
                subtask_node_masks=subtask_node_masks,
            )

        tracks = np.stack(frame_tracks, axis=0).astype(np.float32)
        padded_tracks = np.zeros(
            (len(frames), self.config.max_nodes, self.config.points_per_node, 3),
            dtype=np.float32,
        )
        padded_tracks[:, :tracks.shape[1]] = tracks
        return padded_tracks

    def _load_node_initialization_cache(
        self,
        episode_index: int,
        task_index: int,
        nodes,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        path = self.node_initialization_cache_dir / f"episode_{episode_index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing node initialization cache: {path}. "
                f"Run `python src/dataset/pipeline_init.py initialize "
                f"--dataset-dir {self.dataset_dir}` first."
            )

        with np.load(path, allow_pickle=False) as cache:
            required = {
                "episode_index", "task_index", "node_names", "masks", "metadata_json",
            }
            missing = required - set(cache.files)
            if missing:
                raise ValueError(f"Cache {path} is missing fields: {sorted(missing)}")

            cached_episode = int(cache["episode_index"])
            cached_task = int(cache["task_index"])
            node_names = cache["node_names"].tolist()
            masks = np.asarray(cache["masks"], dtype=bool)
            metadata = json.loads(str(cache["metadata_json"].item()))

        expected_names = [node.name for node in nodes]
        height, width = image_size
        if cached_episode != episode_index or cached_task != task_index:
            raise ValueError(
                f"Cache {path} identifies episode={cached_episode}, task={cached_task}; "
                f"expected episode={episode_index}, task={task_index}"
            )
        if node_names != expected_names:
            raise ValueError(
                f"Cache {path} nodes do not match taskstructure: "
                f"{node_names} != {expected_names}"
            )
        if not isinstance(metadata.get("frame_flipped"), bool):
            raise ValueError(f"Cache {path} is missing the frame_flipped flag")
        if masks.shape != (len(nodes), height, width):
            raise ValueError(
                f"Cache {path} masks have shape {masks.shape}; "
                f"expected {(len(nodes), height, width)}"
            )
        for node_index, mask in enumerate(masks):
            if not mask.any():
                raise ValueError(f"Cache {path} contains an empty mask for node {node_index}")

        cs.print(f"using node initialization cache masks: {path}")
        return masks

    def _build_node_points_xyz(
        self,
        tracks: np.ndarray,
        metric_depths: np.ndarray,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backproject tracked points in their original order and return SAM mask visibility."""
        tracks = np.asarray(tracks, dtype=np.float32)
        metric_depths = np.asarray(metric_depths, dtype=np.float32)
        if tracks.ndim != 4 or tracks.shape[-1] < 3:
            raise ValueError(f"Expected tracks [T,N,P,3], got {tracks.shape}")
        if metric_depths.ndim != 3 or metric_depths.shape[0] != tracks.shape[0]:
            raise ValueError(f"Metric depth {metric_depths.shape} does not match tracks {tracks.shape}")

        height, width = metric_depths.shape[1:]
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        output = np.zeros(tracks.shape[:3] + (3,), dtype=np.float32)
        point_vis = tracks[..., 2] > 0.5
        valid_nodes = np.zeros(tracks.shape[:2], dtype=bool)
        for frame_index in range(tracks.shape[0]):
            for node_index in range(tracks.shape[1]):
                uv = tracks[frame_index, node_index, :, :2]
                u = np.rint(uv[:, 0]).astype(np.int64)
                v = np.rint(uv[:, 1]).astype(np.int64)
                in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
                safe_u = np.clip(u, 0, width - 1)
                safe_v = np.clip(v, 0, height - 1)
                z = metric_depths[frame_index, safe_v, safe_u]
                projectable = in_bounds & np.isfinite(z) & (z > 0.0)
                output[frame_index, node_index, projectable] = np.stack([
                    (uv[projectable, 0] - cx) / fx * z[projectable],
                    (uv[projectable, 1] - cy) / fy * z[projectable],
                    z[projectable],
                ], axis=-1)
                valid_nodes[frame_index, node_index] = np.any(
                    point_vis[frame_index, node_index] & projectable
                )
        return output, point_vis, valid_nodes

    def _node_cache_frame_flipped(self, episode_index: int) -> bool:
        path = self.node_initialization_cache_dir / f"episode_{episode_index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing node initialization cache: {path}. "
                f"Run `python src/dataset/pipeline_init.py initialize "
                f"--dataset-dir {self.dataset_dir}` first."
            )
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"].item()))
        return bool(metadata.get("frame_flipped", True))

    def _read_episode_frames(
        self,
        df: pd.DataFrame,
        *,
        episode_index: int,
        horizontal_flip: bool = True,
    ) -> list[np.ndarray]:
        if self.image_key not in df.columns:
            frames = self._read_video_frames(episode_index)
            if len(frames) != len(df):
                raise ValueError(
                    f"Video episode={episode_index} has {len(frames)} frames; "
                    f"expected {len(df)}"
                )
            return [
                np.ascontiguousarray(np.fliplr(frame))
                if horizontal_flip
                else np.ascontiguousarray(frame)
                for frame in frames
            ]

        frames = []
        for item in df[self.image_key]:
            payload = item.get("bytes") if isinstance(item, Mapping) else None
            if payload is None:
                raise ValueError(f"{self.image_key} must contain embedded image bytes")
            with Image.open(io.BytesIO(payload)) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.append(
                np.ascontiguousarray(np.fliplr(frame))
                if horizontal_flip
                else np.ascontiguousarray(frame)
            )
        return frames

    def _read_video_frames(self, episode_index: int) -> list[np.ndarray]:
        try:
            import av
        except ImportError as error:
            raise RuntimeError("PyAV is required to decode episode videos") from error

        episode_chunk = int(episode_index) // self.chunk_size
        template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        video_path = self.dataset_dir / template.format(
            episode_chunk=episode_chunk,
            episode_index=int(episode_index),
            video_key=self.image_key,
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing episode video: {video_path}")

        frames = []
        try:
            with av.open(str(video_path)) as container:
                for frame in container.decode(video=0):
                    frames.append(
                        np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                    )
        except av.error.FFmpegError as error:
            raise ValueError(f"Unable to decode episode video: {video_path}") from error
        if not frames:
            raise ValueError(f"No decodable frames in episode video: {video_path}")
        return frames

    @staticmethod
    def _read_metric_depths(df: pd.DataFrame, horizontal_flip: bool = True) -> np.ndarray:
        field = next(
            (
                name
                for name in ("observation.depth.agentview", "agentview_real_depth_images")
                if name in df
            ),
            None,
        )
        if field is None:
            raise KeyError("Dataset is missing agent-view metric depth")
        depth = np.stack(
            [np.stack(value.tolist()).astype(np.float32) for value in df[field]],
            axis=0,
        )
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 2:
            side = int(round(np.sqrt(depth.shape[1])))
            if side * side != depth.shape[1]:
                raise ValueError(f"Metric depth size {depth.shape[1]} is not square")
            depth = depth.reshape(-1, side, side)
        if depth.ndim != 3:
            raise ValueError(f"Expected metric depth [T,H,W], got {depth.shape}")
        if horizontal_flip:
            depth = np.flip(depth, axis=2)
        return np.ascontiguousarray(depth)

    def _build_gripper_points_xyz(self, df: pd.DataFrame, task_index: int) -> list[list[list[float]]]:
        camera = self._load_libero_cameras()[int(task_index)]["agentview"]
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
        geometry = self._ensure_gripper_geometry()
        results = []
        state_field = "observation.state" if "observation.state" in df else "state"
        if state_field not in df:
            raise KeyError("Dataset is missing observation state")
        for raw_state in df[state_field]:
            state = np.asarray(raw_state, dtype=np.float64)
            if state.size < 8:
                raise ValueError("state must contain at least 8 values to build gripper_points_xyz")
            gripper_width = abs(float(state[6])) + abs(float(state[7]))
            xyz = geometry.project_gripper_to_xyz(
                tcp_state=state[:6],
                extrinsic=extrinsic,
                gripper_width=gripper_width,
            )
            results.append(np.asarray(xyz, dtype=np.float32).tolist())
        return results

    def _build_subtask_node_mask(self, df: pd.DataFrame, taskstructure: TaskStructure) -> np.ndarray:
        dataset_type = str(self.config.dataset_type).lower()
        if dataset_type != "libero":
            raise KeyError(f"Unsupported dataset_type for subtask_node_mask: {self.config.dataset_type}")
        if "subtask_id" not in df.columns:
            raise KeyError("subtask_id is required to build subtask_node_mask")

        spans = []
        cursor = 0
        for subtask in taskstructure.subtask_list:
            count = sum(
                1
                for node in (subtask.node_list or [])
                if node.role != NodeRole.ACTOR
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
                "parquet fields: node_points_xyz, valid_node_mask, "
                "subtask_node_mask, gripper_points_xyz"
            )
        if self.config.debug:
            cs.print(f"debug SAM initialization images: {self.sam_initialization_vis_dir}")
            cs.print(f"debug SAM tracker videos: {self.sam_tracker_vis_dir}")
        else:
            cs.print("debug outputs: disabled (use --debug to enable)")
        cs.rule()

    def _object_nodes(self, taskstructure: TaskStructure):
        nodes = []
        for subtask in taskstructure.subtask_list:
            for node in subtask.node_list or []:
                if node.role != NodeRole.ACTOR:
                    nodes.append(node)
        return nodes

    def _save_sam_initialization_vis(
        self,
        frame_rgb: np.ndarray,
        nodes,
        sampled_points: Sequence[Sequence[Sequence[float]]],
        episode_index: int,
        task_index: int,
    ):
        self.sam_initialization_vis_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
        colors = [
            (60, 60, 255),
            (255, 144, 30),
            (50, 205, 50),
            (0, 215, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        for node_index, (node, points) in enumerate(zip(nodes, sampled_points)):
            color = colors[node_index % len(colors)]
            for point in points:
                x, y = np.round(point).astype(int)
                cv2.circle(image, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(image, (int(x), int(y)), 7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            if len(points):
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
        save_path = self.sam_initialization_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.png"
        cv2.imwrite(os.fspath(save_path), image)

    def _save_sam_tracker_vis_video(
        self,
        frames: Sequence[np.ndarray],
        sam_results: Sequence[dict],
        episode_index: int,
        task_index: int,
        subtask_node_masks: np.ndarray | None = None,
    ):
        if not frames:
            return
        self.sam_tracker_vis_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.sam_tracker_vis_dir / f"episode_{int(episode_index):06d}_task_{int(task_index):03d}.mp4"
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
            for frame_index, (frame, result) in enumerate(zip(frames, sam_results)):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mask = None if subtask_node_masks is None else subtask_node_masks[frame_index]
                drawn = self._draw_sam_tracker_result(frame_bgr, result, mask)
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
    def _draw_sam_tracker_result(
        image_bgr: np.ndarray,
        result: Mapping[str, np.ndarray],
        subtask_node_mask: np.ndarray | None = None,
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
        if subtask_node_mask is None:
            subtask_node_mask = np.ones(points.shape[0], dtype=bool)
        else:
            subtask_node_mask = np.asarray(subtask_node_mask, dtype=bool)[:points.shape[0]]

        active = np.zeros(points.shape[0], dtype=bool)
        active[:len(subtask_node_mask)] = subtask_node_mask[:points.shape[0]]
        draw_order = list(np.where(~active)[0]) + list(np.where(active)[0])
        for node_index in draw_order:
            color = colors[node_index % len(colors)] if active[node_index] else (145, 145, 145)
            pale_color = tuple(int(round(c * 0.35 + 255 * 0.65)) for c in color)
            for point, visible in zip(points[node_index], visibles[node_index]):
                x, y = np.round(point).astype(int)
                draw_color = color if visible else pale_color
                cv2.circle(drawn, (int(x), int(y)), 2, draw_color, -1, lineType=cv2.LINE_AA)
        return drawn


    def _ensure_node_segmenter(self):
        if self.node_segmenter is not None:
            return
        segmenter_cls = (
            NodeSegmenterSAM2 if self.config.sam_version == "sam2" else NodeSegmenter
        )
        self.node_segmenter = segmenter_cls(device=self.config.sam_device)

    def _preflight_node_initialization_caches(
        self, episode_indices: Sequence[int]
    ) -> None:
        missing = [
            self.node_initialization_cache_dir / f"episode_{episode_index:06d}.npz"
            for episode_index in episode_indices
            if not (
                self.node_initialization_cache_dir / f"episode_{episode_index:06d}.npz"
            ).is_file()
        ]
        if not missing:
            return

        preview = "\n".join(f"  - {path}" for path in missing[:10])
        if len(missing) > 10:
            preview += f"\n  - ... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing node initialization cache for {len(missing)} selected episode(s):\n"
            f"{preview}\n"
            f"Run `python src/dataset/pipeline_init.py initialize "
            f"--dataset-dir {self.dataset_dir}` first."
        )


    def _infer_image_key(self) -> str:
        image_keys = [
            key for key, spec in self.info["features"].items()
            if spec.get("dtype") in {"image", "video"}
        ]
        for key in ("image", "observation.images.image", "observation.images.rgb_static"):
            if key in image_keys:
                return key
        for key in image_keys:
            lowered = key.lower()
            if "wrist" not in lowered and "gripper" not in lowered:
                return key
        raise RuntimeError("No non-wrist image feature found in meta/info.json")

    def _episode_parquet_path(self, episode_index: int) -> Path:
        episode_chunk = int(episode_index) // self.chunk_size
        return self.dataset_dir / self.data_path_template.format(
            episode_chunk=episode_chunk,
            episode_index=int(episode_index),
        )


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
        if self.config.debug:
            cs.print("[yellow]debug dry-run: skip meta/info.json update[/yellow]")
            return
        features = dict(self.info.get("features", {}))
        output_features = {
            "node_points_xyz": {
                "dtype": "float32",
                "shape": [self.config.max_nodes, self.config.points_per_node, 3],
                "names": ["node", "point", "xyz"],
            },
            "valid_node_mask": {
                "dtype": "bool",
                "shape": [self.config.max_nodes],
                "names": ["node"],
            },
            "subtask_node_mask": {
                "dtype": "bool",
                "shape": [self.config.max_nodes],
                "names": ["node"],
            },
            "gripper_points_xyz": {
                "dtype": "float32",
                "shape": [6, 3],
                "names": ["point", "xyz"],
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
    parser.add_argument("--sam-version", choices=("sam2", "sam3"), default="sam3")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--sam-devices", nargs="+", help="Devices used by worker processes.")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--debug-dir", default="/data0/luokang/research/GraphVLA/__tmp__/pipeline_sam")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")

    config = PipelineConfig(
        dataset_dir="/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz",
        dataset_type="libero",
        # episode_selector={
        #     0: ["*"],
        # },
        # episode_selector={
        #     0: [0],
        #     1: [0],
        #     2: [0],
        #     3: [0],
        #     4: [0]
        # },
        episode_selector = {},
        points_per_node=32,
        sam_version=args.sam_version,
        sam_device=args.sam_device,
        overwrite={
            "taskstructure": False,
            "node_points_xyz": True,
            "valid_node_mask": True,
            "subtask_node_mask": True,
            "gripper_points_xyz": False,
        },
        task_analyzer_api_key=api_key,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    devices = args.sam_devices or [args.sam_device]
    if args.workers_per_device <= 0:
        parser.error("--workers-per-device must be positive")
    worker_devices = [
        device
        for device in devices
        for _ in range(args.workers_per_device)
    ]
    if len(worker_devices) == 1:
        config.sam_device = worker_devices[0]
        OfflineSAMPipeline(config).run()
    else:
        coordinator = OfflineSAMPipeline(config)
        task_to_episodes = coordinator._scan_task_episodes()
        episode_indices = sorted(
            episode
            for episodes in task_to_episodes.values()
            for episode in episodes
        )
        coordinator._preflight_node_initialization_caches(episode_indices)
        coordinator._ensure_output_features()
        coordinator.build_taskstructures(sorted(task_to_episodes))
        shards = [episode_indices[index::len(worker_devices)] for index in range(len(worker_devices))]
        context = mp.get_context("spawn")
        processes = []
        for worker_id, (device, shard) in enumerate(zip(worker_devices, shards, strict=True)):
            worker_config = PipelineConfig(**{**config.__dict__, "sam_device": device})
            process = context.Process(
                target=_run_worker,
                args=(worker_config, shard, worker_id),
                name=f"sam-worker-{worker_id}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [process.name for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"SAM workers failed: {failed}")
        cs.print(
            f"all workers finished: workers={len(processes)}, episodes={len(episode_indices)}"
        )
