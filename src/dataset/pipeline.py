from __future__ import annotations

import argparse
import io
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
from src.common.schema import NodeRole, TaskStructure, json_to_taskstructure, taskstructure_to_json
from src.module.node_locator import NodeLocatorLA
from src.module.node_segmenter import NodeSegmenter
from src.module.point_tracker import PointTracker
from src.module.task_analyzer import TaskAnalyzer

cs = Console()


@dataclass
class PipelineConfig:
    dataset_dir: str | os.PathLike
    episode_selector: Mapping[str | int, Sequence[int | str]]
    dataset_type: str = "libero"
    task_analyzer_api_key: str | None = None
    points_per_node: int = 32
    max_nodes: int = 10
    erode_pixel: int = 2
    overwrite: Mapping[str, bool] | None = None
    debug: bool = False
    debug_dir: str | os.PathLike = "/data0/luokang/research/GraphVLA/__tmp__/pipeline"


class OfflinePipeline:
    """Precompute camera-XYZ entity points and gripper keypoints for LIBERO."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        self.meta_dir = self.dataset_dir / "meta"
        self.taskstructures_jsonl_path = self.meta_dir / "taskstructures.jsonl"
        self.tasks_jsonl_path = self.meta_dir / "tasks.jsonl"
        self.debug_dir = Path(config.debug_dir)
        self.node_locator_vis_dir = self.debug_dir / "node_locator"
        self.point_tracker_vis_dir = self.debug_dir / "point_tracker"
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
        self.node_locator = None
        self.node_segmenter = None
        self.point_tracker = None
        self.libero_cameras = None
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

        need_node_xyz = (
            self.overwrite["node_points_xyz"]
            or "node_points_xyz" not in df.columns
            or self.overwrite["valid_node_mask"]
            or "valid_node_mask" not in df.columns
        )
        need_valid_node_mask = need_node_xyz
        need_subtask_node_mask = (
            self.overwrite["subtask_node_mask"]
            or "subtask_node_mask" not in df.columns
        )
        need_gripper_points_xyz = self.overwrite["gripper_points_xyz"] or "gripper_points_xyz" not in df.columns
        if "subtask_id" not in df.columns:
            raise KeyError(f"subtask_id is required in {parquet_path}")
        if (
            not need_node_xyz
            and not need_valid_node_mask
            and not need_subtask_node_mask
            and not need_gripper_points_xyz
        ):
            return

        subtask_node_mask = self._build_subtask_node_mask(df, taskstructures[task_index])
        if need_node_xyz:
            frames = self._read_parquet_frames(df)
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
            node_xyz, valid_node_mask = self._build_node_points_xyz(
                tracks, self._read_metric_depths(df), intrinsic,
            )
            df["node_points_xyz"] = [points.tolist() for points in node_xyz]
        if need_valid_node_mask:
            df["valid_node_mask"] = [mask.tolist() for mask in valid_node_mask]
        if need_subtask_node_mask:
            df["subtask_node_mask"] = [mask.tolist() for mask in subtask_node_mask]

        if need_gripper_points_xyz:
            df["gripper_points_xyz"] = self._build_gripper_points_xyz(df, task_index)

        if self.config.debug:
            cs.print(f"[yellow]debug dry-run: skip writing parquet {parquet_path}[/yellow]")
            return
        self._write_episode_parquet(df, parquet_path)

    @staticmethod
    def _write_episode_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
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
                subtask_node_masks=subtask_node_masks,
            )
        tracks = np.stack(frame_tracks, axis=0).astype(np.float32)
        padded_tracks = np.zeros(
            (len(frames), self.config.max_nodes, self.config.points_per_node, 3),
            dtype=np.float32,
        )
        padded_tracks[:, :tracks.shape[1]] = tracks
        return padded_tracks

    def _build_node_points_xyz(
        self,
        tracks: np.ndarray,
        metric_depths: np.ndarray,
        intrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backproject every tracked task node and return per-frame XYZ validity."""
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
        valid_nodes = np.zeros(tracks.shape[:2], dtype=bool)
        for frame_index in range(tracks.shape[0]):
            for node_index in range(tracks.shape[1]):
                uv = tracks[frame_index, node_index, :, :2]
                u = np.rint(uv[:, 0]).astype(np.int64)
                v = np.rint(uv[:, 1]).astype(np.int64)
                valid = (
                    (tracks[frame_index, node_index, :, 2] > 0.5)
                    & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                )
                safe_u = np.clip(u, 0, width - 1)
                safe_v = np.clip(v, 0, height - 1)
                z = metric_depths[frame_index, safe_v, safe_u]
                valid &= np.isfinite(z) & (z > 0.0)
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size == 0:
                    continue
                fill_indices = np.resize(valid_indices, tracks.shape[2])
                selected_u = uv[fill_indices, 0]
                selected_v = uv[fill_indices, 1]
                selected_z = z[fill_indices]
                output[frame_index, node_index] = np.stack([
                    (selected_u - cx) / fx * selected_z,
                    (selected_v - cy) / fy * selected_z,
                    selected_z,
                ], axis=-1)
                valid_nodes[frame_index, node_index] = True
        return output, valid_nodes

    def _read_parquet_frames(self, df: pd.DataFrame) -> list[np.ndarray]:
        frames = []
        for item in df[self.image_key]:
            payload = item.get("bytes") if isinstance(item, Mapping) else None
            if payload is None:
                raise ValueError(f"{self.image_key} must contain embedded image bytes")
            with Image.open(io.BytesIO(payload)) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.append(np.ascontiguousarray(np.fliplr(frame)))
        return frames

    @staticmethod
    def _read_metric_depths(df: pd.DataFrame) -> np.ndarray:
        field = "agentview_real_depth_images"
        if field not in df:
            raise KeyError(f"Dataset is missing required metric depth field: {field}")
        flat = np.asarray(df[field].tolist(), dtype=np.float32)
        if flat.ndim != 2:
            raise ValueError(f"Expected flattened metric depth [T,HW], got {flat.shape}")
        side = int(round(np.sqrt(flat.shape[1])))
        if side * side != flat.shape[1]:
            raise ValueError(f"Metric depth size {flat.shape[1]} is not a square image")
        return np.ascontiguousarray(np.flip(flat.reshape(-1, side, side), axis=2))

    def _build_gripper_points_xyz(self, df: pd.DataFrame, task_index: int) -> list[list[list[float]]]:
        camera = self._load_libero_cameras()[int(task_index)]["agentview"]
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
        geometry = self._ensure_gripper_geometry()
        results = []
        for raw_state in df["state"]:
            state = np.asarray(raw_state, dtype=np.float64)
            if state.size < 8:
                raise ValueError("state must contain at least 8 values to build gripper_points_xyz")
            xyz = geometry.project_gripper_to_xyz(tcp_state=state[:6], extrinsic=extrinsic)
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
                "parquet fields: node_points_xyz, valid_node_mask, subtask_node_mask, "
                "gripper_points_xyz"
            )
        if self.config.debug:
            cs.print(f"debug node locator images: {self.node_locator_vis_dir}")
            cs.print(f"debug point tracker videos: {self.point_tracker_vis_dir}")
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

    def _locate_node_points(self, frame_rgb: np.ndarray, node_name: str) -> list[list[float]]:
        result = self.node_locator.inference(
            text=node_name,
            image=Image.fromarray(frame_rgb),
        )
        points = result.get("points") or []
        if not points:
            raise RuntimeError(f"NodeLocatorLA found no points for node={node_name!r}")
        return self._locator_points_to_pixels(points, frame_rgb.shape[:2])

    def _locator_points_to_pixels(self, points, image_size: tuple[int, int]) -> list[list[float]]:
        height, width = image_size
        converted = []
        for x, y in points:
            x = float(x) / 1000.0 * width
            y = float(y) / 1000.0 * height
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
        subtask_node_masks: np.ndarray | None = None,
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
                mask = None if subtask_node_masks is None else subtask_node_masks[frame_index]
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


    def _ensure_node_modules(self):
        if self.node_locator is None:
            self.node_locator = NodeLocatorLA()
        if self.node_segmenter is None:
            self.node_segmenter = NodeSegmenter()
        if self.point_tracker is None:
            self.point_tracker = PointTracker()


    def _infer_image_key(self) -> str:
        image_keys = [
            key for key, spec in self.info["features"].items()
            if spec.get("dtype") == "image"
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
                "shape": [3, 3],
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
    parser.add_argument("--debug-dir", default="/data0/luokang/research/GraphVLA/__tmp__/pipeline")
    args = parser.parse_args()
    api_key = "sk-UZpG2yYwDE5itw7s57eIJA"

    config = PipelineConfig(
        dataset_dir="/data0/luokang/research/GraphVLA/examples/libero/extra/libero_with_depth_7",
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
            "node_points_xyz": True,
            "valid_node_mask": True,
            "subtask_node_mask": True,
            "gripper_points_xyz": False,
        },
        task_analyzer_api_key=api_key,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    OfflinePipeline(config).run()
