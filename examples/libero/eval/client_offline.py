#!/usr/bin/env python3
"""Evaluate one dataset frame after warming up the online LIBERO server."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import runpy
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets
from rich.console import Console

from examples.libero.config.data_config import LIBERO_DATASET_DIR
from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
from src.dataset.dataset import make_lerobot_dataset
from src.common.schema import ACTION_DIM, ACTOR_NUM_POINTS, POINT_FEATURE_DIM


cs = Console()
LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_offline"
GRID_SIZE = 4
GRID_CAPACITY = GRID_SIZE * GRID_SIZE


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _to_rgb_uint8(value: Any, *, horizontal_flip: bool = False) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D RGB image, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB channels, got shape={image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    if horizontal_flip:
        image = image[:, ::-1]
    return np.ascontiguousarray(image, dtype=np.uint8)


def _scalar(value: Any) -> int:
    return int(_to_numpy(value).item())


def _normalize_language(value: str) -> str:
    return " ".join(value.lower().split())


def _suite_task_languages(task_suite_name: str) -> list[str]:
    map_path = LIBERO_ROOT / "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = runpy.run_path(str(map_path))["libero_task_map"]
    if task_suite_name not in task_map:
        raise ValueError(
            f"unknown task suite {task_suite_name!r}; available suites: {sorted(task_map)}"
        )
    bddl_dir = LIBERO_ROOT / "libero/libero/bddl_files" / task_suite_name
    languages = []
    for task_name in task_map[task_suite_name]:
        bddl_path = bddl_dir / f"{task_name}.bddl"
        match = re.search(r"\(:language\s+(.+?)\)", bddl_path.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"missing :language declaration in {bddl_path}")
        languages.append(match.group(1).strip())
    return languages


def _select_dataset_episodes(
    dataset_dir: Path,
    *,
    task_suite_name: str,
    task_ids: list[int] | None,
    episode_ids: list[int],
) -> list[tuple[int, int, int]]:
    suite_languages = _suite_task_languages(task_suite_name)
    with (dataset_dir / "meta/tasks.jsonl").open("r", encoding="utf-8") as file:
        dataset_tasks = [json.loads(line) for line in file if line.strip()]
    with (dataset_dir / "meta/episodes.jsonl").open("r", encoding="utf-8") as file:
        dataset_episodes = [json.loads(line) for line in file if line.strip()]
    available = {_normalize_language(str(row["task"])) for row in dataset_tasks}

    if task_ids is None:
        task_ids = [
            index
            for index, language in enumerate(suite_languages)
            if _normalize_language(language) in available
        ]
        if not task_ids:
            raise ValueError(f"dataset contains no tasks from suite {task_suite_name!r}")

    selected = []
    for task_id in task_ids:
        if task_id < 0 or task_id >= len(suite_languages):
            raise ValueError(
                f"task id {task_id} out of range for {task_suite_name}: "
                f"0-{len(suite_languages) - 1}"
            )
        language = suite_languages[task_id]
        normalized = _normalize_language(language)
        if normalized not in available:
            raise ValueError(
                f"task {task_suite_name}[{task_id}] is not present in dataset: {language}"
            )
        dataset_episode_ids = [
            int(row["episode_index"])
            for row in dataset_episodes
            if normalized in {
                _normalize_language(str(item)) for item in row.get("tasks", [])
            }
        ]
        for episode_id in episode_ids:
            if episode_id < 0 or episode_id >= len(dataset_episode_ids):
                raise ValueError(
                    f"episode {episode_id} out of range for {task_suite_name}[{task_id}]: "
                    f"0-{len(dataset_episode_ids) - 1}"
                )
            selected.append((task_id, episode_id, dataset_episode_ids[episode_id]))
    return selected


def _load_cameras(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    cameras = {}
    for row in rows:
        task_index = int(row["task_index"])
        agentview = row["cameras"]["agentview"]
        intrinsic = np.asarray(agentview["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(agentview["extrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
            raise ValueError(
                f"invalid agentview matrices for task {task_index}: "
                f"intrinsic={intrinsic.shape}, extrinsic={extrinsic.shape}"
            )
        cameras[task_index] = intrinsic, extrinsic
    return cameras


def _draw_label(image: np.ndarray, left: str, right: str | None) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, left, (8, 18), font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, left, (8, 18), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if right is None:
        return
    (width, _), _ = cv2.getTextSize(right, font, 0.48, 1)
    position = (max(8, image.shape[1] - 8 - width), 18)
    cv2.putText(image, right, position, font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, right, position, font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_response_points(
    image_rgb: np.ndarray,
    response: dict[str, Any],
    frame_id: int,
    *,
    mode: str,
    intrinsic: np.ndarray | None = None,
) -> np.ndarray:
    image = np.ascontiguousarray(image_rgb.copy())
    height, width = image.shape[:2]
    colors = {
        0: (255, 80, 40),
        1: (60, 60, 255),
        2: (255, 144, 30),
        3: (50, 205, 50),
        4: (0, 215, 255),
    }
    count = 0
    projected_prediction_points: list[tuple[int, int, tuple[int, int, int]]] = []
    if mode == "tracking":
        points = response.get("tracking_point")
        object_ids = response.get("tracking_object_id")
        if points is not None and object_ids is not None:
            points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            object_ids = np.asarray(object_ids, dtype=np.int64).reshape(-1)
            for point, object_id in zip(points, object_ids, strict=True):
                if not np.isfinite(point[:2]).all():
                    continue
                x, y = np.rint(point[:2]).astype(int)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(
                        image, (x, y), 2,
                        colors.get(int(object_id), (160, 80, 160)),
                        -1, lineType=cv2.LINE_AA,
                    )
                    count += 1
        label = f"tracking frame={frame_id} in={count}"
    elif mode == "prediction":
        action_plan = response.get("action_plan")
        if action_plan is not None:
            plan = np.asarray(action_plan, dtype=np.float32)
            plan = plan[0] if plan.ndim == 3 else plan
            future_index = frame_id - 1
            if plan.ndim == 2 and plan.shape[1] == ACTION_DIM and 0 <= future_index < len(plan):
                trajectory = plan[future_index]
                points = trajectory[:-1].reshape(ACTOR_NUM_POINTS, POINT_FEATURE_DIM)
                if intrinsic is not None:
                    camera_matrix = np.asarray(intrinsic, dtype=np.float32)
                    if camera_matrix.shape != (3, 3):
                        raise ValueError(
                            f"intrinsic must have shape (3,3), got {camera_matrix.shape}"
                        )
                    for point, color in zip(points, colors.values(), strict=True):
                        if not np.isfinite(point).all() or point[2] <= 1e-6:
                            continue
                        x = int(round(float(
                            point[0] / point[2] * camera_matrix[0, 0] + camera_matrix[0, 2]
                        )))
                        y = int(round(float(
                            point[1] / point[2] * camera_matrix[1, 1] + camera_matrix[1, 2]
                        )))
                        if 0 <= x < width and 0 <= y < height:
                            projected_prediction_points.append((x, y, color))
                            count += 1
                label = f"prediction points future={frame_id} in={count} grip={trajectory[-1]:.2f}"
            else:
                label = f"prediction points future={frame_id}"
        else:
            label = f"prediction points future={frame_id}"
    else:
        raise ValueError(f"unknown visualization mode: {mode}")

    right_label = None
    if mode == "tracking":
        completion = response.get("is_complete")
        if completion is None:
            right_label = "-"
        else:
            scores = np.asarray(completion, dtype=np.float32).reshape(-1)
            scores = scores[np.isfinite(scores)]
            right_label = "-" if scores.size == 0 else f"{scores.max():.2f}"
    _draw_label(image, label, right_label)
    for x, y, color in projected_prediction_points:
        cv2.circle(image, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
    return image


def _save_grid(frames_rgb: list[np.ndarray], save_path: Path) -> None:
    if not frames_rgb or len(frames_rgb) > GRID_CAPACITY:
        raise ValueError(f"grid expects 1-{GRID_CAPACITY} frames, got {len(frames_rgb)}")
    height, width = frames_rgb[0].shape[:2]
    grid = np.zeros((GRID_SIZE * height, GRID_SIZE * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames_rgb):
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != (height, width, 3):
            raise ValueError(f"grid frame shape mismatch: {frame.shape}")
        row, column = divmod(index, GRID_SIZE)
        grid[row * height:(row + 1) * height, column * width:(column + 1) * width] = frame
    if not cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save grid to {save_path}")
    cs.print(f"saved grid to: {save_path}")


async def _evaluate_samples(
    *,
    uri: str,
    dataset_dir: Path,
    cameras: dict[int, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    task_suite_name: str,
    suite_task_id: int,
    episode_id: int,
    dataset_episode_id: int,
    sample_indices: list[int],
    horizontal_flip: bool,
) -> list[dict[str, Any]]:
    dataset = make_lerobot_dataset(
        dataset_dir, load_videos=True, video_backend="pyav",
        episodes=[dataset_episode_id],
    )
    sample_indices = sorted(set(sample_indices))
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(dataset):
            raise ValueError(
                f"sample {sample_index} out of range for episode {episode_id}: "
                f"0-{len(dataset) - 1}"
            )
    max_sample = sample_indices[-1]
    target_samples = set(sample_indices)
    history_length = int(LIBERO_MODEL_CONFIG.history_horizon) + 1
    future_horizon = int(LIBERO_MODEL_CONFIG.future_horizon)

    session_id = (
        f"libero-dataset-{suite_task_id:03d}-{dataset_episode_id:06d}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    history_frames = []
    records = []
    async with websockets.connect(uri, max_size=None, proxy=None) as websocket:
        for frame_index in range(max_sample + 1):
            frame_sample = dataset[frame_index]
            task_index = _scalar(frame_sample["task_index"])
            if task_index not in cameras:
                raise KeyError(f"cameras.json has no entry for task_index={task_index}")
            intrinsic, extrinsic = cameras[task_index]
            image = _to_rgb_uint8(
                frame_sample["image"],
                horizontal_flip=horizontal_flip,
            )
            state = _to_numpy(frame_sample["state"]).astype(np.float64)
            metric_depth = _to_numpy(
                frame_sample["agentview_real_depth_images"]
            ).astype(np.float32).reshape(256, 256)
            request = {
                "benchmark": "libero",
                "session_id": session_id,
                "language": str(frame_sample["task"]),
                "observation.images.image": [image.tolist()],
                "observation.depth.metric": [metric_depth.tolist()],
                "observation.state": [state.tolist()],
                "camera.intrinsics": [intrinsic.tolist()],
                "camera.extrinsics": [extrinsic.tolist()],
            }
            if frame_index == 0:
                request["reset"] = True
            await websocket.send(json.dumps(request))
            response = json.loads(await websocket.recv())
            if "error" in response:
                raise RuntimeError(
                    f"server failed at task={suite_task_id} episode={episode_id} "
                    f"frame={frame_index}: {response['error']}"
                )

            history_frames.append(
                _draw_response_points(image, response, frame_index, mode="tracking")
            )
            history_frames = history_frames[-history_length:]
            cs.print(
                f"task={suite_task_id} episode={episode_id} "
                f"warmup={frame_index}/{max_sample}"
            )
            if frame_index not in target_samples:
                continue

            sample_history = (
                [history_frames[0]] * (history_length - len(history_frames))
                + history_frames
            )
            future_frames = []
            for future_id in range(1, future_horizon + 1):
                future_sample = dataset[min(frame_index + future_id, len(dataset) - 1)]
                future_frames.append(
                    _draw_response_points(
                        _to_rgb_uint8(
                            future_sample["image"],
                            horizontal_flip=horizontal_flip,
                        ),
                        response,
                        future_id,
                        mode="prediction",
                        intrinsic=intrinsic,
                    )
                )

            stem = (
                f"{task_suite_name}_task_{suite_task_id:03d}_"
                f"episode_{episode_id:03d}_sample_{frame_index:04d}"
            )
            history_paths = []
            for page, page_start in enumerate(
                range(0, history_length, GRID_CAPACITY)
            ):
                grid_path = output_dir / f"{stem}_history_{page:02d}.png"
                _save_grid(
                    sample_history[page_start:page_start + GRID_CAPACITY],
                    grid_path,
                )
                history_paths.append(str(grid_path))
            future_path = output_dir / f"{stem}_future.png"
            _save_grid(future_frames, future_path)

            records.append({
                "task_suite_name": task_suite_name,
                "suite_task_id": suite_task_id,
                "episode_index": episode_id,
                "dataset_episode_index": dataset_episode_id,
                "sample": frame_index,
                "dataset_index": _scalar(frame_sample["index"]),
                "task": str(frame_sample["task"]),
                "gt_subtask": str(frame_sample["subtask"]),
                "gt_subtask_id": _scalar(frame_sample["subtask_id"]),
                "gt_is_complete": bool(
                    _to_numpy(frame_sample["is_complete"]).item()
                ),
                "gt_action": _to_numpy(frame_sample["action"]).tolist(),
                "history_grids": history_paths,
                "future_grid": str(future_path),
                "response": response,
            })
    return records

async def _run(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    camera_path = Path(args.cameras_path) if args.cameras_path else dataset_dir / "meta/cameras.json"
    cameras = _load_cameras(camera_path)
    selected = _select_dataset_episodes(
        dataset_dir,
        task_suite_name=args.task_suite_name,
        task_ids=args.tasks,
        episode_ids=args.episodes,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "responses.jsonl"
    uri = f"ws://{args.host}:{args.port}"

    total_samples = 0
    with result_path.open("w", encoding="utf-8") as output_file:
        for task_id, episode_id, dataset_episode_id in selected:
            records = await _evaluate_samples(
                uri=uri,
                dataset_dir=dataset_dir,
                cameras=cameras,
                output_dir=output_dir,
                task_suite_name=args.task_suite_name,
                suite_task_id=task_id,
                episode_id=episode_id,
                dataset_episode_id=dataset_episode_id,
                sample_indices=args.samples,
                horizontal_flip=args.horizontal_flip,
            )
            for record in records:
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()
            total_samples += len(records)
    cs.print(f"saved {total_samples} sample responses to: {result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up the online LIBERO server through a selected dataset sample, "
            "then save 4x4 history/future visualization grids."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset-dir", default=LIBERO_DATASET_DIR)
    parser.add_argument("--cameras-path", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--tasks", type=_parse_int_list, default=None)
    parser.add_argument("--episodes", type=_parse_int_list, default=[0])
    parser.add_argument(
        "--sample",
        dest="samples",
        type=_parse_int_list,
        required=True,
        help="Comma-separated episode-local frame indices.",
    )
    parser.set_defaults(horizontal_flip=False)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if any(sample < 0 for sample in args.samples):
        parser.error("--sample values must be non-negative")
    return args


def main() -> None:
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
