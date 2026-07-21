#!/usr/bin/env python3
"""Run the observation-driven GraphVLA server on offline LIBERO samples."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import websockets

from src.common.schema import dataset_gripper_action_to_libero


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output_offline"
ACTOR_NAMES = ("root", "left", "right", "tcp")
ACTOR_UVD_NAMES = ("root_uvd", "left_base_uvd", "right_base_uvd", "tcp_uvd")
ACTOR_GT_INDICES = (0, 1, 2, 5)


def _images_to_server_rgb(images: torch.Tensor | np.ndarray) -> np.ndarray:
    array = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    if array.ndim != 4:
        raise ValueError(f"images must have shape TxCxHxW or TxHxWxC, got {array.shape}")
    if array.shape[1] in {1, 3, 4}:
        array = np.moveaxis(array, 1, -1)
    if array.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"images must have 1, 3, or 4 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(np.rint(array).astype(np.uint8)[:, :, ::-1])


def build_observation_request(
    *,
    images: torch.Tensor | np.ndarray,
    states: torch.Tensor | np.ndarray,
    prompt: str,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    history_horizon: int,
    execute_chunk_len: int,
    session_id: str,
    observation_count: int | None = None,
) -> dict[str, Any]:
    request_count = history_horizon + 1 if observation_count is None else observation_count
    if request_count < 1:
        raise ValueError(f"observation_count must be positive, got {request_count}")
    server_images = _images_to_server_rgb(images)[:request_count]
    state_array = states.detach().cpu().numpy() if isinstance(states, torch.Tensor) else np.asarray(states)
    state_array = state_array[:request_count]
    if len(server_images) != request_count or len(state_array) != request_count:
        raise ValueError(
            f"offline request must contain {request_count} observation frames, "
            f"got images={len(server_images)} states={len(state_array)}"
        )
    intrinsic_list = np.asarray(intrinsic, dtype=np.float64).tolist()
    extrinsic_list = np.asarray(extrinsic, dtype=np.float64).tolist()
    return {
        "benchmark": "libero",
        "session_id": session_id,
        "language": prompt,
        "execute_chunk_len": execute_chunk_len,
        "observation.images.image": server_images.tolist(),
        "observation.state": state_array.tolist(),
        "camera.intrinsics": [intrinsic_list] * request_count,
        "camera.extrinsics": [extrinsic_list] * request_count,
        "reset": True,
    }


def _first_batch(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return array[0] if array.ndim >= 2 and array.shape[0] == 1 else array


def _actor_uvd_for_frame(
    points: np.ndarray,
    metric_depth: np.ndarray,
    object_id: np.ndarray,
    point_id: np.ndarray,
    frame_id: np.ndarray,
    target_frame_id: int,
) -> dict[str, np.ndarray] | None:
    mask = (object_id == 0) & (frame_id == target_frame_id)
    by_id = {int(pid): point for pid, point in zip(point_id[mask], points[mask], strict=True)}
    metric_by_id = {
        int(pid): depth
        for pid, depth in zip(point_id[mask], metric_depth[mask], strict=True)
    }
    if not all(index in by_id for index in range(len(ACTOR_NAMES))):
        return None
    return {
        name: np.asarray([by_id[index][0], by_id[index][1], metric_by_id[index]])
        for index, name in enumerate(ACTOR_UVD_NAMES)
    }


def _json_uvd(uvd: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        output_name: np.asarray(uvd[input_name], dtype=np.float64).round(6).tolist()
        for output_name, input_name in zip(ACTOR_NAMES, ACTOR_UVD_NAMES, strict=True)
    }


def build_actor_diagnostics(
    *,
    response: dict[str, Any],
    gt_gripper_uvd: torch.Tensor | np.ndarray,
    gt_gripper_openness: torch.Tensor | np.ndarray,
    gt_action: torch.Tensor | np.ndarray,
    history_horizon: int,
) -> list[dict[str, Any]]:
    points = _first_batch(response["point"]).astype(np.float64)
    metric_depth = _first_batch(response["metric_depth"]).astype(np.float64).reshape(-1)
    object_id = _first_batch(response["object_id"]).astype(np.int64)
    point_id = _first_batch(response["point_id"]).astype(np.int64)
    frame_id = _first_batch(response["frame_id"]).astype(np.int64)
    actor_query_frame_id = _first_batch(response["actor_query_frame_id"]).astype(np.int64)
    predicted_openness = _first_batch(response["gripper_openness"]).astype(np.float64).reshape(-1)
    predicted_action = _first_batch(response["gripper_action"]).astype(np.float64).reshape(-1)
    gt_uvd = np.asarray(gt_gripper_uvd)
    gt_openness = np.asarray(gt_gripper_openness).reshape(-1)
    dataset_action = np.asarray(gt_action)[..., -1].reshape(-1)
    libero_action = dataset_gripper_action_to_libero(dataset_action)
    executed_actions = np.asarray(response.get("action", []), dtype=np.float64)

    rows = []
    for future_frame_id in sorted(int(value) for value in np.unique(frame_id) if value > 0):
        pred_uvd = _actor_uvd_for_frame(points, metric_depth, object_id, point_id, frame_id, future_frame_id)
        gt_index = future_frame_id + history_horizon
        query_indices = np.flatnonzero(actor_query_frame_id == future_frame_id)
        if pred_uvd is None or not 0 <= gt_index < len(gt_uvd) or len(query_indices) != 1:
            continue
        query_index = int(query_indices[0])
        action_gt_index = gt_index
        if not 0 <= action_gt_index < len(libero_action):
            continue
        action_index = future_frame_id - 1
        executed_action = float(executed_actions[action_index, 6]) if action_index < len(executed_actions) else None
        pred_open = float(predicted_openness[query_index])
        target_open = float(gt_openness[gt_index])
        pred_action = float(predicted_action[query_index])
        target_action = float(libero_action[action_gt_index])
        frame_gt_uvd = {
            name: np.asarray(gt_uvd[gt_index, source_index], dtype=np.float64)
            for name, source_index in zip(ACTOR_UVD_NAMES, ACTOR_GT_INDICES, strict=True)
        }
        rows.append({
            "frame_id": future_frame_id,
            "predicted_openness": round(pred_open, 6),
            "target_openness": round(target_open, 6),
            "openness_abs_error": round(abs(pred_open - target_open), 6),
            "predicted_action": round(pred_action, 6),
            "target_action": round(target_action, 6),
            "action_abs_error": round(abs(pred_action - target_action), 6),
            "executed_action": executed_action,
            "pred_uvd": _json_uvd(pred_uvd),
            "gt_uvd": _json_uvd(frame_gt_uvd),
        })
    return rows


async def _websocket_json_async(uri: str, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    async with websockets.connect(uri, open_timeout=timeout, max_size=None, proxy=None) as websocket:
        await asyncio.wait_for(websocket.send(json.dumps(request)), timeout=timeout)
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    return json.loads(message)


def websocket_json(uri: str, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    return asyncio.run(_websocket_json_async(uri, request, timeout))


def save_model_input_capture(output_path: Path, model_input: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"arrays": {}, "non_array": {}}
    for key, value in model_input.items():
        if value is None or isinstance(value, str | dict):
            metadata["non_array"][key] = value
            continue
        array = np.asarray(value)
        if array.dtype == object:
            metadata["non_array"][key] = value
            continue
        arrays[key] = array
        metadata["arrays"][key] = {"shape": list(array.shape), "dtype": str(array.dtype)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(np.asarray(value).reshape(-1)[0])


def load_camera(dataset_dir: Path, task_index: int, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / "meta/cameras.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in records if int(item["task_index"]) == task_index)
    camera = record["cameras"][camera_name]
    return np.asarray(camera["intrinsic"], dtype=np.float64), np.asarray(camera["extrinsic"], dtype=np.float64)


def make_single_frame_dataset(dataset_dir: Path) -> Any:
    from src.dataset.dataset import make_lerobot_dataset

    return make_lerobot_dataset(dataset_dir, video_backend="pyav")


def load_episode_observation_prefix(
    dataset: Any,
    episode_indices: list[int],
    local_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= local_index < len(episode_indices):
        raise ValueError(
            f"local_index={local_index} is outside episode with {len(episode_indices)} samples"
        )
    rows = [dataset[index] for index in episode_indices[: local_index + 1]]
    images = np.stack([
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for value in (row["observation.images.image"] for row in rows)
    ])
    states = np.stack([
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for value in (row["observation.state"] for row in rows)
    ])
    return images, states


def make_offline_dataset(dataset_dir: Path, history_horizon: int, future_horizon: int) -> Any:
    from src.dataset.dataset import make_lerobot_dataset

    info = json.loads((dataset_dir / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    offsets = list(range(-history_horizon, future_horizon + 1))
    delta_timestamps = {
        key: [offset / fps for offset in offsets]
        for key in (
            "observation.images.image",
            "observation.state",
            "gripper_uvd",
            "gripper_openness",
            "action",
        )
    }
    return make_lerobot_dataset(
        dataset_dir,
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
    )


def _light_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(channel * 0.35 + 255 * 0.65)) for channel in color)


def draw_point_grid(
    *,
    output_path: Path,
    images: torch.Tensor | np.ndarray,
    points: np.ndarray,
    object_id: np.ndarray,
    point_frame_id: np.ndarray,
    frame_ids: list[int],
    history_horizon: int,
    label: str,
    point_id: np.ndarray | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> None:
    rgb_frames = _images_to_server_rgb(images)
    height, width = rgb_frames.shape[1:3]
    grid = np.full((4 * height, 4 * width, 3), 245, dtype=np.uint8)
    points = _first_batch(points).astype(np.float32)
    object_id = _first_batch(object_id).astype(np.int64)
    point_frame_id = _first_batch(point_frame_id).astype(np.int64)
    point_id = None if point_id is None else _first_batch(point_id).astype(np.int64)
    diagnostics_by_frame = {
        int(row["frame_id"]): row for row in (diagnostics or [])
    }
    colors = {
        0: (40, 80, 255),
        1: (255, 80, 40),
        2: (255, 210, 40),
    }
    gt_color = (40, 210, 40)

    for index, vis_frame_id in enumerate(frame_ids[:16]):
        tile_row, tile_col = divmod(index, 4)
        tile = grid[tile_row * height : (tile_row + 1) * height, tile_col * width : (tile_col + 1) * width]
        image_index = vis_frame_id + history_horizon
        if 0 <= image_index < len(rgb_frames):
            tile[:] = cv2.cvtColor(rgb_frames[image_index], cv2.COLOR_RGB2BGR)

        mask = point_frame_id == vis_frame_id
        frame_points = points[mask]
        frame_objects = object_id[mask]
        frame_point_ids = point_id[mask] if point_id is not None else None
        actor_xy: dict[int, tuple[int, int]] = {}
        count = 0
        for actor_layer in (False, True):
            for row_index, (point, obj_id) in enumerate(zip(frame_points, frame_objects, strict=True)):
                is_actor = int(obj_id) == 0
                if is_actor != actor_layer or point.shape[0] < 2 or not np.isfinite(point[:2]).all():
                    continue
                x, y = np.rint(point[:2]).astype(int)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                color = colors.get(int(obj_id), (180, 80, 180))
                if point.shape[0] > 3 and float(point[3]) <= 0.5:
                    color = _light_color(color)
                cv2.circle(tile, (x, y), 4 if is_actor else 2, color, -1, lineType=cv2.LINE_AA)
                if is_actor and frame_point_ids is not None:
                    actor_xy[int(frame_point_ids[row_index])] = (x, y)
                count += 1
        if 1 in actor_xy and 2 in actor_xy:
            cv2.line(tile, actor_xy[1], actor_xy[2], colors[0], 2, cv2.LINE_AA)

        diagnostic = diagnostics_by_frame.get(vis_frame_id)
        if diagnostic is not None:
            gt_xy = {
                name: tuple(np.rint(diagnostic["gt_uvd"][name][:2]).astype(int))
                for name in ACTOR_NAMES
            }
            for xy in gt_xy.values():
                if 0 <= xy[0] < width and 0 <= xy[1] < height:
                    cv2.circle(tile, xy, 6, gt_color, 2, lineType=cv2.LINE_AA)
            cv2.line(tile, gt_xy["left"], gt_xy["right"], gt_color, 1, cv2.LINE_AA)
            status = (
                f" open={diagnostic['predicted_openness']:.3f}/{diagnostic['target_openness']:.3f}"
                f" action={diagnostic['predicted_action']:.3f}/{diagnostic['target_action']:.1f}"
            )
        else:
            status = ""

        text = f"{label} f={vis_frame_id} n={count}{status}"
        cv2.putText(tile, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (210, 210, 210), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), grid):
        raise RuntimeError(f"failed to save visualization to {output_path}")


def run_sample(
    *,
    sample: dict[str, Any],
    global_index: int,
    local_index: int,
    dataset_dir: Path,
    tasks: dict[int, str],
    server_uri: str,
    args: argparse.Namespace,
    observation_dataset: Any | None,
    episode_indices: list[int],
) -> dict[str, Any]:
    task_index = _scalar_int(sample["task_index"])
    intrinsic, extrinsic = load_camera(dataset_dir, task_index, args.camera_name)
    if args.warmup_from_episode_start:
        if observation_dataset is None:
            raise ValueError("observation_dataset is required for episode warm-up")
        request_images, request_states = load_episode_observation_prefix(
            observation_dataset,
            episode_indices,
            local_index,
        )
        observation_start_frame = 0
        observation_count = local_index + 1
    else:
        request_images = sample["observation.images.image"]
        request_states = sample["observation.state"]
        observation_start_frame = max(0, local_index - args.history_horizon)
        observation_count = None
    request = build_observation_request(
        images=request_images,
        states=request_states,
        prompt=tasks[task_index],
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        history_horizon=args.history_horizon,
        observation_count=observation_count,
        execute_chunk_len=args.execute_chunk_len,
        session_id=f"offline-ep{args.episode_index}-sample{local_index}",
    )
    if args.model_input_output is not None:
        request["return_model_input"] = True
    response = websocket_json(server_uri, request, timeout=args.timeout)
    if "error" in response:
        raise RuntimeError(response["error"])
    required_response_fields = (
        "point",
        "metric_depth",
        "gripper_openness",
        "gripper_action",
        "actor_query_frame_id",
        "object_id",
        "point_id",
        "frame_id",
        "input_point",
        "input_object_id",
        "input_point_id",
        "input_frame_id",
        "action",
    )
    missing = [key for key in required_response_fields if key not in response]
    if missing:
        raise KeyError(f"server response missing fields: {missing}")
    if args.model_input_output is not None:
        if "model_input" not in response:
            raise KeyError("server response missing model_input for capture request")
        capture_path = args.model_input_output
        if args.num_samples > 1:
            capture_path = capture_path.with_name(
                f"{capture_path.stem}_sample_{local_index:04d}{capture_path.suffix}"
            )
        save_model_input_capture(capture_path, response["model_input"])

    diagnostics = build_actor_diagnostics(
        response=response,
        gt_gripper_uvd=sample["gripper_uvd"],
        gt_gripper_openness=sample["gripper_openness"],
        gt_action=sample["action"],
        history_horizon=args.history_horizon,
    )
    stem = f"episode_{args.episode_index:06d}_sample_{local_index:04d}"
    prediction_path = args.output_dir / f"{stem}_prediction.png"
    tracking_path = args.output_dir / f"{stem}_tracking.png"
    json_path = args.output_dir / f"{stem}.json"
    draw_point_grid(
        output_path=prediction_path,
        images=sample["observation.images.image"],
        points=response["point"],
        object_id=response["object_id"],
        point_id=response["point_id"],
        point_frame_id=response["frame_id"],
        frame_ids=list(range(1, args.future_horizon + 1)),
        history_horizon=args.history_horizon,
        label="prediction",
    )
    draw_point_grid(
        output_path=tracking_path,
        images=sample["observation.images.image"],
        points=response["input_point"],
        object_id=response["input_object_id"],
        point_id=response["input_point_id"],
        point_frame_id=response["input_frame_id"],
        frame_ids=list(range(-args.history_horizon, 1)),
        history_horizon=args.history_horizon,
        label="tracking",
    )
    result = {
        "global_index": global_index,
        "episode_index": args.episode_index,
        "sample_index": local_index,
        "task_index": task_index,
        "prompt": tasks[task_index],
        "subtask": response.get("subtask"),
        "subtask_index": response.get("subtask_index"),
        "warmup_from_episode_start": args.warmup_from_episode_start,
        "server_anchor_frame": observation_start_frame,
        "server_observation_frame_range": [observation_start_frame, local_index],
        "model_history_frame_range": [
            max(0, local_index - args.history_horizon),
            local_index,
        ],
        "diagnostics": diagnostics,
        "visualizations": {
            "prediction": str(prediction_path),
            "tracking": str(tracking_path),
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    mean_action_error = (
        float(np.mean([row["action_abs_error"] for row in diagnostics]))
        if diagnostics
        else float("nan")
    )
    print(
        f"sample={local_index} global_index={global_index} frames={len(diagnostics)} "
        f"mean_action_error={mean_action_error:.6f} output={json_path}"
    )
    return result


def parse_args() -> argparse.Namespace:
    from examples.libero.config.data_config import LIBERO_DATASET_DIR, LIBERO_FUTURE_HORIZON, LIBERO_HISTORY_HORIZON

    parser = argparse.ArgumentParser(
        description="Send offline LIBERO training samples through the observation-driven GraphVLA server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset-dir", type=Path, default=LIBERO_DATASET_DIR)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0, help="Frame-local index inside the selected episode.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument(
        "--warmup-from-episode-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Warm up server temporal state with every episode frame from frame 0 to the "
            "selected sample; disable to send only the fixed history window."
        ),
    )
    parser.add_argument("--history-horizon", type=int, default=LIBERO_HISTORY_HORIZON)
    parser.add_argument("--future-horizon", type=int, default=LIBERO_FUTURE_HORIZON)
    parser.add_argument("--execute-chunk-len", type=int, default=LIBERO_FUTURE_HORIZON)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-input-output",
        type=Path,
        default=None,
        help="Optional .npz path for the exact tensor inputs entering PointQueryModel.infer.",
    )
    args = parser.parse_args()
    if args.num_samples < 1 or args.sample_stride < 1:
        parser.error("--num-samples and --sample-stride must be positive")
    if not 1 <= args.execute_chunk_len <= args.future_horizon:
        parser.error("--execute-chunk-len must be in [1, --future-horizon]")
    return args


def main() -> None:
    from examples.libero.config.data_config import load_lerobot_tasks

    args = parse_args()
    dataset = make_offline_dataset(args.dataset_dir, args.history_horizon, args.future_horizon)
    observation_dataset = (
        make_single_frame_dataset(args.dataset_dir)
        if args.warmup_from_episode_start
        else None
    )
    episode_values = dataset.hf_dataset["episode_index"]
    episode_indices = [index for index, value in enumerate(episode_values) if int(value) == args.episode_index]
    if not episode_indices:
        raise ValueError(f"episode_index={args.episode_index} not found")
    local_indices = list(
        range(args.sample_index, len(episode_indices), args.sample_stride)
    )[: args.num_samples]
    if not local_indices:
        raise ValueError(
            f"sample_index={args.sample_index} is outside episode with {len(episode_indices)} samples"
        )

    tasks = load_lerobot_tasks(args.dataset_dir)
    server_uri = f"ws://{args.host}:{args.port}"
    print(f"server_uri={server_uri} episode_index={args.episode_index} sample_indices={local_indices}")
    for local_index in local_indices:
        global_index = episode_indices[local_index]
        run_sample(
            sample=dataset[global_index],
            global_index=global_index,
            local_index=local_index,
            dataset_dir=args.dataset_dir,
            tasks=tasks,
            server_uri=server_uri,
            args=args,
            observation_dataset=observation_dataset,
            episode_indices=episode_indices,
        )


if __name__ == "__main__":
    main()
