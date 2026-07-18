#!/usr/bin/env python3
"""Offline client for a minimal GraphVLA server round trip.

This script samples one transformed LIBERO dataset item, sends model-ready input
to scripts/server.py, receives server outputs, and visualizes returned UV points.
It does not depend on a LIBERO simulation environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import websockets

from examples.libero.config.data_config import LIBERO_DATA_CONFIG
from examples.libero.embodiment.robot import GeomFrankaPanda
from src.dataset.dataset import GenericDataset

LIBERO_CAMERA_DATASET_DIR = PROJECT_ROOT / "examples/libero/extra/libero_31_no_noops_1.0.0_lerobot_10hz"


def to_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_json_value(item) for item in value]
    if isinstance(value, list):
        return [to_json_value(item) for item in value]
    return value


def add_batch_dim(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    return value


def build_request(sample: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "point_feats",
        "actor_feats",
        "object_condition",
        "actor_condition",
        "object_id",
        "point_id",
        "frame_id",
        "frame_query_frame_id",
    )
    data = {key: to_json_value(add_batch_dim(sample[key])) for key in keys if key in sample}
    data["head_names"] = ["point", "metric_depth"]
    data["frame_head_names"] = "is_complete"
    return data


async def websocket_json_async(uri: str, data: dict[str, Any], timeout: float) -> dict[str, Any]:
    async with websockets.connect(uri, open_timeout=timeout, max_size=None, proxy=None) as websocket:
        await asyncio.wait_for(websocket.send(json.dumps(data)), timeout=timeout)
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    return json.loads(message)


def websocket_json(uri: str, data: dict[str, Any], timeout: float) -> dict[str, Any]:
    return asyncio.run(websocket_json_async(uri, data, timeout))


def image_to_bgr(image: torch.Tensor, *, width: int, height: int) -> np.ndarray:
    array = image.detach().cpu().float().clamp(0, 1).numpy()
    array = np.moveaxis(array, 0, -1)
    array = (array * 255.0).round().astype(np.uint8)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    array = cv2.flip(array, 1)
    if array.shape[:2] != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    return array


def draw_text(image: np.ndarray, text: str, xy: tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, xy, font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def light_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(channel * 0.35 + 255 * 0.65)) for channel in color)


def load_camera(dataset_root: Path, task_index: int, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    cameras = json.loads((dataset_root / "meta" / "cameras.json").read_text(encoding="utf-8"))
    record = next(item for item in cameras if int(item["task_index"]) == int(task_index))
    camera = record["cameras"][camera_name]
    return np.asarray(camera["intrinsic"], dtype=np.float64), np.asarray(camera["extrinsic"], dtype=np.float64)


def standard_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape[-1] >= 8:
        gripper = np.abs(state[..., 6:7]) + np.abs(state[..., 7:8])
        return np.concatenate([state[..., :6], gripper], axis=-1)
    return state[..., :7]


def draw_uv_grid(
    output_path: Path,
    points: np.ndarray,
    object_id: np.ndarray,
    point_frame_id: np.ndarray,
    complete: np.ndarray | None,
    complete_frame_id: np.ndarray | None,
    images: torch.Tensor | None,
    *,
    frame_ids: list[int],
    history_horizon: int,
    width: int,
    height: int,
    radius: int,
) -> int:
    rows, cols = 4, 4
    grid = np.full((rows * height, cols * width, 3), 245, dtype=np.uint8)
    colors = {0: (40, 80, 255), 1: (40, 180, 40), 2: (220, 80, 40)}
    total = 0

    complete_by_frame = {}
    if complete is not None and complete_frame_id is not None:
        flat_complete = complete.reshape(-1)
        flat_frame = complete_frame_id.reshape(-1)
        for fid, score in zip(flat_frame, flat_complete):
            complete_by_frame[int(fid)] = float(score)

    for index, vis_frame_id in enumerate(frame_ids[: rows * cols]):
        row, col = divmod(index, cols)
        tile = grid[row * height : (row + 1) * height, col * width : (col + 1) * width]
        image_index = vis_frame_id + history_horizon
        if images is not None and 0 <= image_index < images.shape[0]:
            tile[:] = image_to_bgr(images[image_index], width=width, height=height)
        else:
            tile[:] = 255
        mask = point_frame_id == vis_frame_id
        tile_points = points[mask]
        uv = tile_points[:, :2]
        vis = tile_points[:, 3] if tile_points.shape[1] > 3 else np.ones(len(tile_points), dtype=np.float32)
        obj = object_id[mask]
        count = 0

        for actor_layer in (False, True):
            for xy, obj_id, visibility in zip(uv, obj, vis):
                is_actor = int(obj_id) == 0
                if is_actor != actor_layer or not np.isfinite(xy).all():
                    continue
                x, y = np.rint(xy).astype(int)
                if 0 <= x < width and 0 <= y < height:
                    color = colors.get(int(obj_id), (160, 80, 160))
                    if visibility <= 0.5:
                        color = light_color(color)
                    cv2.circle(tile, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
                    count += 1
                    total += 1

        draw_text(tile, f"f={vis_frame_id} n={count}", (8, 18))
        if vis_frame_id in complete_by_frame:
            text = f"c={complete_by_frame[vis_frame_id]:.2f}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            draw_text(tile, text, (width - tw - 8, 18))
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (210, 210, 210), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GraphVLA server client for UV visualization.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10092)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/libero/test/output/offline_uv.png"),
    )
    return parser.parse_args()


def gripper_state_errors(
    *,
    robot: GeomFrankaPanda,
    points: np.ndarray,
    metric_depth: np.ndarray,
    object_id: np.ndarray,
    point_id: np.ndarray,
    frame_id: np.ndarray,
    state: torch.Tensor | None,
    frame_ids: list[int],
    history_horizon: int,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> dict[str, float] | None:
    if state is None:
        return None
    gt_states = standard_state(state.detach().cpu().numpy())
    pos_errors = []
    rot_errors = []
    gripper_errors = []
    for fid in frame_ids:
        image_index = fid + history_horizon
        if image_index < 0 or image_index >= gt_states.shape[0]:
            continue
        mask = (object_id == 0) & (frame_id == fid)
        actor_points = points[mask]
        actor_metric_depth = metric_depth[mask]
        actor_point_ids = point_id[mask]
        if actor_points.shape[0] < 3:
            continue
        by_id = {int(pid): actor_points[index] for index, pid in enumerate(actor_point_ids)}
        metric_by_id = {
            int(pid): actor_metric_depth[index] for index, pid in enumerate(actor_point_ids)
        }
        if not all(index in by_id for index in (0, 1, 2)):
            continue
        uvd = {
            "root_uvd": np.asarray([by_id[0][0], by_id[0][1], metric_by_id[0]]),
            "left_uvd": np.asarray([by_id[1][0], by_id[1][1], metric_by_id[1]]),
            "right_uvd": np.asarray([by_id[2][0], by_id[2][1], metric_by_id[2]]),
        }
        if not all(np.isfinite(value).all() and value[2] > 1e-6 for value in uvd.values()):
            continue
        pred = robot.project_uvd_to_gripper(uvd, intrinsic=intrinsic, extrinsic=extrinsic)
        gt = gt_states[image_index]
        pos_errors.append(float(np.linalg.norm(pred[:3] - gt[:3])))
        rot_errors.append(float(np.linalg.norm(pred[3:6] - gt[3:6])))
        gripper_errors.append(float(abs(pred[6] - gt[6])))
    if not pos_errors:
        return None
    return {
        "pos": float(np.mean(pos_errors)),
        "rot": float(np.mean(rot_errors)),
        "gripper": float(np.mean(gripper_errors)),
        "count": float(len(pos_errors)),
    }


def output_path_for_sample(output: Path, sample_index: int, num_samples: int) -> Path:
    if num_samples == 1:
        return output
    return output.with_name(f"{output.stem}_sample_{sample_index:04d}{output.suffix}")


def run_sample(
    *,
    sample: dict[str, Any],
    sample_index: int,
    server_uri: str,
    output_path: Path,
    timeout: float,
    width: int,
    height: int,
    radius: int,
    history_horizon: int,
    robot: GeomFrankaPanda,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> int:
    request_data = build_request(sample)
    response = websocket_json(server_uri, request_data, timeout=timeout)
    if "error" in response:
        raise RuntimeError(response["error"])
    outputs = response["outputs"]
    if "point" not in outputs or "metric_depth" not in outputs:
        raise KeyError("server response missing outputs['point'] or outputs['metric_depth']")

    points = np.asarray(outputs["point"], dtype=np.float32)[0]
    metric_depth = np.asarray(outputs["metric_depth"], dtype=np.float32)[0].reshape(-1)
    object_id = np.asarray(request_data["object_id"], dtype=np.int64)[0]
    frame_id = np.asarray(request_data["frame_id"], dtype=np.int64)[0]
    point_id = np.asarray(request_data["point_id"], dtype=np.int64)[0]
    complete = np.asarray(outputs["is_complete"], dtype=np.float32)[0] if "is_complete" in outputs else None
    complete_frame_id = np.asarray(request_data["frame_query_frame_id"], dtype=np.int64)[0] if complete is not None else None
    images = sample.get("images.image")
    past_frame_ids = np.rint(np.linspace(-15, 0, 8)).astype(np.int64).tolist()
    future_frame_ids = np.rint(np.linspace(1, 16, 8)).astype(np.int64).tolist()
    frame_ids = past_frame_ids + future_frame_ids
    errors = gripper_state_errors(
        robot=robot,
        points=points,
        metric_depth=metric_depth,
        object_id=object_id,
        point_id=point_id,
        frame_id=frame_id,
        state=sample.get("state"),
        frame_ids=frame_ids,
        history_horizon=history_horizon,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    count = draw_uv_grid(
        output_path,
        points,
        object_id,
        frame_id,
        complete,
        complete_frame_id,
        images,
        frame_ids=frame_ids,
        history_horizon=history_horizon,
        width=width,
        height=height,
        radius=radius,
    )
    err_text = "" if errors is None else (
        f" state_err_pos={errors['pos']:.4f} state_err_rot={errors['rot']:.4f} "
        f"state_err_gripper={errors['gripper']:.4f} state_err_count={int(errors['count'])}"
    )
    print(f"sample_index={sample_index} frame_ids={frame_ids} drawn_uv_points={count}{err_text} output={output_path}")
    return count


def main() -> None:
    args = parse_args()
    data_config = LIBERO_DATA_CONFIG
    dataset = GenericDataset(data_config)
    history_horizon = int(getattr(data_config, "transforms", ())[2].history_horizon)
    episode_values = dataset.dataset.hf_dataset["episode_index"]
    episode_indices = [index for index, value in enumerate(episode_values) if int(value) == args.episode_index]
    if not episode_indices:
        raise ValueError(f"episode_index={args.episode_index} not found")

    start_index = args.sample_index if args.sample_index is not None else 0
    local_indices = list(range(start_index, len(episode_indices), args.sample_stride))
    last_index = len(episode_indices) - 1
    if local_indices and local_indices[-1] != last_index:
        local_indices.append(last_index)
    if args.num_samples is not None:
        local_indices = local_indices[: args.num_samples]
    intrinsic, extrinsic = load_camera(LIBERO_CAMERA_DATASET_DIR, task_index=0, camera_name=args.camera_name)
    robot = GeomFrankaPanda()
    server_uri = f"ws://{args.host}:{args.port}"

    print(f"server_uri: {server_uri}")
    print(f"episode_index: {args.episode_index}")
    print(f"local_sample_indices: {local_indices}")
    for local_index in local_indices:
        global_index = episode_indices[local_index]
        sample = dataset[global_index]
        run_sample(
            sample=sample,
            sample_index=local_index,
            server_uri=server_uri,
            output_path=output_path_for_sample(args.output, local_index, len(local_indices)),
            timeout=args.timeout,
            width=args.width,
            height=args.height,
            radius=args.radius,
            history_horizon=history_horizon,
            robot=robot,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )


if __name__ == "__main__":
    main()
