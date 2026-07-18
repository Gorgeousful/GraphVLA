#!/usr/bin/env python3
"""Compare online server inputs with GT and test patient point-order sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.libero.config.data_config import LIBERO_DATA_CONFIG, LIBERO_HISTORY_HORIZON
from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
from src.dataset.dataset import GenericDataset
from src.model.model import PointQueryModel
from src.training.checkpoint import TrainingCheckpoint

ARRAY_FIELDS = (
    "point_feats",
    "actor_feats",
    "object_condition",
    "actor_condition",
    "object_id",
    "point_id",
    "frame_id",
    "frame_query_frame_id",
)
ROLE_NAMES = ("patient/mug", "target/plate")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=28)
    parser.add_argument(
        "--server-input",
        type=Path,
        default=root / "server_model_input.npz",
        help="Capture produced by eval/client_offline.py --model-input-output.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=PROJECT_ROOT / "examples/libero/result/0716/checkpoints/step_15000.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_gt_sample(episode_index: int, sample_index: int) -> tuple[dict[str, Any], int]:
    dataset = GenericDataset(LIBERO_DATA_CONFIG)
    episode_values = dataset.dataset.hf_dataset["episode_index"]
    episode_indices = [index for index, value in enumerate(episode_values) if int(value) == episode_index]
    if not 0 <= sample_index < len(episode_indices):
        raise IndexError(
            f"sample_index={sample_index} outside episode {episode_index} range [0, {len(episode_indices)})"
        )
    global_index = episode_indices[sample_index]
    return dataset[global_index], global_index


def save_gt_input(path: Path, sample: dict[str, Any]) -> None:
    arrays = {key: to_numpy(sample[key])[None] for key in ARRAY_FIELDS}
    arrays["target_point"] = to_numpy(sample["target"]["point"])[None]
    arrays["target_metric_depth"] = to_numpy(sample["target"]["metric_depth"])[None]
    arrays["target_metric_depth_mask"] = to_numpy(sample["target"]["metric_depth_mask"])[None]
    arrays["target_point_mask"] = to_numpy(sample["target"]["point_mask"])[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def error_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    diff = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def symmetric_chamfer(lhs: np.ndarray, rhs: np.ndarray) -> float:
    distances = np.linalg.norm(lhs[:, None] - rhs[None, :], axis=-1)
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def compare_inputs(
    server_input: dict[str, np.ndarray],
    sample: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {"fields": {}, "object_point_features": {}}
    for key in ARRAY_FIELDS:
        gt = to_numpy(sample[key])[None]
        if key not in server_input:
            report["fields"][key] = {"missing_from_server": True, "gt_shape": list(gt.shape)}
            continue
        online = np.asarray(server_input[key])
        row: dict[str, Any] = {
            "server_shape": list(online.shape),
            "gt_shape": list(gt.shape),
            "shape_equal": online.shape == gt.shape,
        }
        if online.shape == gt.shape:
            if np.issubdtype(gt.dtype, np.integer):
                row["exact_equal"] = bool(np.array_equal(online, gt))
                row["mismatch_count"] = int(np.count_nonzero(online != gt))
            else:
                row.update(error_stats(online, gt))
        report["fields"][key] = row

    online_points = np.asarray(server_input["point_feats"])[0]
    gt_points = to_numpy(sample["point_feats"])
    for slot, role in enumerate(ROLE_NAMES):
        online = online_points[:, slot]
        gt = gt_points[:, slot]
        per_dim = {
            name: error_stats(online[..., dim], gt[..., dim])
            for dim, name in enumerate(("u_norm", "v_norm", "depth_norm", "visible", "metric", "metric_mask"))
        }
        chamfer_uv = [
            symmetric_chamfer(online[frame, :, :2], gt[frame, :, :2])
            for frame in range(online.shape[0])
        ]
        centroid_px = np.linalg.norm(
            (online[..., :2].mean(axis=1) - gt[..., :2].mean(axis=1)) * 128.0,
            axis=-1,
        )
        report["object_point_features"][role] = {
            "aligned_per_dimension": per_dim,
            "unordered_uv_chamfer_norm_mean": float(np.mean(chamfer_uv)),
            "unordered_uv_chamfer_px_mean": float(np.mean(chamfer_uv) * 128.0),
            "centroid_uv_error_px_mean": float(np.mean(centroid_px)),
            "centroid_uv_error_px_per_frame": centroid_px.round(6).tolist(),
        }

    online_actor = np.asarray(server_input["actor_feats"])
    gt_actor = to_numpy(sample["actor_feats"])[None]
    report["actor_features"] = {
        "overall": error_stats(online_actor, gt_actor),
        "aligned_per_dimension": {
            name: error_stats(online_actor[..., dim], gt_actor[..., dim])
            for dim, name in enumerate(
                ("u_norm", "v_norm", "depth_norm", "visible", "gripper_metric", "metric_mask")
            )
        },
    }
    return report


def uv_pixels(points: np.ndarray) -> np.ndarray:
    return (np.asarray(points)[..., :2] + 1.0) * 128.0


def draw_input_comparison(
    output_path: Path,
    server_input: dict[str, np.ndarray],
    sample: dict[str, Any],
) -> None:
    online_points = np.asarray(server_input["point_feats"])[0]
    gt_points = to_numpy(sample["point_feats"])
    online_actor = np.asarray(server_input["actor_feats"])[0, :, 0]
    gt_actor = to_numpy(sample["actor_feats"])[:, 0]
    tile_size = 256
    grid = np.full((4 * tile_size, 4 * tile_size, 3), 250, dtype=np.uint8)
    colors = ((30, 170, 30), (210, 100, 30))

    for frame in range(16):
        row, col = divmod(frame, 4)
        tile = grid[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size]
        for slot, color in enumerate(colors):
            for xy in uv_pixels(gt_points[frame, slot]):
                cv2.circle(tile, tuple(np.rint(xy).astype(int)), 2, color, 1, cv2.LINE_AA)
            for xy in uv_pixels(online_points[frame, slot]):
                cv2.drawMarker(
                    tile,
                    tuple(np.rint(xy).astype(int)),
                    (180, 30, 180),
                    cv2.MARKER_CROSS,
                    5,
                    1,
                    cv2.LINE_AA,
                )
        for xy in uv_pixels(gt_actor[frame]):
            cv2.circle(tile, tuple(np.rint(xy).astype(int)), 4, (20, 20, 20), 1, cv2.LINE_AA)
        for xy in uv_pixels(online_actor[frame]):
            cv2.drawMarker(
                tile,
                tuple(np.rint(xy).astype(int)),
                (0, 0, 240),
                cv2.MARKER_TILTED_CROSS,
                7,
                1,
                cv2.LINE_AA,
            )
        label = f"f={frame - LIBERO_HISTORY_HORIZON} GT=o online=x"
        cv2.putText(tile, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (255, 255), (210, 210, 210), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), grid):
        raise RuntimeError(f"failed to save {output_path}")


def model_batch(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: sample[key].unsqueeze(0).to(device)
        for key in (
            "point_feats",
            "actor_feats",
            "object_condition",
            "actor_condition",
            "object_id",
            "point_id",
            "frame_id",
        )
    }


def permute_patient_points(point_feats: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(point_feats.shape[3], generator=generator)
    if torch.equal(permutation, torch.arange(len(permutation))):
        raise RuntimeError("seed produced an identity permutation")
    permuted = point_feats.clone()
    permuted[:, :, 0] = permuted[:, :, 0, permutation.to(point_feats.device)]
    return permuted, permutation


@torch.inference_mode()
def infer_points(model: PointQueryModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model.infer(**batch, head_names=("point", "metric_depth"))
    return torch.cat([outputs["point"], outputs["metric_depth"]], dim=-1).detach().cpu()


def future_actor_mask(sample: dict[str, Any]) -> np.ndarray:
    object_id = to_numpy(sample["object_id"])
    frame_id = to_numpy(sample["frame_id"])
    point_mask = to_numpy(sample["target"]["point_mask"]).astype(bool)
    return (object_id == 0) & (frame_id > 0) & point_mask


def action_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction)[0, mask]
    gt = np.asarray(target)[mask]
    diff = pred - gt
    uv_l2_norm = np.linalg.norm(diff[:, :2], axis=-1)
    return {
        "query_count": int(mask.sum()),
        "all_feature_mae_norm": float(np.mean(np.abs(diff))),
        "uv_mae_norm": float(np.mean(np.abs(diff[:, :2]))),
        "uv_rmse_norm": float(np.sqrt(np.mean(diff[:, :2] ** 2))),
        "uv_l2_px_mean": float(np.mean(uv_l2_norm) * 128.0),
        "uv_l2_px_max": float(np.max(uv_l2_norm) * 128.0),
        "depth_rel_mae_norm": float(np.mean(np.abs(diff[:, 2]))),
        "visibility_mae": float(np.mean(np.abs(diff[:, 3]))),
        "gripper_metric_mae_norm": float(np.mean(np.abs(diff[:, 4]))),
    }


def sensitivity_metrics(control: np.ndarray, permuted: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    lhs = np.asarray(control)[0, mask]
    rhs = np.asarray(permuted)[0, mask]
    diff = rhs - lhs
    uv_l2 = np.linalg.norm(diff[:, :2], axis=-1)
    return {
        "all_feature_mae_norm": float(np.mean(np.abs(diff))),
        "uv_l2_px_mean": float(np.mean(uv_l2) * 128.0),
        "uv_l2_px_max": float(np.max(uv_l2) * 128.0),
        "depth_rel_mae_norm": float(np.mean(np.abs(diff[:, 2]))),
        "gripper_metric_mae_norm": float(np.mean(np.abs(diff[:, 4]))),
        "max_abs_any_feature": float(np.max(np.abs(diff))),
    }


def display_points(raw_points: torch.Tensor, object_id: torch.Tensor) -> np.ndarray:
    data: dict[str, Any] = {
        "outputs": {
            "point": raw_points[..., :4].clone(),
            "metric_depth": raw_points[..., 4:5].clone(),
        },
        "batch": {"object_id": object_id.unsqueeze(0)},
    }
    for transform in LIBERO_DATA_CONFIG.out_transforms:
        data = transform(data)
    point = data["outputs"]["point"]
    metric_depth = data["outputs"]["metric_depth"]
    return torch.cat([point, metric_depth], dim=-1).detach().cpu().numpy()[0]


def actor_uvd_by_frame(points: np.ndarray, sample: dict[str, Any]) -> dict[int, np.ndarray]:
    object_id = to_numpy(sample["object_id"])
    point_id = to_numpy(sample["point_id"])
    frame_id = to_numpy(sample["frame_id"])
    rows: dict[int, np.ndarray] = {}
    for fid in range(1, 9):
        values = np.zeros((3, 3), dtype=np.float32)
        for pid in range(3):
            matches = np.flatnonzero((object_id == 0) & (point_id == pid) & (frame_id == fid))
            if len(matches) != 1:
                raise ValueError(f"expected one actor query for frame={fid} point={pid}, got {len(matches)}")
            point = points[matches[0]]
            values[pid] = point[[0, 1, 4]]
        rows[fid] = values
    return rows


def draw_future_actions(
    output_path: Path,
    gt_points: np.ndarray,
    control_points: np.ndarray,
    permuted_points: np.ndarray,
    sample: dict[str, Any],
) -> None:
    series = {
        "GT": (actor_uvd_by_frame(gt_points, sample), (20, 20, 20)),
        "control": (actor_uvd_by_frame(control_points, sample), (230, 80, 30)),
        "patient permuted": (actor_uvd_by_frame(permuted_points, sample), (30, 30, 230)),
    }
    tile_size = 256
    grid = np.full((2 * tile_size, 4 * tile_size, 3), 250, dtype=np.uint8)
    for fid in range(1, 9):
        row, col = divmod(fid - 1, 4)
        tile = grid[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size]
        for name, (frames, color) in series.items():
            uvd = frames[fid]
            xy = np.rint(uvd[:, :2]).astype(int)
            cv2.line(tile, tuple(xy[1]), tuple(xy[2]), color, 1, cv2.LINE_AA)
            for point_xy in xy:
                cv2.circle(tile, tuple(point_xy), 4, color, 1 if name == "GT" else -1, cv2.LINE_AA)
        cv2.putText(tile, f"future f={fid}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.rectangle(tile, (0, 0), (255, 255), (210, 210, 210), 1)

    cv2.putText(
        grid,
        "GT=black  control=blue  patient-permuted=red",
        (8, grid.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), grid):
        raise RuntimeError(f"failed to save {output_path}")


def run_point_order_experiment(
    sample: dict[str, Any],
    ckpt_path: Path,
    output_dir: Path,
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(device_name)
    model = PointQueryModel(**LIBERO_MODEL_CONFIG.to_kwargs()).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(TrainingCheckpoint.unwrap_model_state(state), strict=False)
    model.eval()

    control_batch = model_batch(sample, device)
    permuted_batch = {key: value.clone() for key, value in control_batch.items()}
    permuted_batch["point_feats"], permutation = permute_patient_points(
        control_batch["point_feats"],
        seed,
    )

    control = infer_points(model, control_batch)
    permuted = infer_points(model, permuted_batch)
    target = torch.cat(
        [sample["target"]["point"], sample["target"]["metric_depth"]],
        dim=-1,
    )
    mask = future_actor_mask(sample)

    gt_display = display_points(target.unsqueeze(0), sample["object_id"])
    control_display = display_points(control, sample["object_id"])
    permuted_display = display_points(permuted, sample["object_id"])
    draw_future_actions(
        output_dir / "future_action_comparison.png",
        gt_display,
        control_display,
        permuted_display,
        sample,
    )
    np.savez_compressed(
        output_dir / "point_order_predictions.npz",
        permutation=permutation.numpy(),
        control_raw=control.numpy(),
        patient_permuted_raw=permuted.numpy(),
        target_raw=target.numpy(),
        control_display=control_display,
        patient_permuted_display=permuted_display,
        target_display=gt_display,
        future_actor_mask=mask,
    )
    return {
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "permuted_role": "patient",
        "permuted_object_slot": 0,
        "permutation": permutation.tolist(),
        "same_permutation_for_all_history_frames": True,
        "unchanged": ["actor_feats", "conditions", "query ids", "targets"],
        "checkpoint_load": {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "control_vs_gt": action_metrics(control.numpy(), target.numpy(), mask),
        "patient_permuted_vs_gt": action_metrics(permuted.numpy(), target.numpy(), mask),
        "patient_permuted_vs_control": sensitivity_metrics(control.numpy(), permuted.numpy(), mask),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample, global_index = load_gt_sample(args.episode_index, args.sample_index)
    save_gt_input(args.output_dir / "dataset_gt_model_input.npz", sample)

    if not args.server_input.exists():
        raise FileNotFoundError(
            f"server input capture not found: {args.server_input}. "
            "Run eval/client_offline.py with --model-input-output first."
        )
    with np.load(args.server_input) as capture:
        server_input = {key: capture[key] for key in capture.files}

    input_report = compare_inputs(server_input, sample)
    input_report.update(
        {
            "episode_index": args.episode_index,
            "sample_index": args.sample_index,
            "global_index": global_index,
            "server_capture": str(args.server_input),
            "dataset_gt": str(args.output_dir / "dataset_gt_model_input.npz"),
        }
    )
    (args.output_dir / "input_comparison.json").write_text(
        json.dumps(input_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    draw_input_comparison(args.output_dir / "input_comparison.png", server_input, sample)

    experiment = run_point_order_experiment(
        sample,
        args.ckpt_path,
        args.output_dir,
        args.device,
        args.seed,
    )
    experiment.update(
        {
            "episode_index": args.episode_index,
            "sample_index": args.sample_index,
            "global_index": global_index,
        }
    )
    (args.output_dir / "point_order_experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"input_comparison": input_report, "point_order_experiment": experiment}, indent=2))


if __name__ == "__main__":
    main()
