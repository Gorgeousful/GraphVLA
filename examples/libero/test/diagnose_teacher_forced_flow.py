#!/usr/bin/env python3
"""Evaluate checkpoint motion predictions from transformed GT dataset inputs."""
from __future__ import annotations

import argparse, gc, glob, json, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.server import InferenceModel
from src.dataset.dataset import GenericDataset
from src.training.checkpoint import TrainingCheckpoint

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--samples-per-phase", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()

def select_samples(dataset_dir: Path, count: int):
    candidates = defaultdict(list)
    for path in sorted(glob.glob(str(dataset_dir / "data/chunk-*/*.parquet"))):
        df = pd.read_parquet(path, columns=["index", "episode_index", "frame_index", "subtask_id", "is_contact", "is_complete"])
        for sid in sorted(df.subtask_id.unique()):
            local = df.index[df.subtask_id == sid].to_numpy()
            contact = local[df.loc[local, "is_contact"].to_numpy(bool)]
            if len(local) < 20 or len(contact) == 0:
                continue
            onset = int(contact[0]); complete = local[df.loc[local, "is_complete"].to_numpy(bool)]
            choices = {
                "approach": max(int(local[0]) + 5, onset - 10),
                "contact_onset": onset,
                "transport": min(onset + 15, int(local[-1]) - 10),
                "pre_completion": max(onset + 1, int(complete[0]) - 5) if len(complete) else int(local[-1]) - 5,
            }
            for phase, row_index in choices.items():
                row = df.loc[row_index]
                candidates[phase].append({"dataset_index": int(row["index"]), "episode_index": int(row["episode_index"]), "frame_index": int(row["frame_index"]), "subtask_id": int(row["subtask_id"]), "phase": phase})
    selected = []
    for phase in ("approach", "contact_onset", "transport", "pre_completion"):
        values = candidates[phase]
        for index in np.linspace(0, len(values) - 1, min(count, len(values)), dtype=int):
            selected.append(values[index])
    return selected

def unnormalize_targets(sample, transforms):
    target = sample["target"]
    data = {"outputs": {"action_plan": target["action"].unsqueeze(0).clone(), "point_plan": target["points"][:, 0, :3].unsqueeze(0).clone()}, "batch": {}}
    for transform in transforms:
        data = transform(data)
    return tuple(data["outputs"][key].detach().cpu().numpy() for key in ("action_plan", "point_plan"))

def metrics(pred, gt):
    error = np.linalg.norm(pred - gt, axis=-1); pd = pred[-1] - pred[0]; gd = gt[-1] - gt[0]
    pm = float(np.linalg.norm(pd)); gm = float(np.linalg.norm(gd))
    return {"ade": float(error.mean()), "fde": float(error[-1]), "pred_motion": pm, "gt_motion": gm, "motion_ratio": pm / max(gm, 1e-12), "direction_cosine": float(np.dot(pd, gd) / max(pm * gm, 1e-12))}

def evaluate(checkpoint, device, selected, seed):
    data_config, model_config, _ = TrainingCheckpoint.load_config_snapshots(checkpoint)
    data_config.load_videos = False; dataset = GenericDataset(data_config)
    inference = InferenceModel(model_kwargs=model_config.to_kwargs(), data_kwargs=data_config.to_kwargs(), ckpt_path=checkpoint, device=device)
    records = []
    for order, item in enumerate(selected):
        sample = dataset[item["dataset_index"]]; torch.manual_seed(seed + order)
        outputs = inference.infer({"entity_points": sample["entity_points"].unsqueeze(0), "entity_point_mask": sample["entity_point_mask"].unsqueeze(0), "scene_condition": sample["scene_condition"].unsqueeze(0)})
        gt_action, gt_points = unnormalize_targets(sample, tuple(data_config.out_transforms))
        pred_action = np.asarray(outputs["action_plan"], float); pred_points = np.asarray(outputs["point_plan"], float)
        action = metrics(pred_action[0, :, :3], gt_action[0, :, :3]); points = metrics(pred_points[0].mean(1), gt_points[0].mean(1))
        records.append({**item, "checkpoint": checkpoint.stem, "action": action, "actor_points": points, "pred_action_xyz": pred_action[0, :, :3].tolist(), "gt_action_xyz": gt_action[0, :, :3].tolist()})
        print(f"{checkpoint.stem} {item['phase']} ep={item['episode_index']} frame={item['frame_index']} action_ratio={action['motion_ratio']:.3f} point_ratio={points['motion_ratio']:.3f}", flush=True)
    del inference, dataset; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return records

def summarize(records):
    groups = defaultdict(list)
    for record in records: groups[(record["checkpoint"], record["phase"])].append(record)
    result = []
    for (checkpoint, phase), values in groups.items():
        row = {"checkpoint": checkpoint, "phase": phase, "count": len(values)}
        for head in ("action", "actor_points"):
            for metric in ("ade", "fde", "pred_motion", "gt_motion", "motion_ratio", "direction_cosine"):
                row[f"{head}_{metric}"] = float(np.mean([value[head][metric] for value in values]))
        result.append(row)
    return result

def plot(summary, path):
    checkpoints = list(dict.fromkeys(row["checkpoint"] for row in summary)); phases = ["approach", "contact_onset", "transport", "pre_completion"]; lookup = {(r["checkpoint"], r["phase"]): r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True); width = .8 / max(len(checkpoints), 1); x = np.arange(4)
    for ax, head, title in zip(axes, ("action", "actor_points"), ("Action motion ratio", "Actor-point motion ratio")):
        for i, checkpoint in enumerate(checkpoints):
            values = [lookup.get((checkpoint, phase), {}).get(f"{head}_motion_ratio", np.nan) for phase in phases]
            ax.bar(x + (i - (len(checkpoints)-1)/2)*width, values, width, label=checkpoint)
        ax.axhline(1, color="black", linewidth=1, linestyle="--"); ax.set_title(title); ax.set_xticks(x, [p.replace("_", "\n") for p in phases]); ax.set_ylabel("Predicted / GT motion"); ax.grid(axis="y", alpha=.25)
    axes[1].legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    first_data, _, _ = TrainingCheckpoint.load_config_snapshots(args.checkpoints[0]); selected = select_samples(Path(first_data.dataset_dir), args.samples_per_phase); records = []
    for checkpoint in args.checkpoints: records.extend(evaluate(checkpoint, torch.device(args.device), selected, args.seed))
    summary = summarize(records); (args.output_dir/"records.json").write_text(json.dumps(records, indent=2)); (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)); plot(summary, args.output_dir/"motion_ratio.png"); print(f"saved results to {args.output_dir}")

if __name__ == "__main__": main()
