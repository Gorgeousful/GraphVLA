#!/usr/bin/env python3
"""Verify the dataset GT: does integrating actions reproduce the state?

No model involved. LIBERO raw actions are controller-scaled deltas (robosuite
action space), not meters/radians: the dataset stores them before the
env-side scaling, so this script first fits the linear action -> state scales
from the GT itself, then checks:

  1. actions_camera vs raw world actions: rotation transform exact?
  2. integrate scaled actions from state[i] and compare with state[i+t]
     (position in mm, rotation in degrees), for several conventions.

Usage:
  python examples/libero/test/diagnose_point_action_consistency.py [--episode N]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.spatial.transform import Rotation as R

cs = Console()

FUTURE_HORIZON = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data0/luokang/dataset/luokang/lerobot/libero/libero_with_depth_7_action"),
    )
    parser.add_argument("--episode", type=int, default=None, help="Episode index (default: random)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for the random draw")
    return parser.parse_args()


def load_camera(dataset_dir: Path, task_index: int) -> np.ndarray:
    rows = json.loads((dataset_dir / "meta" / "cameras.json").read_text())
    for row in rows:
        if int(row["task_index"]) == int(task_index):
            return np.asarray(row["cameras"]["agentview"]["extrinsic"], dtype=np.float64)
    raise KeyError(f"Missing camera for task_index={task_index}")


def pick_episode(dataset_dir: Path, episode: int | None, rng: random.Random) -> Path:
    files = sorted(glob.glob(str(dataset_dir / "data" / "chunk-*" / "episode_*.parquet")))
    if not files:
        raise RuntimeError(f"No episode parquet files under {dataset_dir / 'data'}")
    if episode is None:
        return Path(rng.choice(files))
    matches = [
        path for path in files
        if int(pd.read_parquet(path, columns=["episode_index"]).iloc[0]["episode_index"]) == episode
    ]
    if not matches:
        raise RuntimeError(f"Episode {episode} not found in {dataset_dir}")
    return Path(matches[0])


def median_scale_per_dim(delta: np.ndarray, action: np.ndarray) -> np.ndarray:
    scales = np.zeros(action.shape[1])
    for k in range(action.shape[1]):
        mask = np.abs(action[:, k]) > 1e-3
        scales[k] = float(np.median(delta[mask, k] / action[mask, k]))
    return scales


def integrate_world(
    state: np.ndarray,
    actions_world: np.ndarray,
    steps: int,
    *,
    s_pos: np.ndarray,
    s_rot: np.ndarray,
    compose: bool,
) -> np.ndarray:
    """Integrate scaled raw actions in the world frame; return [steps, 6] TCP states."""
    pos = np.asarray(state[:3], dtype=np.float64).copy()
    rot = R.from_rotvec(np.asarray(state[3:6], dtype=np.float64))
    out = np.zeros((steps, 6), dtype=np.float64)
    for step in range(steps):
        action = np.asarray(actions_world[step], dtype=np.float64)
        pos = pos + s_pos * action[:3]
        drot = s_rot * action[3:6]
        rot = R.from_rotvec(drot) * rot if compose else R.from_rotvec(rot.as_rotvec() + drot)
        out[step, :3] = pos
        out[step, 3:6] = rot.as_rotvec()
    return out


def rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Angle between predicted and target rotvec states, in degrees."""
    relative = (R.from_rotvec(pred) * R.from_rotvec(target).inv()).as_rotvec()
    return np.degrees(np.linalg.norm(relative, axis=-1))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    path = pick_episode(args.dataset_dir, args.episode, rng)
    df = pd.read_parquet(path)
    episode = int(df["episode_index"].iloc[0])
    task_index = int(df["task_index"].iloc[0])
    n = len(df)
    cs.print(f"episode={episode} task_index={task_index} n_frames={n}")

    states = np.stack([np.asarray(v, dtype=np.float64) for v in df["state"]])
    actions_world = np.stack([np.asarray(v, dtype=np.float64) for v in df["actions"]])
    actions_cam = np.stack([np.asarray(v, dtype=np.float64) for v in df["actions_camera"]])

    # --- check 1: actions_camera vs raw world actions ---
    extrinsic = load_camera(args.dataset_dir, task_index)
    world_to_camera = extrinsic[:3, :3].T
    expected = np.stack([
        np.concatenate([world_to_camera @ a[:3], world_to_camera @ a[3:6], a[6:7]])
        for a in actions_world
    ])
    transform_err = np.abs(expected[:, :6] - actions_cam[:, :6]).mean() * 1000.0
    cs.print(f"actions_camera transform: mean err={transform_err:.4f} (mm/rad)")

    # --- fit action -> state scales from GT (alignment A: action[i] -> i+1) ---
    dpos = np.diff(states[:, :3], axis=0)
    r0 = R.from_rotvec(states[:-1, 3:6])
    r1 = R.from_rotvec(states[1:, 3:6])
    drot = (r1 * r0.inv()).as_rotvec()
    s_pos = median_scale_per_dim(dpos, actions_world[:-1, :3])
    s_rot = median_scale_per_dim(drot, actions_world[:-1, 3:6])
    dq = np.diff(states[:, 6:8], axis=0)
    mask_grip = np.abs(actions_world[:-1, 6]) > 1e-3
    s_grip = float(np.median((dq[mask_grip, 0] + dq[mask_grip, 1]) / 2.0 / actions_world[:-1, 6][mask_grip]))
    cs.print("fitted scales (per dim): "
             f"s_pos={np.round(s_pos, 5).tolist()} m/unit  "
             f"s_rot={np.round(s_rot, 5).tolist()} rad/unit  "
             f"s_grip={s_grip:.5f} (gripper is a saturated +/-1 command)")

    # --- 1-step alignment check: A (i -> i+1) vs B (action[i] produced state[i]) ---
    one_step = {}
    for align in ("a", "b"):
        if align == "a":  # action[i] transitions state i -> i+1
            start, actions, target = states[:-1], actions_world[:-1], states[1:]
        else:             # action[i] produced state[i] (transitions i-1 -> i)
            start, actions, target = states[:-2], actions_world[1:-1], states[1:-1]
        pred = np.stack([
            integrate_world(start[i], actions[i : i + 1], 1, s_pos=s_pos, s_rot=s_rot, compose=True)[0]
            for i in range(len(actions))
        ])
        one_step[align] = (
            np.linalg.norm(pred[:, :3] - target[:, :3], axis=1).mean() * 1000.0,
            rotation_error_deg(pred[:, 3:6], target[:, 3:6]).mean(),
        )
    cs.print(
        f"1-step alignment: A(i->i+1) pos={one_step['a'][0]:.2f} mm rot={one_step['a'][1]:.3f} deg | "
        f"B(action produced i) pos={one_step['b'][0]:.2f} mm rot={one_step['b'][1]:.3f} deg"
    )

    # --- multi-step: integrate t actions from state[i], compare state[i+t] ---
    valid = list(range(n - FUTURE_HORIZON))
    pos_err = {True: [], False: []}
    rot_err = {True: [], False: []}
    for i in valid:
        for compose in (True, False):
            pred = integrate_world(
                states[i], actions_world[i : i + FUTURE_HORIZON], FUTURE_HORIZON,
                s_pos=s_pos, s_rot=s_rot, compose=compose,
            )
            target = states[i + 1 : i + 1 + FUTURE_HORIZON]
            pos_err[compose].append(np.linalg.norm(pred[:, :3] - target[:, :3], axis=1) * 1000.0)
            rot_err[compose].append(rotation_error_deg(pred[:, 3:6], target[:, 3:6]))
    pos_compose = np.stack(pos_err[True])
    pos_additive = np.stack(pos_err[False])
    rot_compose = np.stack(rot_err[True])
    rot_additive = np.stack(rot_err[False])

    table = Table(title=f"integrate action -> state residual (align A, compose), episode {episode}")
    table.add_column("step", justify="right")
    table.add_column("pos mean (mm)", justify="right")
    table.add_column("pos p95 (mm)", justify="right")
    table.add_column("rot mean (deg)", justify="right")
    table.add_column("rot p95 (deg)", justify="right")
    for t in range(FUTURE_HORIZON):
        table.add_row(
            str(t + 1),
            f"{pos_compose[:, t].mean():7.2f}",
            f"{np.percentile(pos_compose[:, t], 95):7.2f}",
            f"{rot_compose[:, t].mean():7.3f}",
            f"{np.percentile(rot_compose[:, t], 95):7.3f}",
        )
    cs.print(table)

    for compose, pos, rot in (
        (True, pos_compose, rot_compose),
        (False, pos_additive, rot_additive),
    ):
        cs.print(
            f"rot={'compose' if compose else 'additive'}: "
            f"pos mean={pos.mean():.2f} mm (step10={pos[:, 9].mean():.2f}) | "
            f"rot mean={rot.mean():.3f} deg (step10={rot[:, 9].mean():.3f})"
        )

    best_pos = pos_compose.mean()
    if one_step["a"][0] > 5.0:
        verdict = "1-step action->state already off -> action scale or alignment is wrong"
    elif best_pos > 10.0:
        verdict = f"actions roughly reproduce state (1-step {one_step['a'][0]:.1f} mm) but 10-step drift is large ({pos_compose[:, 9].mean():.1f} mm)"
    else:
        verdict = f"GT actions reproduce state well (1-step {one_step['a'][0]:.1f} mm, 10-step {pos_compose[:, 9].mean():.1f} mm)"
    cs.print(f"[bold]{verdict}[/bold]")


if __name__ == "__main__":
    main()
