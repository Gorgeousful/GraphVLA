#!/usr/bin/env python3
"""Save the first LIBERO rendered frame for quick environment debugging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/graphvla_libero_config")
libero_config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
libero_config_dir.mkdir(parents=True, exist_ok=True)
libero_config_file = libero_config_dir / "config.yaml"
if not libero_config_file.exists():
    benchmark_root = LIBERO_ROOT / "libero" / "libero"
    libero_config_file.write_text(
        "benchmark_root: {0}\n"
        "bddl_files: {0}/bddl_files\n"
        "init_states: {0}/init_files\n"
        "datasets: {1}/datasets\n"
        "assets: {0}/assets\n".format(benchmark_root, LIBERO_ROOT / "libero"),
        encoding="utf-8",
    )

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="examples/libero/test/first_frame.png",
        help="Output image path, relative to GraphVLA root or absolute.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save raw robosuite image without the LIBERO client flip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark(args.task_suite_name)()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.episode])
        frame = np.asarray(obs["agentview_image"])
        if not args.raw:
            frame = np.ascontiguousarray(frame[::-1, :])

        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_path, frame)
        print(f"task {args.task_id}: {task.language}")
        print(f"saved first frame to: {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
