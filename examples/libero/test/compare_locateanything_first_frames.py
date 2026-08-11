#!/usr/bin/env python3
"""Compare LocateAnything point and box prompts on LIBERO initial frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = Path("/data0/luokang/research/LIBERO")
for path in (PROJECT_ROOT, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/graphvla_libero_config")
config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / "config.yaml"
if not config_path.exists():
    benchmark_root = LIBERO_ROOT / "libero" / "libero"
    config_path.write_text(
        "benchmark_root: {0}\n"
        "bddl_files: {0}/bddl_files\n"
        "init_states: {0}/init_files\n"
        "datasets: {1}/datasets\n"
        "assets: {0}/assets\n".format(benchmark_root, LIBERO_ROOT / "libero"),
        encoding="utf-8",
    )

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from src.module.node_locator import NodeLocatorLA


COLORS = ((255, 80, 40), (60, 210, 90), (70, 150, 255), (235, 90, 220))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--num-init-states", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locator-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--taskstructures",
        type=Path,
        default=Path(
            "/data0/luokang/dataset/luokang/lerobot/libero/"
            "libero_with_depth_7_action_0807/meta/taskstructures.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "__tmp__" / "locateanything_first_frames",
    )
    return parser.parse_args()


def load_taskstructures(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["task"]): row for row in rows}


def unique_object_names(taskstructure: dict[str, Any]) -> list[str]:
    names = []
    for subtask in taskstructure.get("subtasks", []):
        for node in subtask.get("nodes", []):
            name = str(node.get("name", ""))
            if str(node.get("role", "")) != "actor" and name and name not in names:
                names.append(name)
    return names


def to_pixels(values: list[list[float]] | list[tuple[float, float]], shape: tuple[int, int]) -> list[list[float]]:
    height, width = shape
    result = []
    for value in values:
        scale = [width, height] if len(value) == 2 else [width, height, width, height]
        converted = np.asarray(value, dtype=np.float64) / 1000.0 * np.asarray(scale)
        converted[0::2] = np.clip(converted[0::2], 0, width - 1)
        converted[1::2] = np.clip(converted[1::2], 0, height - 1)
        if len(value) == 4:
            converted[[0, 2]] = np.sort(converted[[0, 2]])
            converted[[1, 3]] = np.sort(converted[[1, 3]])
        result.append(converted.tolist())
    return result


def draw_result(
    image: np.ndarray,
    name: str,
    color: tuple[int, int, int],
    mode: str,
    pixels: list[list[float]],
) -> None:
    if mode == "point":
        for x, y in pixels:
            cv2.circle(image, (round(x), round(y)), 5, color, -1, lineType=cv2.LINE_AA)
        anchor = pixels[0] if pixels else None
    else:
        anchor = None
        if pixels:
            x1, y1, x2, y2 = pixels[0]
            cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
            anchor = [x1, y1]
    if anchor is not None:
        x, y = map(round, anchor)
        cv2.putText(image, name, (max(3, x + 5), max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, name, (max(3, x + 5), max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    canvas = np.zeros((image.shape[0] + 32, image.shape[1], 3), dtype=np.uint8)
    canvas[32:] = image
    cv2.putText(canvas, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    if args.num_init_states <= 0:
        raise ValueError("--num-init-states must be positive")
    taskstructures = load_taskstructures(args.taskstructures)
    suite = benchmark.get_benchmark("libero_10")()
    output_dir = args.output_dir.resolve() / datetime.now().strftime("%m%d-%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    locator = NodeLocatorLA(device_map=args.device)
    records = []

    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        if task.language not in taskstructures:
            raise KeyError(f"Missing taskstructure for {task.language!r}")
        names = unique_object_names(taskstructures[task.language])
        initial_states = suite.get_task_init_states(task_id)
        if len(initial_states) < args.num_init_states:
            raise ValueError(f"task {task_id} has only {len(initial_states)} initial states")
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=args.resolution,
            camera_widths=args.resolution,
        )
        env.seed(args.seed)
        try:
            for init_state_id in range(args.num_init_states):
                env.reset()
                observation = env.set_init_state(initial_states[init_state_id])
                frame = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, :])
                panels = {mode: frame.copy() for mode in ("point", "box")}
                record = {
                    "task_id": task_id,
                    "task": task.language,
                    "init_state_id": init_state_id,
                    "nodes": {},
                }
                for node_index, name in enumerate(names):
                    color = COLORS[node_index % len(COLORS)]
                    record["nodes"][name] = {}
                    for mode in ("point", "box"):
                        try:
                            result = locator.inference(
                                text=name,
                                image=Image.fromarray(frame),
                                task="pointing" if mode == "point" else "grounding",
                                resize_scale=args.locator_scale,
                            )
                            normalized = result.get("points" if mode == "point" else "boxes") or []
                            pixels = to_pixels(normalized, frame.shape[:2])
                            if mode == "box":
                                pixels = pixels[:1]
                            record["nodes"][name][mode] = {
                                "answer": result.get("answer", ""),
                                "normalized": normalized,
                                "pixels_used": pixels,
                            }
                            if not pixels:
                                message = f"LocateAnything returned no {mode} result"
                                record["nodes"][name][mode]["error"] = message
                                print(
                                    f"task={task_id} init={init_state_id} node={name!r} "
                                    f"mode={mode} error={message}",
                                    flush=True,
                                )
                                continue
                            draw_result(panels[mode], name, color, mode, pixels)
                        except Exception as error:
                            record["nodes"][name][mode] = {"error": str(error)}
                            print(f"task={task_id} init={init_state_id} node={name!r} mode={mode} error={error}", flush=True)
                comparison = np.hstack([
                    add_header(panels["point"], f"Task {task_id} | Init {init_state_id} | Point"),
                    add_header(panels["box"], f"Task {task_id} | Init {init_state_id} | Box"),
                ])
                save_path = output_dir / f"task_{task_id:02d}_init_{init_state_id:02d}_point_vs_box.png"
                Image.fromarray(comparison).save(save_path)
                record["image"] = str(save_path)
                records.append(record)
                print(f"saved: {save_path}", flush=True)
        finally:
            env.close()

    (output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
