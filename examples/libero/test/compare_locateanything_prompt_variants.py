#!/usr/bin/env python3
"""Compare box-grounding prompts for the LIBERO cream-cheese package."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from compare_locateanything_first_frames import LIBERO_ROOT, PROJECT_ROOT, to_pixels
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from src.module.node_locator import NodeLocatorLA


PROMPTS = (
    "small white box",
    "cream cheese box",
    "cream cheese package",
    "blue cream cheese box",
    "small blue rectangular box",
    "blue rectangular cream cheese package",
    "small blue box",
    "small blue and white box",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--num-init-states", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locator-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "__tmp__" / "locateanything_prompt_variants",
    )
    return parser.parse_args()


def reference_points(path: Path, task_id: int) -> dict[int, list[float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    points = {}
    for row in rows:
        if int(row["task_id"]) != task_id:
            continue
        result = row["nodes"]["small white box"]["point"]
        pixels = result.get("pixels_used") or []
        if not pixels:
            raise ValueError(f"Missing reference point for init state {row['init_state_id']}")
        points[int(row["init_state_id"])] = list(map(float, pixels[0]))
    return points


def contains(box: list[float], point: list[float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def add_fitted_header(image: np.ndarray, text: str) -> np.ndarray:
    header_height = 34
    canvas = np.zeros((image.shape[0] + header_height, image.shape[1], 3), dtype=np.uint8)
    canvas[header_height:] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
    if text_width > image.shape[1] - 12:
        scale *= (image.shape[1] - 12) / text_width
    cv2.putText(canvas, text, (6, 22), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_panel(
    frame: np.ndarray,
    prompt_index: int,
    prompt: str,
    point: list[float],
    box: list[float] | None,
) -> np.ndarray:
    panel = frame.copy()
    x, y = map(round, point)
    cv2.drawMarker(panel, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 11, 2)
    if box is None:
        cv2.putText(panel, "NO BOX", (82, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 80), 2, cv2.LINE_AA)
    else:
        x1, y1, x2, y2 = map(round, box)
        color = (70, 230, 90) if contains(box, point) else (255, 70, 70)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
    return add_fitted_header(panel, f"P{prompt_index}: {prompt}")


def main() -> None:
    args = parse_args()
    references = reference_points(args.reference_results.resolve(), args.task_id)
    suite = benchmark.get_benchmark("libero_10")()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if len(initial_states) < args.num_init_states:
        raise ValueError(f"task {args.task_id} has only {len(initial_states)} initial states")

    output_dir = args.output_dir.resolve() / datetime.now().strftime("%m%d-%H%M")
    output_dir.mkdir(parents=True, exist_ok=True)
    locator = NodeLocatorLA(device_map=args.device)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"box": 0, "contains_reference": 0})
    try:
        for init_state_id in range(args.num_init_states):
            env.reset()
            observation = env.set_init_state(initial_states[init_state_id])
            frame = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, :])
            reference = references[init_state_id]
            panels = []
            prompt_results = []
            for prompt_index, prompt in enumerate(PROMPTS):
                result = locator.inference(
                    text=prompt,
                    image=Image.fromarray(frame),
                    task="grounding",
                    resize_scale=args.locator_scale,
                )
                normalized = result.get("boxes") or []
                pixel_boxes = to_pixels(normalized, frame.shape[:2])
                box = pixel_boxes[0] if pixel_boxes else None
                covers_reference = box is not None and contains(box, reference)
                if box is not None:
                    counts[prompt]["box"] += 1
                if covers_reference:
                    counts[prompt]["contains_reference"] += 1
                prompt_results.append({
                    "prompt": prompt,
                    "answer": result.get("answer", ""),
                    "normalized_boxes": normalized,
                    "box_used": box,
                    "contains_reference": covers_reference,
                })
                panels.append(draw_panel(frame, prompt_index, prompt, reference, box))

            rows = [np.hstack(panels[index:index + 4]) for index in range(0, len(panels), 4)]
            comparison = np.vstack(rows)
            save_path = output_dir / f"task_{args.task_id:02d}_init_{init_state_id:02d}_prompt_variants.png"
            Image.fromarray(comparison).save(save_path)
            records.append({
                "task_id": args.task_id,
                "task": task.language,
                "init_state_id": init_state_id,
                "reference_point": reference,
                "prompts": prompt_results,
                "image": str(save_path),
            })
            print(f"saved: {save_path}", flush=True)
    finally:
        env.close()

    summary = [
        {"prompt_index": index, "prompt": prompt, **counts[prompt]}
        for index, prompt in enumerate(PROMPTS)
    ]
    payload = {"summary": summary, "records": records}
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in summary:
        print(
            f"P{item['prompt_index']} {item['prompt']!r}: "
            f"box={item['box']}/{args.num_init_states} "
            f"contains_reference={item['contains_reference']}/{args.num_init_states}"
        )
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
