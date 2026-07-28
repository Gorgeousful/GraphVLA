from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from src.common.schema import NodeRole, json_to_taskstructure


COLORS = [
    (255, 80, 80),
    (30, 160, 255),
    (50, 205, 80),
    (255, 190, 20),
    (220, 70, 220),
    (30, 220, 220),
]
CACHE_VERSION = 1


def load_task_nodes(dataset_dir: Path) -> dict[int, list[str]]:
    task_indices = {}
    for line in (dataset_dir / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            task_indices[str(row["task"])] = int(row["task_index"])

    nodes_by_task = {}
    for line in (dataset_dir / "meta" / "taskstructures.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        structure = json_to_taskstructure(json.loads(line))
        if structure.task in task_indices:
            nodes_by_task[task_indices[structure.task]] = [
                node.name
                for subtask in structure.subtask_list
                for node in (subtask.node_list or [])
                if node.role != NodeRole.ACTOR
            ]
    return nodes_by_task


def read_first_rgb(parquet_path: Path) -> tuple[np.ndarray, int, int]:
    row = pd.read_parquet(
        parquet_path, columns=["image", "episode_index", "task_index"]
    ).iloc[0]
    item = row["image"]
    payload = item.get("bytes") if isinstance(item, dict) else None
    if payload is None:
        raise ValueError(f"image in {parquet_path} must contain embedded bytes")
    with Image.open(io.BytesIO(payload)) as image:
        frame_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return (
        np.ascontiguousarray(np.fliplr(frame_rgb)),
        int(row["episode_index"]),
        int(row["task_index"]),
    )
INDEX_HTML = """\
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Node Initialization Cache</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main>
    <section id="overview">
      <header>
        <div><p class="eyebrow">GraphVLA · First-frame inspection</p><h1>Node initialization cache</h1></div>
        <label class="select-field">任务<select id="task-select"></select></label>
      </header>
      <nav class="pagination" aria-label="分页">
        <button id="previous-page">← 上一页</button><span id="page-label"></span><button id="next-page">下一页 →</button>
      </nav>
      <div id="episode-grid" aria-live="polite"></div>
    </section>

    <section id="detail" hidden>
      <header class="detail-header">
        <button id="back-button" class="quiet">← 返回任务预览</button>
        <div><p class="eyebrow">Point correction</p><h1 id="detail-title"></h1></div>
        <label class="select-field">节点<select id="node-select"></select></label>
      </header>
      <div class="annotation-layout">
        <div class="canvas-shell"><canvas id="annotation-canvas"></canvas></div>
        <aside>
          <h2>重新选择 point</h2>
          <p>点击画面选择新位置。白色十字是尚未保存的新 point，保存后只重新分割当前节点。</p>
          <div id="point-readout">尚未选择新 point</div>
          <button id="save-button" class="primary" disabled>重新分割并保存</button>
          <button id="restore-button">恢复初始结果</button>
          <p id="status" role="status"></p>
        </aside>
      </div>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
"""

APP_JS = """\
const state = { task: null, page: 0, episode: null, node: 0, pending: null, baseImage: null };
const $ = (selector) => document.querySelector(selector);
const overview = $("#overview"), detail = $("#detail"), taskSelect = $("#task-select");
const nodeSelect = $("#node-select"), grid = $("#episode-grid"), pageLabel = $("#page-label");
const previousPage = $("#previous-page"), nextPage = $("#next-page");
const canvas = $("#annotation-canvas"), context = canvas.getContext("2d");
const pointReadout = $("#point-readout"), saveButton = $("#save-button");
const restoreButton = $("#restore-button"), status = $("#status");

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadTasks() {
  const data = await api("/api/tasks");
  taskSelect.replaceChildren(...data.tasks.map((task) => {
    const option = document.createElement("option");
    option.value = task.task_index;
    option.textContent = `Task ${task.task_index} · ${task.episode_count} episodes`;
    return option;
  }));
  state.task = Number(taskSelect.value);
  await loadPage();
}

async function loadPage() {
  grid.classList.add("loading");
  const data = await api(`/api/tasks/${state.task}/episodes?page=${state.page}`);
  state.page = data.page;
  pageLabel.textContent = `Task ${state.task} · 第 ${data.page + 1}/${data.page_count} 页`;
  previousPage.disabled = data.page === 0;
  nextPage.disabled = data.page + 1 >= data.page_count;
  grid.replaceChildren(...data.episodes.map((episode) => {
    const card = document.createElement("button");
    card.className = "episode-card";
    card.innerHTML = `<img src="/api/episodes/${episode}/preview?t=${Date.now()}" alt="Episode ${episode}"><span>episode ${String(episode).padStart(6, "0")}</span>`;
    card.addEventListener("click", () => openEpisode(episode));
    return card;
  }));
  grid.classList.remove("loading");
}

async function openEpisode(episode) {
  const data = await api(`/api/episodes/${episode}`);
  state.episode = episode;
  state.node = 0;
  state.pending = null;
  $("#detail-title").textContent = `Episode ${String(episode).padStart(6, "0")} · Task ${data.task_index}`;
  nodeSelect.replaceChildren(...data.node_names.map((name, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `N${index}: ${name}`;
    return option;
  }));
  overview.hidden = true;
  detail.hidden = false;
  setStatus("");
  await loadOverlay();
}

async function loadOverlay() {
  state.pending = null;
  saveButton.disabled = true;
  pointReadout.textContent = "尚未选择新 point";
  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    state.baseImage = image;
    drawCanvas();
  };
  image.src = `/api/episodes/${state.episode}/overlay?node_index=${state.node}&t=${Date.now()}`;
}

function drawCanvas() {
  if (!state.baseImage) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(state.baseImage, 0, 0);
  if (state.pending) {
    const [x, y] = state.pending;
    context.save();
    context.strokeStyle = "#fff";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x - 9, y); context.lineTo(x + 9, y);
    context.moveTo(x, y - 9); context.lineTo(x, y + 9);
    context.stroke();
    context.restore();
  }
}

canvas.addEventListener("click", (event) => {
  const bounds = canvas.getBoundingClientRect();
  const x = Math.round((event.clientX - bounds.left) * canvas.width / bounds.width);
  const y = Math.round((event.clientY - bounds.top) * canvas.height / bounds.height);
  state.pending = [Math.max(0, Math.min(canvas.width - 1, x)), Math.max(0, Math.min(canvas.height - 1, y))];
  pointReadout.textContent = `待保存 point：(${state.pending[0]}, ${state.pending[1]})`;
  saveButton.disabled = false;
  setStatus("");
  drawCanvas();
});

saveButton.addEventListener("click", async () => {
  if (!state.pending) return;
  setBusy(true);
  try {
    const [x, y] = state.pending;
    const result = await api(`/api/episodes/${state.episode}/nodes/${state.node}/point`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ x, y }),
    });
    await loadOverlay();
    setStatus(result.message);
  } catch (error) { setStatus(error.message, true); }
  finally { setBusy(false); }
});

restoreButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    const result = await api(`/api/episodes/${state.episode}/nodes/${state.node}/restore`, { method: "POST" });
    await loadOverlay();
    setStatus(result.message);
  } catch (error) { setStatus(error.message, true); }
  finally { setBusy(false); }
});

function setBusy(busy) {
  saveButton.disabled = busy || !state.pending;
  restoreButton.disabled = busy;
  nodeSelect.disabled = busy;
}
function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}

taskSelect.addEventListener("change", async () => { state.task = Number(taskSelect.value); state.page = 0; await loadPage(); });
previousPage.addEventListener("click", async () => { state.page -= 1; await loadPage(); });
nextPage.addEventListener("click", async () => { state.page += 1; await loadPage(); });
nodeSelect.addEventListener("change", async () => { state.node = Number(nodeSelect.value); setStatus(""); await loadOverlay(); });
$("#back-button").addEventListener("click", async () => { detail.hidden = true; overview.hidden = false; await loadPage(); });
loadTasks().catch((error) => { grid.textContent = `加载失败：${error.message}`; });
"""

STYLE_CSS = """\
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0c0e12;
  color: #f4f5f7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-width: 320px;
  background: radial-gradient(circle at top left, rgba(75, 104, 255, .12), transparent 36rem), #0c0e12;
}
main { width: min(1480px, calc(100% - 48px)); margin: 0 auto; padding: 42px 0 64px; }
header, .pagination, .detail-header, .annotation-layout { display: flex; align-items: center; gap: 20px; }
header { justify-content: space-between; margin-bottom: 24px; }
h1, h2, p { margin-top: 0; }
h1 { margin-bottom: 0; font-size: clamp(28px, 4vw, 48px); letter-spacing: -.045em; }
h2 { font-size: 20px; }
.eyebrow { margin-bottom: 8px; color: #8994ad; font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.select-field { display: grid; gap: 7px; min-width: 230px; color: #a8afbf; font-size: 12px; }
select, button { border: 1px solid #2d323d; border-radius: 10px; background: #171a20; color: inherit; font: inherit; }
select { padding: 11px 13px; }
button { padding: 10px 15px; cursor: pointer; }
button:hover:not(:disabled) { border-color: #637cff; background: #202532; }
button:disabled { cursor: not-allowed; opacity: .4; }
.primary { border-color: #637cff; background: #5169e8; }
.quiet { background: transparent; }
.pagination { justify-content: center; margin-bottom: 22px; }
#page-label { min-width: 210px; color: #bfc5d2; text-align: center; }
#episode-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; transition: opacity 120ms ease; }
#episode-grid.loading { opacity: .45; }
.episode-card { overflow: hidden; padding: 0; text-align: left; }
.episode-card img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; background: #050607; }
.episode-card span { display: block; padding: 10px 12px; color: #c9ced8; font-family: ui-monospace, monospace; font-size: 12px; }
.detail-header { justify-content: space-between; }
.annotation-layout { align-items: stretch; justify-content: center; }
.canvas-shell { display: grid; flex: 1 1 760px; min-height: 560px; place-items: center; border: 1px solid #242934; border-radius: 16px; background: #050607; }
#annotation-canvas { width: min(100%, 760px); cursor: crosshair; }
aside { flex: 0 0 310px; padding: 24px; border: 1px solid #242934; border-radius: 16px; background: #12151a; }
aside p { color: #9ba3b3; line-height: 1.6; }
aside button { width: 100%; margin-top: 10px; }
#point-readout, #status { min-height: 46px; margin: 18px 0 4px; padding: 12px; border-radius: 9px; background: #090b0e; color: #cfd4df; font-family: ui-monospace, monospace; font-size: 12px; }
#status.error { color: #ff8c8c; }
@media (max-width: 900px) {
  main { width: min(100% - 24px, 720px); padding-top: 24px; }
  header, .detail-header, .annotation-layout { align-items: stretch; flex-direction: column; }
  #episode-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .canvas-shell { min-height: 0; }
  aside { flex-basis: auto; }
}
"""


class PointRequest(BaseModel):
    x: int
    y: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache first-frame node localization/masks and correct them in a local web UI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-dir", type=Path, required=True)
        subparser.add_argument(
            "--cache-dir",
            type=Path,
            help="Defaults to <dataset-dir>/meta/node_initialization_cache.",
        )

    initialize = subparsers.add_parser("initialize")
    add_common(initialize)
    initialize.add_argument(
        "--locator",
        choices=("robobrain", "locateanything"),
        default="robobrain",
    )
    initialize.add_argument(
        "--segmenter",
        choices=("sam2", "sam3"),
        default="sam2",
    )
    initialize.add_argument("--scale", type=float, default=2.0)
    initialize.add_argument("--episodes", type=int, nargs="*")
    initialize.add_argument("--overwrite", action="store_true")

    serve = subparsers.add_parser("serve")
    add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_dir = args.dataset_dir.resolve()
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else dataset_dir / "meta" / "node_initialization_cache"
    )
    return dataset_dir, cache_dir


def episode_parquet_paths(dataset_dir: Path) -> dict[int, Path]:
    paths = {}
    for path in sorted((dataset_dir / "data").glob("chunk-*/*.parquet")):
        try:
            episode_index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unexpected episode parquet name: {path}") from exc
        paths[episode_index] = path
    if not paths:
        raise RuntimeError(f"No episode parquet files found under {dataset_dir / 'data'}")
    return paths


def pack_points(point_groups: list[list[list[float]]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(points) for points in point_groups], dtype=np.int32)
    max_points = max(int(counts.max(initial=0)), 1)
    packed = np.full((len(point_groups), max_points, 2), np.nan, dtype=np.float32)
    for node_index, points in enumerate(point_groups):
        if points:
            packed[node_index, : len(points)] = np.asarray(points, dtype=np.float32)
    return packed, counts


def atomic_save_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}_", suffix=".npz", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            np.savez_compressed(temp_file, **data)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cached:
        return {key: cached[key].copy() for key in cached.files}


def cache_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(data["metadata_json"].item()))


def cache_path(cache_dir: Path, episode_index: int) -> Path:
    return cache_dir / f"episode_{episode_index:06d}.npz"


def build_models(locator_name: str, segmenter_name: str) -> tuple[Any, Any]:
    if locator_name == "robobrain":
        from src.module.node_locator import NodeLocatorRobo

        locator = NodeLocatorRobo()
    else:
        from src.module.node_locator import NodeLocatorLA

        locator = NodeLocatorLA()

    if segmenter_name == "sam2":
        from src.module.node_segmenter import NodeSegmenterSAM2

        segmenter = NodeSegmenterSAM2()
    else:
        from src.module.node_segmenter import NodeSegmenter

        segmenter = NodeSegmenter()
    return locator, segmenter


def initialize_cache(args: argparse.Namespace) -> None:
    dataset_dir, output_dir = resolve_paths(args)
    if args.scale <= 0:
        raise ValueError("--scale must be positive")

    parquet_paths = episode_parquet_paths(dataset_dir)
    selected = sorted(set(args.episodes)) if args.episodes else sorted(parquet_paths)
    missing = [episode for episode in selected if episode not in parquet_paths]
    if missing:
        raise KeyError(f"Unknown episode indices: {missing}")

    nodes_by_task = load_task_nodes(dataset_dir)
    locator, segmenter = build_models(args.locator, args.segmenter)
    for episode_index in selected:
        frame_rgb, stored_episode, task_index = read_first_rgb(parquet_paths[episode_index])
        if stored_episode != episode_index:
            raise ValueError(
                f"Parquet episode mismatch: filename={episode_index}, row={stored_episode}"
            )
        node_names = nodes_by_task.get(task_index)
        if not node_names:
            raise KeyError(f"No taskstructure nodes for task_index={task_index}")

        output_path = cache_path(output_dir, episode_index)
        expected_metadata = {
            "version": CACHE_VERSION,
            "locator": args.locator,
            "segmenter": args.segmenter,
            "scale": float(args.scale),
            "height": int(frame_rgb.shape[0]),
            "width": int(frame_rgb.shape[1]),
            "frame_flipped": True,
        }
        if output_path.exists() and not args.overwrite:
            existing = load_cache(output_path)
            if (
                cache_metadata(existing) != expected_metadata
                or existing["node_names"].tolist() != node_names
            ):
                raise RuntimeError(
                    f"Cache configuration mismatch: {output_path}. "
                    "Use --overwrite or a different --cache-dir."
                )
            print(f"cached: {output_path}")
            continue

        point_groups = []
        height, width = frame_rgb.shape[:2]
        for node_name in node_names:
            result = locator.inference(
                text=node_name,
                image=Image.fromarray(frame_rgb),
                resize_scale=args.scale,
            )
            normalized = np.asarray(result.get("points") or [], dtype=np.float32).reshape(-1, 2)
            if not len(normalized):
                raise RuntimeError(
                    f"Locator found no point for episode={episode_index}, node={node_name!r}"
                )
            normalized[:, 0] = np.clip(normalized[:, 0] / 1000.0 * width, 0, width - 1)
            normalized[:, 1] = np.clip(normalized[:, 1] / 1000.0 * height, 0, height - 1)
            point_groups.append(normalized.tolist())

        masks = segmenter.predict(frame_rgb, points=point_groups, anchor_frame=True)
        if len(masks) != len(node_names):
            raise RuntimeError(
                f"Segmenter returned {len(masks)} masks for {len(node_names)} nodes "
                f"in episode={episode_index}"
            )
        masks_array = np.stack([np.asarray(mask, dtype=bool) for mask in masks])
        points_array, point_counts = pack_points(point_groups)
        data = {
            "episode_index": np.asarray(episode_index, dtype=np.int64),
            "task_index": np.asarray(task_index, dtype=np.int64),
            "node_names": np.asarray(node_names),
            "initial_points_xy": points_array,
            "initial_point_counts": point_counts,
            "initial_masks": masks_array,
            "points_xy": points_array.copy(),
            "point_counts": point_counts.copy(),
            "masks": masks_array.copy(),
            "metadata_json": np.asarray(json.dumps(expected_metadata, sort_keys=True)),
        }
        atomic_save_cache(output_path, data)
        print(f"initialized: {output_path}")


class CacheEditor:
    def __init__(self, dataset_dir: Path, cache_dir: Path):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.parquet_paths = episode_parquet_paths(dataset_dir)
        self.cache_paths = {
            int(path.stem.rsplit("_", 1)[1]): path
            for path in sorted(cache_dir.glob("episode_*.npz"))
        }
        if not self.cache_paths:
            raise RuntimeError(f"No episode caches found under {cache_dir}")

        self.episodes_by_task: dict[int, list[int]] = {}
        segmenters = set()
        for episode_index, cache_file in self.cache_paths.items():
            data = load_cache(cache_file)
            task_index = int(data["task_index"])
            self.episodes_by_task.setdefault(task_index, []).append(episode_index)
            segmenters.add(cache_metadata(data)["segmenter"])
        for episodes in self.episodes_by_task.values():
            episodes.sort()
        if len(segmenters) != 1:
            raise RuntimeError(f"Mixed segmenter configurations in cache: {sorted(segmenters)}")
        self.segmenter_name = segmenters.pop()
        self.segmenter = build_segmenter_for_editor(self.segmenter_name)
        self.model_lock = threading.Lock()

    @lru_cache(maxsize=64)
    def frame(self, episode_index: int) -> np.ndarray:
        frame_rgb, stored_episode, _ = read_first_rgb(self.parquet_paths[episode_index])
        if stored_episode != episode_index:
            raise ValueError(f"Episode mismatch for {self.parquet_paths[episode_index]}")
        return frame_rgb

    def episode_info(self, episode_index: int) -> dict[str, Any]:
        data = load_cache(self.cache_paths[episode_index])
        return {
            "episode_index": episode_index,
            "task_index": int(data["task_index"]),
            "node_names": data["node_names"].tolist(),
            "points": [
                data["points_xy"][index, : int(count)].tolist()
                for index, count in enumerate(data["point_counts"])
            ],
            "height": int(data["masks"].shape[1]),
            "width": int(data["masks"].shape[2]),
        }

    def overlay(self, episode_index: int, focused_node: int | None = None) -> np.ndarray:
        data = load_cache(self.cache_paths[episode_index])
        image = self.frame(episode_index).copy()
        node_indices = (
            [focused_node]
            if focused_node is not None
            else list(range(len(data["node_names"])))
        )
        for node_index in node_indices:
            color = np.asarray(COLORS[node_index % len(COLORS)], dtype=np.uint8)
            mask = np.asarray(data["masks"][node_index], dtype=bool)
            image[mask] = (
                image[mask].astype(np.float32) * 0.6 + color.astype(np.float32) * 0.4
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color.tolist(), 1, cv2.LINE_AA)
            count = int(data["point_counts"][node_index])
            for x, y in data["points_xy"][node_index, :count]:
                cv2.circle(
                    image,
                    (int(round(x)), int(round(y))),
                    5,
                    color.tolist(),
                    -1,
                    cv2.LINE_AA,
                )
        return image

    def apply_point(
        self, episode_index: int, node_index: int, point: list[int]
    ) -> dict[str, Any]:
        if episode_index not in self.cache_paths:
            raise KeyError(f"Unknown cached episode: {episode_index}")
        data = load_cache(self.cache_paths[episode_index])
        if not 0 <= node_index < len(data["node_names"]):
            raise IndexError(f"Invalid node_index={node_index}")

        frame_rgb = self.frame(episode_index)
        height, width = frame_rgb.shape[:2]
        point = [
            min(max(int(round(point[0])), 0), width - 1),
            min(max(int(round(point[1])), 0), height - 1),
        ]
        with self.model_lock:
            masks = self.segmenter.predict(
                frame_rgb, points=[[point]], anchor_frame=True
            )
        if len(masks) != 1:
            raise RuntimeError(f"Segmenter returned {len(masks)} masks; expected 1")

        data["points_xy"][node_index] = np.nan
        data["points_xy"][node_index, 0] = np.asarray(point, dtype=np.float32)
        data["point_counts"][node_index] = 1
        data["masks"][node_index] = np.asarray(masks[0], dtype=bool)
        atomic_save_cache(self.cache_paths[episode_index], data)
        return {
            "message": f"已保存 N{node_index}: {data['node_names'][node_index]}",
            "point": point,
        }

    def restore_node(self, episode_index: int, node_index: int) -> dict[str, str]:
        if episode_index not in self.cache_paths:
            raise KeyError(f"Unknown cached episode: {episode_index}")
        data = load_cache(self.cache_paths[episode_index])
        if not 0 <= node_index < len(data["node_names"]):
            raise IndexError(f"Invalid node_index={node_index}")
        data["points_xy"][node_index] = data["initial_points_xy"][node_index]
        data["point_counts"][node_index] = data["initial_point_counts"][node_index]
        data["masks"][node_index] = data["initial_masks"][node_index]
        atomic_save_cache(self.cache_paths[episode_index], data)
        return {"message": f"已恢复 N{node_index}: {data['node_names'][node_index]}"}


def build_segmenter_for_editor(segmenter_name: str) -> Any:
    if segmenter_name == "sam2":
        from src.module.node_segmenter import NodeSegmenterSAM2

        return NodeSegmenterSAM2()
    if segmenter_name == "sam3":
        from src.module.node_segmenter import NodeSegmenter

        return NodeSegmenter()
    raise ValueError(f"Unsupported cached segmenter: {segmenter_name}")


def encode_jpeg(image_rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    if not ok:
        raise RuntimeError("Failed to encode image")
    return encoded.tobytes()


def serve_editor(args: argparse.Namespace) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, Response

    dataset_dir, cache_dir = resolve_paths(args)
    editor = CacheEditor(dataset_dir, cache_dir)
    app = FastAPI(title="Node Initialization Cache Editor")

    def checked_episode(episode_index: int) -> None:
        if episode_index not in editor.cache_paths:
            raise HTTPException(status_code=404, detail="Episode cache not found")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/static/style.css")
    def style():
        return Response(STYLE_CSS, media_type="text/css")

    @app.get("/static/app.js")
    def script():
        return Response(APP_JS, media_type="application/javascript")

    @app.get("/api/tasks")
    def tasks():
        return {
            "tasks": [
                {
                    "task_index": task_index,
                    "episode_count": len(episodes),
                    "page_count": (len(episodes) + 15) // 16,
                }
                for task_index, episodes in sorted(editor.episodes_by_task.items())
            ]
        }

    @app.get("/api/tasks/{task_index}/episodes")
    def task_episodes(task_index: int, page: int = Query(0, ge=0)):
        episodes = editor.episodes_by_task.get(task_index)
        if episodes is None:
            raise HTTPException(status_code=404, detail="Task not found")
        page_count = max((len(episodes) + 15) // 16, 1)
        page = min(page, page_count - 1)
        return {
            "task_index": task_index,
            "page": page,
            "page_count": page_count,
            "episodes": episodes[page * 16 : (page + 1) * 16],
        }

    @app.get("/api/episodes/{episode_index}")
    def episode_info(episode_index: int):
        checked_episode(episode_index)
        return editor.episode_info(episode_index)

    @app.get("/api/episodes/{episode_index}/preview")
    def episode_preview(episode_index: int):
        checked_episode(episode_index)
        return Response(
            encode_jpeg(editor.overlay(episode_index)),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/episodes/{episode_index}/overlay")
    def episode_overlay(episode_index: int, node_index: int = Query(..., ge=0)):
        checked_episode(episode_index)
        data = load_cache(editor.cache_paths[episode_index])
        if node_index >= len(data["node_names"]):
            raise HTTPException(status_code=404, detail="Node not found")
        return Response(
            encode_jpeg(editor.overlay(episode_index, focused_node=node_index)),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/episodes/{episode_index}/nodes/{node_index}/point")
    def update_point(episode_index: int, node_index: int, request: PointRequest):
        try:
            return editor.apply_point(episode_index, node_index, [request.x, request.y])
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/episodes/{episode_index}/nodes/{node_index}/restore")
    def restore_node(episode_index: int, node_index: int):
        try:
            return editor.restore_node(episode_index, node_index)
        except (KeyError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    uvicorn.run(app, host=args.host, port=args.port)

def main() -> None:
    args = parse_args()
    if args.command == "initialize":
        initialize_cache(args)
    else:
        serve_editor(args)


if __name__ == "__main__":
    main()
