#!/usr/bin/env python
"""Serve a local web UI for annotating LeRobot episode subtask boundaries."""

from __future__ import annotations

import argparse
import copy
import fcntl
import io
import json
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel


AGENT_IMAGE_KEYS = ("image", "observation.images.image")
ANNOTATIONS_FILE = "subtask_annotations.jsonl"
SUBTASK_FEATURE = {"dtype": "int64", "shape": [1], "names": None}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_dataset_lock(meta_dir: Path):
    lock_path = meta_dir / ".subtask_annotator.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def encode_jpeg(image: Image.Image | np.ndarray) -> bytes:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


class AnnotationRequest(BaseModel):
    boundaries: list[int]


class DatasetStore:
    def __init__(self, root: Path, preview_fps: float, cache_episodes: int) -> None:
        self.root = root.resolve()
        self.meta_dir = self.root / "meta"
        self.info_path = self.meta_dir / "info.json"
        self.annotations_path = self.meta_dir / ANNOTATIONS_FILE
        self.info = load_json(self.info_path)
        if preview_fps <= 0:
            raise ValueError("preview_fps must be greater than zero")
        self.tasks = load_jsonl(self.meta_dir / "tasks.jsonl")
        self.episodes = load_jsonl(self.meta_dir / "episodes.jsonl")
        self.preview_fps = preview_fps
        self.cache_episodes = max(1, cache_episodes)
        self.lock = threading.RLock()
        self.frame_cache: OrderedDict[int, dict[int, bytes]] = OrderedDict()
        self.image_payload_cache: dict[int, list[bytes]] = {}
        self.episodes_by_index = {int(row["episode_index"]): row for row in self.episodes}
        self.tasks_by_index = {int(row["task_index"]): row for row in self.tasks}
        self.task_index_by_text = {row["task"]: int(row["task_index"]) for row in self.tasks}
        self.task_to_episodes = self._index_task_episodes()
        self.image_key, self.image_dtype = self._agent_image_feature()
        self.annotations = self._load_annotations()
        self.annotated_episodes = {
            episode_index
            for episode_index in self.episodes_by_index
            if "subtask_id" in pq.read_schema(self.episode_path(episode_index)).names
        }

    def _index_task_episodes(self) -> dict[int, list[int]]:
        output = {index: [] for index in self.tasks_by_index}
        for episode in sorted(self.episodes, key=lambda row: int(row["episode_index"])):
            indices = {
                self.task_index_by_text[text]
                for text in episode.get("tasks", [])
                if text in self.task_index_by_text
            }
            if len(indices) != 1:
                raise ValueError(
                    f"Episode {episode['episode_index']} must map to exactly one task, got {sorted(indices)}"
                )
            output.setdefault(indices.pop(), []).append(int(episode["episode_index"]))
        return output

    def _agent_image_feature(self) -> tuple[str, str]:
        features = self.info.get("features", {})
        for key in AGENT_IMAGE_KEYS:
            dtype = features.get(key, {}).get("dtype")
            if dtype in {"image", "video"}:
                return key, dtype
        raise ValueError(
            "Dataset has no agent-view RGB feature; expected 'image' or 'observation.images.image'"
        )

    def _load_annotations(self) -> dict[int, dict[str, Any]]:
        if not self.annotations_path.is_file():
            return {}
        return {
            int(row["episode_index"]): row
            for row in load_jsonl(self.annotations_path)
        }

    def episode_path(self, episode_index: int) -> Path:
        chunks_size = int(self.info["chunks_size"])
        relative = self.info["data_path"].format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        return self.root / relative

    def video_path(self, episode_index: int) -> Path:
        chunks_size = int(self.info["chunks_size"])
        relative = self.info["video_path"].format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            video_key=self.image_key,
        )
        return self.root / relative

    def task_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "task_index": task_index,
                "task": self.tasks_by_index[task_index]["task"],
                "episodes": len(episode_indices),
                "annotated": sum(index in self.annotated_episodes for index in episode_indices),
            }
            for task_index, episode_indices in sorted(self.task_to_episodes.items())
        ]

    def episode_rows(self, task_index: int) -> list[dict[str, Any]]:
        if task_index not in self.tasks_by_index:
            raise KeyError(f"Unknown task_index: {task_index}")
        return [
            {
                "episode_index": episode_index,
                "length": int(self.episodes_by_index[episode_index]["length"]),
                "annotated": episode_index in self.annotated_episodes,
                "boundaries": self.boundaries(episode_index),
            }
            for episode_index in self.task_to_episodes.get(task_index, [])
        ]

    def episode_detail(self, episode_index: int) -> dict[str, Any]:
        episode = self._episode(episode_index)
        fps = float(self.info.get("fps", 10))
        length = int(episode["length"])
        return {
            "episode_index": episode_index,
            "length": length,
            "fps": fps,
            "preview_stride": max(1, round(fps / self.preview_fps)),
            "boundaries": self.boundaries(episode_index),
            "segments": segments_from_boundaries(self.boundaries(episode_index), length),
            "annotated": episode_index in self.annotated_episodes,
        }

    def boundaries(self, episode_index: int) -> list[int]:
        path = self.episode_path(episode_index)
        if "subtask_id" in pq.read_schema(path).names:
            values = pq.read_table(path, columns=["subtask_id"])["subtask_id"].to_pylist()
            return [index for index in range(1, len(values)) if values[index] != values[index - 1]]
        if episode_index in self.annotations:
            return list(self.annotations[episode_index]["boundaries"])
        return []

    def frame(self, episode_index: int, frame_index: int) -> bytes:
        episode = self._episode(episode_index)
        length = int(episode["length"])
        if not 0 <= frame_index < length:
            raise IndexError(f"frame_index must be in [0, {length - 1}]")
        with self.lock:
            frames = self.frame_cache.setdefault(episode_index, {})
            if frame_index not in frames:
                frames[frame_index] = self._load_frame(episode_index, frame_index, length)
            self.frame_cache.move_to_end(episode_index)
            while len(self.frame_cache) > self.cache_episodes:
                evicted, _ = self.frame_cache.popitem(last=False)
                self.image_payload_cache.pop(evicted, None)
            return frames[frame_index]

    def _load_frame(self, episode_index: int, frame_index: int, expected_length: int) -> bytes:
        if self.image_dtype == "image":
            payloads = self.image_payload_cache.get(episode_index)
            if payloads is None:
                column = pq.read_table(
                    self.episode_path(episode_index), columns=[self.image_key]
                )[self.image_key]
                payloads = [
                    item.get("bytes") if isinstance(item, dict) else None
                    for item in column.to_pylist()
                ]
                if len(payloads) != expected_length or any(payload is None for payload in payloads):
                    raise ValueError(f"Episode {episode_index} contains invalid embedded RGB frames")
                self.image_payload_cache[episode_index] = payloads
            with Image.open(io.BytesIO(payloads[frame_index])) as image:
                return encode_jpeg(image)

        path = self.video_path(episode_index)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open episode video: {path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError(f"Unable to decode episode {episode_index} frame {frame_index}")
        return encode_jpeg(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def save_annotation(self, episode_index: int, boundaries: list[int]) -> dict[str, Any]:
        episode = self._episode(episode_index)
        length = int(episode["length"])
        validated = validate_boundaries(boundaries, length)
        task_index = next(
            self.task_index_by_text[text]
            for text in episode["tasks"]
            if text in self.task_index_by_text
        )
        subtask_ids = np.searchsorted(validated, np.arange(length), side="right") + 1
        with self.lock, exclusive_dataset_lock(self.meta_dir):
            self.annotations = self._load_annotations()
            self._write_episode_subtask_ids(episode_index, subtask_ids)
            self.annotated_episodes.add(episode_index)
            self._ensure_info_feature()
            self.annotations[episode_index] = {
                "episode_index": episode_index,
                "task_index": task_index,
                "length": length,
                "boundaries": validated,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            rows = [self.annotations[index] for index in sorted(self.annotations)]
            atomic_write_text(
                self.annotations_path,
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            )
        return self.episode_detail(episode_index)

    def _write_episode_subtask_ids(self, episode_index: int, values: np.ndarray) -> None:
        path = self.episode_path(episode_index)
        source = pq.ParquetFile(path)
        if source.metadata.num_rows != len(values):
            raise ValueError(
                f"Episode {episode_index} parquet has {source.metadata.num_rows} rows, expected {len(values)}"
            )
        compression = source.metadata.row_group(0).column(0).compression.lower()
        handle, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
        )
        os.close(handle)
        writer = None
        offset = 0
        try:
            for row_group in range(source.num_row_groups):
                table = source.read_row_group(row_group)
                length = table.num_rows
                column = pa.array(values[offset : offset + length], type=pa.int64())
                if "subtask_id" in table.column_names:
                    position = table.column_names.index("subtask_id")
                    table = table.set_column(position, "subtask_id", column)
                else:
                    table = table.append_column("subtask_id", column)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression=compression)
                writer.write_table(table, row_group_size=length)
                offset += length
            if writer is None:
                raise ValueError(f"Episode {episode_index} parquet has no row groups")
            writer.close()
            writer = None
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
        finally:
            if writer is not None:
                writer.close()
            Path(temporary).unlink(missing_ok=True)

    def _ensure_info_feature(self) -> None:
        if self.info.setdefault("features", {}).get("subtask_id") == SUBTASK_FEATURE:
            return
        updated_info = copy.deepcopy(self.info)
        updated_info.setdefault("features", {})["subtask_id"] = SUBTASK_FEATURE
        atomic_write_text(
            self.info_path, json.dumps(updated_info, ensure_ascii=False, indent=2) + "\n"
        )
        self.info = updated_info

    def _episode(self, episode_index: int) -> dict[str, Any]:
        if episode_index not in self.episodes_by_index:
            raise KeyError(f"Unknown episode_index: {episode_index}")
        return self.episodes_by_index[episode_index]


def validate_boundaries(boundaries: list[int], length: int) -> list[int]:
    if boundaries != sorted(set(boundaries)):
        raise ValueError("boundaries must be unique and sorted")
    if any(boundary <= 0 or boundary >= length for boundary in boundaries):
        raise ValueError(f"boundaries must lie strictly inside the episode: 0 < boundary < {length}")
    return boundaries


def segments_from_boundaries(boundaries: list[int], length: int) -> list[dict[str, int]]:
    starts = [0, *boundaries]
    ends = [boundary - 1 for boundary in boundaries] + [length - 1]
    return [
        {"subtask_id": index + 1, "start": start, "end": end}
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def create_app(dataset: Path | str, preview_fps: float = 5, cache_episodes: int = 2) -> FastAPI:
    store = DatasetStore(Path(dataset), preview_fps, cache_episodes)
    app = FastAPI(title="LeRobot Subtask Annotator")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/tasks")
    def tasks() -> list[dict[str, Any]]:
        return store.task_rows()

    @app.get("/api/tasks/{task_index}/episodes")
    def episodes(task_index: int) -> list[dict[str, Any]]:
        try:
            return store.episode_rows(task_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/episodes/{episode_index}")
    def episode(episode_index: int) -> dict[str, Any]:
        try:
            return store.episode_detail(episode_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/episodes/{episode_index}/frames/{frame_index}")
    def frame(episode_index: int, frame_index: int) -> Response:
        try:
            return Response(store.frame(episode_index, frame_index), media_type="image/jpeg")
        except (KeyError, IndexError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.post("/api/episodes/{episode_index}/annotation")
    def annotate(episode_index: int, request: AnnotationRequest) -> dict[str, Any]:
        try:
            return store.save_annotation(episode_index, request.boundaries)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeRobot 子任务标注</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#e8edf5;background:#0f131a}*{box-sizing:border-box}
body{margin:0}header{padding:14px 22px;background:#171d27;border-bottom:1px solid #2b3442;display:flex;gap:20px;align-items:center}
h1{font-size:18px;margin:0}select,button,input{font:inherit}select,button{background:#222b38;color:#eef3fa;border:1px solid #3b4758;border-radius:7px;padding:8px 11px}
button{cursor:pointer}button:hover{background:#2d394a}.layout{display:grid;grid-template-columns:280px minmax(500px,1fr) 300px;height:calc(100vh - 59px)}
aside,.right{padding:14px;overflow:auto;background:#141a23}.right{border-left:1px solid #2b3442}.episodes{border-right:1px solid #2b3442}
.episode{display:flex;width:100%;justify-content:space-between;margin:5px 0;text-align:left}.done{color:#77d49b}.pending{color:#e5b86b}
main{padding:18px;display:flex;flex-direction:column;align-items:center;overflow:auto}.viewer{width:min(100%,900px);background:#080a0e;border-radius:10px;overflow:hidden;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center}
#frame{max-width:100%;max-height:100%;image-rendering:auto}.timeline{width:min(100%,900px);margin-top:14px}.timeline input{width:100%}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
.frame-label{font-variant-numeric:tabular-nums;min-width:160px}.boundary{display:flex;justify-content:space-between;align-items:center;background:#202936;border-radius:7px;padding:8px;margin:6px 0}
.segment{font-size:13px;color:#b9c4d2;padding:5px 0;border-bottom:1px solid #293342}.primary{background:#246bce;border-color:#347be0}.danger{background:#7b2930}.message{min-height:24px;color:#7ed79f;margin-top:10px}
</style></head><body>
<header><h1>LeRobot 子任务边界标注</h1><label>Task <select id="task"></select></label><span id="taskProgress"></span></header>
<div class="layout"><aside class="episodes"><b>Episodes</b><div id="episodes"></div></aside>
<main><div class="viewer"><img id="frame" alt="episode frame"></div><div class="timeline"><input id="slider" type="range" min="0" value="0" step="1">
<div class="controls"><button id="play">▶ 播放</button><button id="prev">← 原始帧</button><button id="next">原始帧 →</button><span class="frame-label" id="frameLabel"></span><button class="primary" id="add">在当前帧添加边界</button></div></div></main>
<aside class="right"><b>边界（该帧开始新 subtask）</b><div id="boundaries"></div><h3>全帧区间</h3><div id="segments"></div><button class="primary" id="save">保存并写入 Parquet</button><div class="message" id="message"></div></aside></div>
<script>
const $=id=>document.getElementById(id);let tasks=[],episodes=[],detail=null,current=0,boundaries=[],savedBoundaries=[],timer=null,activeTask=null;
function isDirty(){return JSON.stringify(boundaries)!==JSON.stringify(savedBoundaries)}
async function json(url,options){const r=await fetch(url,options);const body=await r.json();if(!r.ok)throw new Error(body.detail||r.statusText);return body}
async function init(){tasks=await json('/api/tasks');$('task').innerHTML='';for(const t of tasks){const o=document.createElement('option');o.value=t.task_index;o.textContent=`${t.task_index}: ${t.task}`;$('task').appendChild(o)}$('task').onchange=loadTask;if(tasks.length)await loadTask()}
async function loadTask(){const id=Number($('task').value);if(detail&&isDirty()&&!confirm('当前 episode 有未保存边界，确定放弃吗？')){$('task').value=activeTask;return}stop();detail=null;activeTask=id;episodes=await json(`/api/tasks/${id}/episodes`);renderEpisodes();const t=tasks.find(x=>x.task_index===id);$('taskProgress').textContent=`${t.annotated}/${t.episodes} 已标注`;if(episodes.length)await selectEpisode(episodes[0].episode_index)}
function renderEpisodes(){const box=$('episodes');box.innerHTML='';for(const ep of episodes){const b=document.createElement('button');b.className='episode';b.onclick=()=>selectEpisode(ep.episode_index);const left=document.createElement('span');left.textContent=`Episode ${ep.episode_index}`;const right=document.createElement('span');right.className=ep.annotated?'done':'pending';right.textContent=ep.annotated?'✓':'待标注';b.append(left,right);box.appendChild(b)}}
async function selectEpisode(id){if(detail&&detail.episode_index!==id&&isDirty()&&!confirm('当前 episode 有未保存边界，确定放弃吗？'))return;stop();detail=await json(`/api/episodes/${id}`);current=0;boundaries=[...detail.boundaries];savedBoundaries=[...boundaries];$('slider').max=detail.length-1;$('slider').value=0;showFrame();renderBoundaries();$('message').textContent=''}
function showFrame(){if(!detail)return;$('slider').value=current;$('frame').src=`/api/episodes/${detail.episode_index}/frames/${current}`;$('frameLabel').textContent=`原始帧 ${current} / ${detail.length-1} · subtask ${subtaskAt(current)}`}
function subtaskAt(frame){return 1+boundaries.filter(x=>x<=frame).length}function move(delta){current=Math.max(0,Math.min(detail.length-1,current+delta));showFrame()}
function play(){if(timer){stop();return}$('play').textContent='⏸ 暂停';timer=setInterval(()=>{if(current>=detail.length-1){stop();return}move(detail.preview_stride)},1000/detail.fps*detail.preview_stride)}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ 播放'}
function renderBoundaries(){const box=$('boundaries');box.innerHTML='';for(const value of boundaries){const row=document.createElement('div');row.className='boundary';row.innerHTML=`<span>Frame ${value} → subtask ${1+boundaries.indexOf(value)+1}</span>`;const del=document.createElement('button');del.className='danger';del.textContent='删除';del.onclick=()=>{boundaries=boundaries.filter(x=>x!==value);renderBoundaries();showFrame()};row.appendChild(del);box.appendChild(row)}renderSegments()}
function renderSegments(){if(!detail)return;const starts=[0,...boundaries],ends=[...boundaries.map(x=>x-1),detail.length-1];$('segments').innerHTML=starts.map((s,i)=>`<div class="segment">subtask ${i+1}: frame ${s}–${ends[i]}</div>`).join('')}
async function save(){try{const result=await json(`/api/episodes/${detail.episode_index}/annotation`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({boundaries})});detail=result;savedBoundaries=[...boundaries];$('message').textContent='已原子写入 Parquet';tasks=await json('/api/tasks');episodes=await json(`/api/tasks/${$('task').value}/episodes`);renderEpisodes();const t=tasks.find(x=>x.task_index===Number($('task').value));$('taskProgress').textContent=`${t.annotated}/${t.episodes} 已标注`}catch(e){$('message').textContent=e.message}}
$('slider').oninput=e=>{stop();current=Number(e.target.value);showFrame()};$('play').onclick=play;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('add').onclick=()=>{if(current<=0||current>=detail.length)return;if(!boundaries.includes(current)){boundaries.push(current);boundaries.sort((a,b)=>a-b);renderBoundaries();showFrame()}};$('save').onclick=save;
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key===' '){e.preventDefault();play()}});window.onbeforeunload=()=>isDirty()?'有未保存标注':undefined;init().catch(e=>$('message').textContent=e.message);
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Local LeRobot v2 dataset directory to modify.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--preview-fps", type=float, default=5)
    parser.add_argument("--cache-episodes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.dataset, args.preview_fps, args.cache_episodes)
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
