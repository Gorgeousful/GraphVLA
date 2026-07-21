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
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel


AGENT_IMAGE_KEYS = ("image", "observation.images.image")
ANNOTATIONS_FILE = "subtask_annotations.jsonl"
SUBTASK_FEATURE = {"dtype": "int64", "shape": [1], "names": None}
IS_COMPLETE_FEATURE = {"dtype": "bool", "shape": [1], "names": None}


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


class EpisodeAnnotationRequest(AnnotationRequest):
    episode_index: int


class BatchAnnotationRequest(BaseModel):
    annotations: list[EpisodeAnnotationRequest]


class DatasetStore:
    def __init__(
        self, root: Path, preview_fps: float, cache_episodes: int, completion_frames: int
    ) -> None:
        self.root = root.resolve()
        self.meta_dir = self.root / "meta"
        self.info_path = self.meta_dir / "info.json"
        self.annotations_path = self.meta_dir / ANNOTATIONS_FILE
        self.info = load_json(self.info_path)
        if preview_fps <= 0:
            raise ValueError("preview_fps must be greater than zero")
        if completion_frames <= 0:
            raise ValueError("completion_frames must be greater than zero")
        self.tasks = load_jsonl(self.meta_dir / "tasks.jsonl")
        self.episodes = load_jsonl(self.meta_dir / "episodes.jsonl")
        self.preview_fps = preview_fps
        self.completion_frames = completion_frames
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
        schemas = {
            episode_index: set(pq.read_schema(self.episode_path(episode_index)).names)
            for episode_index in self.episodes_by_index
        }
        self.annotated_episodes = {
            episode_index for episode_index, names in schemas.items() if "subtask_id" in names
        }
        self.completion_column_episodes = {
            episode_index for episode_index, names in schemas.items() if "is_complete" in names
        }
        self.completion_episodes = {
            episode_index
            for episode_index in self.completion_column_episodes
            if self.annotations.get(episode_index, {}).get("completion_frames")
            == self.completion_frames
        }
        features = self.info.get("features", {})
        if self.annotated_episodes and features.get("subtask_id") != SUBTASK_FEATURE:
            self._ensure_info_features(include_completion=False)
        if (
            len(self.completion_column_episodes) == len(self.episodes_by_index)
            and features.get("is_complete") != IS_COMPLETE_FEATURE
        ):
            self._ensure_info_features(include_completion=True)

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
                "completion_annotated": episode_index in self.completion_episodes,
                "boundaries": self.boundaries(episode_index),
            }
            for episode_index in self.task_to_episodes.get(task_index, [])
        ]

    def completion_pending_rows(self) -> list[dict[str, Any]]:
        return [
            {"episode_index": episode_index, "boundaries": self.boundaries(episode_index)}
            for episode_index in sorted(self.annotated_episodes - self.completion_episodes)
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
            "completion_annotated": episode_index in self.completion_episodes,
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
        result = self.save_annotations([(episode_index, boundaries)])
        if result.get("failures"):
            raise RuntimeError(result["failures"][0]["error"])
        return self.episode_detail(episode_index)

    def save_annotations(self, requests: list[tuple[int, list[int]]]) -> dict[str, Any]:
        episode_indices = [episode_index for episode_index, _ in requests]
        if len(episode_indices) != len(set(episode_indices)):
            raise ValueError("each episode_index may appear only once")

        prepared = []
        for episode_index, boundaries in requests:
            episode = self._episode(episode_index)
            length = int(episode["length"])
            validated = validate_boundaries(boundaries, length)
            task_index = next(
                self.task_index_by_text[text]
                for text in episode["tasks"]
                if text in self.task_index_by_text
            )
            subtask_ids = np.searchsorted(validated, np.arange(length), side="right") + 1
            is_complete = completion_mask(validated, length, self.completion_frames)
            prepared.append(
                (episode_index, task_index, length, validated, subtask_ids, is_complete)
            )

        saved = []
        failures = []
        metadata_errors = []
        with self.lock, exclusive_dataset_lock(self.meta_dir):
            self.annotations = self._load_annotations()
            timestamp = datetime.now(timezone.utc).isoformat()
            for episode_index, task_index, length, boundaries, subtask_ids, is_complete in prepared:
                try:
                    self._write_episode_annotations(episode_index, subtask_ids, is_complete)
                except Exception as error:
                    failures.append({"episode_index": episode_index, "error": str(error)})
                    continue
                saved.append(episode_index)
                self.annotated_episodes.add(episode_index)
                self.completion_column_episodes.add(episode_index)
                self.completion_episodes.add(episode_index)
                self.annotations[episode_index] = {
                    "episode_index": episode_index,
                    "task_index": task_index,
                    "length": length,
                    "boundaries": boundaries,
                    "completion_frames": self.completion_frames,
                    "updated_at": timestamp,
                }
            if saved:
                try:
                    self._ensure_info_features(
                        include_completion=len(self.completion_column_episodes)
                        == len(self.episodes_by_index)
                    )
                except Exception as error:
                    metadata_errors.append({"path": str(self.info_path), "error": str(error)})
                rows = [self.annotations[index] for index in sorted(self.annotations)]
                try:
                    atomic_write_text(
                        self.annotations_path,
                        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    )
                except Exception as error:
                    metadata_errors.append(
                        {"path": str(self.annotations_path), "error": str(error)}
                    )
        result: dict[str, Any] = {"saved_episode_indices": saved}
        if failures:
            result["failures"] = failures
        if metadata_errors:
            result["metadata_errors"] = metadata_errors
        return result

    def _write_episode_annotations(
        self, episode_index: int, subtask_ids: np.ndarray, is_complete: np.ndarray
    ) -> None:
        path = self.episode_path(episode_index)
        source = pq.ParquetFile(path)
        if source.metadata.num_rows != len(subtask_ids) or len(subtask_ids) != len(is_complete):
            raise ValueError(
                f"Episode {episode_index} parquet has {source.metadata.num_rows} rows, "
                f"expected {len(subtask_ids)}"
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
                columns = (
                    ("subtask_id", pa.array(subtask_ids[offset : offset + length], type=pa.int64())),
                    ("is_complete", pa.array(is_complete[offset : offset + length], type=pa.bool_())),
                )
                for name, column in columns:
                    if name in table.column_names:
                        table = table.set_column(table.column_names.index(name), name, column)
                    else:
                        table = table.append_column(name, column)
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

    def _ensure_info_features(self, include_completion: bool) -> None:
        required = {"subtask_id": SUBTASK_FEATURE}
        if include_completion:
            required["is_complete"] = IS_COMPLETE_FEATURE
        features = self.info.setdefault("features", {})
        if all(features.get(name) == feature for name, feature in required.items()):
            return
        updated_info = copy.deepcopy(self.info)
        updated_info.setdefault("features", {}).update(required)
        atomic_write_text(
            self.info_path, json.dumps(updated_info, ensure_ascii=False, indent=2) + "\n"
        )
        self.info = updated_info

    def _episode(self, episode_index: int) -> dict[str, Any]:
        if episode_index not in self.episodes_by_index:
            raise KeyError(f"Unknown episode_index: {episode_index}")
        return self.episodes_by_index[episode_index]


def completion_mask(
    boundaries: list[int], length: int, completion_frames: int
) -> np.ndarray:
    if completion_frames <= 0:
        raise ValueError("completion_frames must be greater than zero")
    mask = np.zeros(length, dtype=np.bool_)
    for start, end in zip([0, *boundaries], [*boundaries, length], strict=True):
        mask[max(start, end - completion_frames) : end] = True
    return mask


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


def create_app(
    dataset: Path | str,
    preview_fps: float = 5,
    cache_episodes: int = 2,
    completion_frames: int = 5,
) -> FastAPI:
    store = DatasetStore(Path(dataset), preview_fps, cache_episodes, completion_frames)
    app = FastAPI(title="LeRobot Subtask Annotator")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/tasks")
    def tasks() -> list[dict[str, Any]]:
        return store.task_rows()

    @app.get("/api/completion-pending")
    def completion_pending() -> list[dict[str, Any]]:
        return store.completion_pending_rows()

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

    @app.post("/api/annotations/batch")
    def annotate_batch(request: BatchAnnotationRequest) -> dict[str, Any]:
        try:
            return store.save_annotations(
                [(item.episode_index, item.boundaries) for item in request.annotations]
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/annotations/stream")
    def annotate_stream(request: BatchAnnotationRequest) -> StreamingResponse:
        episode_indices = [item.episode_index for item in request.annotations]
        if len(episode_indices) != len(set(episode_indices)):
            raise HTTPException(status_code=422, detail="each episode_index may appear only once")

        def events():
            total = len(request.annotations)
            saved = 0
            failed = 0
            for completed, item in enumerate(request.annotations, start=1):
                try:
                    result = store.save_annotations([(item.episode_index, item.boundaries)])
                    failures = result.get("failures", [])
                    if failures:
                        failed += 1
                        event = {
                            "episode_index": item.episode_index,
                            "status": "failed",
                            "error": failures[0]["error"],
                            "completed": completed,
                            "total": total,
                        }
                    else:
                        saved += 1
                        event = {
                            "episode_index": item.episode_index,
                            "status": "saved",
                            "completed": completed,
                            "total": total,
                        }
                        if result.get("metadata_errors"):
                            event["metadata_errors"] = result["metadata_errors"]
                except Exception as error:
                    failed += 1
                    event = {
                        "episode_index": item.episode_index,
                        "status": "failed",
                        "error": str(error),
                        "completed": completed,
                        "total": total,
                    }
                yield json.dumps(event, ensure_ascii=False) + "\n"
            yield json.dumps(
                {"status": "done", "saved": saved, "failed": failed, "total": total}
            ) + "\n"

        return StreamingResponse(
            events(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeRobot Subtask Annotator</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#e8edf5;background:#0f131a}*{box-sizing:border-box}
body{margin:0}header{padding:14px 22px;background:#171d27;border-bottom:1px solid #2b3442;display:flex;gap:20px;align-items:center}
h1{font-size:18px;margin:0}select,button,input{font:inherit}select,button{background:#222b38;color:#eef3fa;border:1px solid #3b4758;border-radius:7px;padding:8px 11px}
button{cursor:pointer}button:hover{background:#2d394a}button:disabled{cursor:default;opacity:.55}.layout{display:grid;grid-template-columns:280px minmax(500px,1fr) 320px;height:calc(100vh - 59px)}
aside,.right{padding:14px;overflow:auto;background:#141a23}.right{border-left:1px solid #2b3442}.episodes{border-right:1px solid #2b3442}
.episode{display:flex;width:100%;justify-content:space-between;margin:5px 0;text-align:left}.episode.current{background:#34465f;border-color:#6b8fbd;box-shadow:inset 3px 0 #72a7e8}.episode-status{display:flex;gap:8px;align-items:center}.persisted{color:#77d49b;font-size:12px}.done{color:#77d49b}.pending{color:#e5b86b}.unsaved{color:#ffad55}
main{padding:18px;display:flex;flex-direction:column;align-items:center;overflow:auto}.viewer{width:min(100%,900px);background:#080a0e;border-radius:10px;overflow:hidden;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center}
#frame{max-width:100%;max-height:100%;image-rendering:auto}.timeline{width:min(100%,900px);margin-top:14px}.range-wrap{position:relative;padding-bottom:15px}.range-wrap input{width:100%;margin:0}
.marker-track{position:absolute;left:8px;right:8px;bottom:0;height:12px;pointer-events:none}.marker{position:absolute;top:0;width:3px;height:12px;transform:translateX(-1px);border-radius:2px}.marker.saved{background:#63d68b}.marker.unsaved{background:#ffad55}.marker.removed{background:#ef6b73;opacity:.8}
.legend{display:flex;gap:14px;margin-top:5px;color:#aeb9c7;font-size:12px}.legend span:before{content:"";display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:2px}.legend .saved-key:before{background:#63d68b}.legend .unsaved-key:before{background:#ffad55}.legend .removed-key:before{background:#ef6b73}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}.frame-label{font-variant-numeric:tabular-nums;min-width:190px}
.boundary{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;background:#202936;border-radius:7px;padding:8px;margin:6px 0}.boundary-thumbnail{grid-column:1/-1;width:100%;max-height:220px;object-fit:contain;background:#080a0e;border-radius:5px}.segment{font-size:13px;color:#b9c4d2;padding:5px 0;border-bottom:1px solid #293342}
.primary{background:#246bce;border-color:#347be0}.danger{background:#7b2930}.message{min-height:24px;color:#7ed79f;margin-top:10px}.save-all{width:100%;margin-top:14px}
</style></head><body>
<header><h1>LeRobot Subtask Boundary Annotator</h1><label>Task <select id="task"></select></label><span id="taskProgress"></span></header>
<div class="layout"><aside class="episodes"><b>Episodes</b><div id="episodes"></div></aside>
<main><div class="viewer"><img id="frame" alt="Episode frame"></div><div class="timeline">
<div class="range-wrap"><input id="slider" type="range" min="0" value="0" step="1"><div id="markers" class="marker-track"></div></div>
<div class="legend"><span class="saved-key">Saved boundary</span><span class="unsaved-key">Unsaved boundary</span><span class="removed-key">Pending removal</span></div>
<div class="controls"><button id="play">▶ Play</button><button id="prev">← Frame</button><button id="next">Frame →</button><span class="frame-label" id="frameLabel"></span><button class="primary" id="add">Add Boundary at Current Frame</button></div>
</div></main>
<aside class="right"><b>Boundaries (this frame starts the next subtask)</b><div id="boundaries"></div><h3>Full-frame Segments</h3><div id="segments"></div>
<button class="primary save-all" id="save">Save All Changes (0)</button><div class="message" id="message"></div></aside></div>
<script>
const $=id=>document.getElementById(id);
let tasks=[],episodes=[],detail=null,current=0,boundaries=[],timer=null,activeTask=null;
const drafts=new Map(),savedByEpisode=new Map(),annotatedByEpisode=new Map(),completionByEpisode=new Map();
const edited=new Set(),completionPending=new Set();
async function json(url,options){const response=await fetch(url,options);const body=await response.json();if(!response.ok)throw new Error(body.detail||response.statusText);return body}
function equal(a,b){return JSON.stringify(a||[])===JSON.stringify(b||[])}
function stashCurrent(){if(detail)drafts.set(detail.episode_index,[...boundaries])}
function needsSave(id){return completionPending.has(id)||(edited.has(id)&&!equal(drafts.get(id),savedByEpisode.get(id)))}
function dirtyIds(){return [...new Set([...edited,...completionPending])].filter(needsSave).sort((a,b)=>a-b)}
function rememberEpisode(episode){const id=episode.episode_index;annotatedByEpisode.set(id,episode.annotated);completionByEpisode.set(id,episode.completion_annotated);if(episode.annotated&&!episode.completion_annotated){completionPending.add(id);if(!savedByEpisode.has(id))savedByEpisode.set(id,[...episode.boundaries]);if(!drafts.has(id))drafts.set(id,[...episode.boundaries])}else completionPending.delete(id)}
function rememberCompletionPending(episode){const id=episode.episode_index;annotatedByEpisode.set(id,true);completionByEpisode.set(id,false);completionPending.add(id);savedByEpisode.set(id,[...episode.boundaries]);if(!drafts.has(id))drafts.set(id,[...episode.boundaries])}
async function init(){tasks=await json('/api/tasks');const pending=await json('/api/completion-pending');for(const episode of pending)rememberCompletionPending(episode);$('task').innerHTML='';for(const task of tasks){const option=document.createElement('option');option.value=task.task_index;option.textContent=task.task_index+': '+task.task;$('task').appendChild(option)}$('task').onchange=loadTask;if(tasks.length)await loadTask()}
async function loadTask(){stashCurrent();stop();activeTask=Number($('task').value);episodes=await json('/api/tasks/'+activeTask+'/episodes');for(const episode of episodes)rememberEpisode(episode);renderEpisodes();renderProgress();if(episodes.length)await selectEpisode(episodes[0].episode_index)}
function renderProgress(){const task=tasks.find(item=>item.task_index===activeTask);if(!task)return;$('taskProgress').textContent=task.annotated+'/'+task.episodes+' saved · '+dirtyIds().length+' unsaved';$('save').textContent='Save All Changes ('+dirtyIds().length+')'}
function renderEpisodes(){const box=$('episodes');box.innerHTML='';for(const episode of episodes){const id=episode.episode_index;const button=document.createElement('button');button.className='episode';button.classList.toggle('current',Boolean(detail)&&id===detail.episode_index);button.onclick=()=>selectEpisode(id);const left=document.createElement('span');left.textContent='Episode '+id;const right=document.createElement('span');right.className='episode-status';if(annotatedByEpisode.get(id)){const persisted=document.createElement('span');persisted.className='persisted';persisted.textContent='subtask_id ✓';right.appendChild(persisted)}if(completionByEpisode.get(id)){const complete=document.createElement('span');complete.className='persisted';complete.textContent='is_complete ✓';right.appendChild(complete)}const state=document.createElement('span');if(completionPending.has(id)&&!edited.has(id)){state.className='unsaved';state.textContent='Needs is_complete'}else if(needsSave(id)){state.className='unsaved';state.textContent='Unsaved'}else if(!annotatedByEpisode.get(id)){state.className='pending';state.textContent='Pending'}right.appendChild(state);button.append(left,right);box.appendChild(button)}}
async function selectEpisode(id){stashCurrent();stop();const next=await json('/api/episodes/'+id);detail=next;current=0;savedByEpisode.set(id,[...next.boundaries]);annotatedByEpisode.set(id,next.annotated);completionByEpisode.set(id,next.completion_annotated);if(next.annotated&&!next.completion_annotated)completionPending.add(id);else completionPending.delete(id);if(!drafts.has(id))drafts.set(id,[...next.boundaries]);boundaries=[...drafts.get(id)];$('slider').max=next.length-1;$('slider').value=0;showFrame();renderAnnotation();$('message').textContent='';renderEpisodes();renderProgress()}
function showFrame(){if(!detail)return;$('slider').value=current;$('frame').src='/api/episodes/'+detail.episode_index+'/frames/'+current;$('frameLabel').textContent='Original frame '+current+' / '+(detail.length-1)+' · subtask '+subtaskAt(current)}
function subtaskAt(frame){return 1+boundaries.filter(value=>value<=frame).length}
function move(delta){if(!detail)return;current=Math.max(0,Math.min(detail.length-1,current+delta));showFrame()}
function play(){if(timer){stop();return}$('play').textContent='⏸ Pause';timer=setInterval(()=>{if(current>=detail.length-1){stop();return}move(detail.preview_stride)},1000/detail.fps*detail.preview_stride)}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Play'}
function updateDraft(){const id=detail.episode_index;drafts.set(id,[...boundaries]);if(equal(boundaries,savedByEpisode.get(id)))edited.delete(id);else edited.add(id);renderAnnotation();showFrame();renderEpisodes();renderProgress()}
function renderAnnotation(){renderBoundaries();renderSegments();renderMarkers()}
function renderBoundaries(){const box=$('boundaries');box.innerHTML='';const saved=savedByEpisode.get(detail.episode_index)||[];for(const value of boundaries){const row=document.createElement('div');row.className='boundary';const label=document.createElement('span');label.textContent='Frame '+value+' → subtask '+(boundaries.indexOf(value)+2)+(saved.includes(value)?' (saved)':' (unsaved)');const remove=document.createElement('button');remove.className='danger';remove.textContent='Remove';remove.onclick=()=>{boundaries=boundaries.filter(item=>item!==value);updateDraft()};const thumbnail=document.createElement('img');thumbnail.className='boundary-thumbnail';thumbnail.loading='lazy';thumbnail.alt='Agent-view RGB at frame '+value;thumbnail.src='/api/episodes/'+detail.episode_index+'/frames/'+value;row.append(label,remove,thumbnail);box.appendChild(row)}}
function renderSegments(){const starts=[0,...boundaries],ends=[...boundaries.map(value=>value-1),detail.length-1];$('segments').innerHTML=starts.map((start,index)=>'<div class="segment">subtask '+(index+1)+': frame '+start+'–'+ends[index]+'</div>').join('')}
function renderMarkers(){const saved=savedByEpisode.get(detail.episode_index)||[],currentSet=new Set(boundaries),savedSet=new Set(saved),all=[...new Set([...saved,...boundaries])].sort((a,b)=>a-b),max=Math.max(1,detail.length-1),box=$('markers');box.innerHTML='';for(const value of all){const marker=document.createElement('span');const state=currentSet.has(value)?(savedSet.has(value)?'saved':'unsaved'):'removed';marker.className='marker '+state;marker.style.left=(value/max*100)+'%';marker.title=(state==='saved'?'Saved boundary':state==='unsaved'?'Unsaved boundary':'Saved boundary pending removal')+' at frame '+value;box.appendChild(marker)}}
async function saveAll(){
  stashCurrent();const ids=dirtyIds();
  if(!ids.length){$('message').textContent='There are no unsaved changes.';return}
  const button=$('save');button.disabled=true;button.textContent='Saving 0 / '+ids.length;
  let summary=null,metadataErrorCount=0;
  try{
    const submittedByEpisode=new Map(ids.map(id=>[id,[...(drafts.get(id)||[])]]));
    const annotations=ids.map(id=>({episode_index:id,boundaries:[...submittedByEpisode.get(id)]}));
    const response=await fetch('/api/annotations/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({annotations})});
    if(!response.ok){const body=await response.json();throw new Error(body.detail||response.statusText)}
    if(!response.body)throw new Error('Streaming responses are not supported by this browser.');
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){
      const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});
      const lines=buffer.split('\n');buffer=lines.pop()||'';
      for(const line of lines){
        if(!line.trim())continue;const event=JSON.parse(line);
        if(event.status==='saved'){
          const id=event.episode_index,submitted=submittedByEpisode.get(id)||[];savedByEpisode.set(id,[...submitted]);annotatedByEpisode.set(id,true);completionByEpisode.set(id,true);completionPending.delete(id);
          if(equal(drafts.get(id),submitted))edited.delete(id);else edited.add(id);
          if(detail&&id===detail.episode_index){detail.boundaries=[...submitted];detail.annotated=true;renderAnnotation()}
          metadataErrorCount+=(event.metadata_errors||[]).length;renderEpisodes();renderProgress();button.textContent='Saving '+event.completed+' / '+event.total;
          $('message').textContent='Saved episode '+id+' ('+event.completed+' / '+event.total+').';
        }else if(event.status==='failed'){
          renderEpisodes();renderProgress();button.textContent='Saving '+event.completed+' / '+event.total;
          $('message').textContent='Episode '+event.episode_index+' failed and remains unsaved: '+event.error;
        }else if(event.status==='done')summary=event;
      }
      if(chunk.done)break;
    }
    tasks=await json('/api/tasks');episodes=await json('/api/tasks/'+activeTask+'/episodes');for(const episode of episodes)rememberEpisode(episode);
    renderEpisodes();renderProgress();
    if(summary){$('message').textContent='Saved '+summary.saved+' / '+summary.total+' episode(s).'+(summary.failed?' '+summary.failed+' failed and remain unsaved.':'')+(metadataErrorCount?' Metadata sync reported '+metadataErrorCount+' error(s).':'')}
  }catch(error){$('message').textContent=error.message}finally{button.disabled=false;renderProgress()}
}
$('slider').oninput=event=>{stop();current=Number(event.target.value);showFrame()};$('play').onclick=play;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('add').onclick=()=>{if(!detail||current<=0||current>=detail.length||boundaries.includes(current))return;boundaries.push(current);boundaries.sort((a,b)=>a-b);updateDraft()};$('save').onclick=saveAll;
document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);if(event.key===' '){event.preventDefault();play()}});
window.onbeforeunload=()=>dirtyIds().length?'You have unsaved annotations.':undefined;init().catch(error=>$('message').textContent=error.message);
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Local LeRobot v2 dataset directory to modify.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--preview-fps", type=float, default=5)
    parser.add_argument("--cache-episodes", type=int, default=2)
    parser.add_argument(
        "--completion-frames",
        type=int,
        default=5,
        help="Mark the last N frames of each subtask as is_complete=true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        args.dataset, args.preview_fps, args.cache_episodes, args.completion_frames
    )
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
