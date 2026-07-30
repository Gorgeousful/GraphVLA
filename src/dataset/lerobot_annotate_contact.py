#!/usr/bin/env python
"""Serve a local web UI for annotating LeRobot contact intervals."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from lerobot_annotate_subtasks import (
    DatasetStore as BaseDatasetStore,
    atomic_write_text,
    exclusive_dataset_lock,
)


IS_CONTACT_FEATURE = {"dtype": "bool", "shape": [1], "names": None}
IS_CONTACT_SOFT_FEATURE = {"dtype": "float32", "shape": [1], "names": None}


class ContactRange(BaseModel):
    start: int
    end: int


class AnnotationRequest(BaseModel):
    ranges: list[ContactRange]


class EpisodeAnnotationRequest(AnnotationRequest):
    episode_index: int


class BatchAnnotationRequest(BaseModel):
    annotations: list[EpisodeAnnotationRequest]


class DatasetStore(BaseDatasetStore):
    def __init__(
        self,
        root: Path,
        preview_fps: float,
        cache_episodes: int,
        soft_width: int,
        horizontal_flip: bool,
        preview_scale: float = 0.5,
        save_workers: int = 4,
    ) -> None:
        super().__init__(
            root=root,
            preview_fps=preview_fps,
            cache_episodes=cache_episodes,
            completion_frames=1,
            horizontal_flip=horizontal_flip,
            preview_scale=preview_scale,
            save_workers=save_workers,
        )
        if soft_width < 0:
            raise ValueError("soft_width must be non-negative")
        self.soft_width = soft_width
        self.contact_episodes = {
            episode_index
            for episode_index in self.episodes_by_index
            if "is_contact" in pq.read_schema(self.episode_path(episode_index)).names
        }
        self.contact_soft_episodes = {
            episode_index
            for episode_index in self.episodes_by_index
            if "is_contact_soft" in pq.read_schema(self.episode_path(episode_index)).names
        }
        features = self.info.get("features", {})
        if (
            self.contact_episodes
            and features.get("is_contact") != IS_CONTACT_FEATURE
        ) or (
            self.contact_soft_episodes
            and features.get("is_contact_soft") != IS_CONTACT_SOFT_FEATURE
        ):
            self._ensure_contact_features()

    def task_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "task_index": task_index,
                "task": self.tasks_by_index[task_index]["task"],
                "episodes": len(episode_indices),
                "annotated": sum(
                    index in self.contact_episodes for index in episode_indices
                ),
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
                "annotated": episode_index in self.contact_episodes,
                "ranges": self.contact_ranges(episode_index),
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
            "ranges": self.contact_ranges(episode_index),
            "soft_width": self.soft_width,
            "annotated": episode_index in self.contact_episodes,
        }

    def contact_ranges(self, episode_index: int) -> list[dict[str, int]]:
        path = self.episode_path(episode_index)
        if "is_contact" in pq.read_schema(path).names:
            values = pq.read_table(path, columns=["is_contact"])["is_contact"].to_pylist()
            return true_ranges(values)
        return []

    def save_annotation(
        self, episode_index: int, ranges: list[dict[str, int]]
    ) -> dict[str, Any]:
        result = self.save_annotations([(episode_index, ranges)])
        if result.get("failures"):
            raise RuntimeError(result["failures"][0]["error"])
        return self.episode_detail(episode_index)

    def save_annotations(
        self, requests: list[tuple[int, list[dict[str, int]]]]
    ) -> dict[str, Any]:
        events = list(self.save_annotation_events(requests))
        saved_events = [event for event in events if event["status"] == "saved"]
        result: dict[str, Any] = {
            "saved_episode_indices": [event["episode_index"] for event in saved_events],
            "changed_episode_indices": [
                event["episode_index"] for event in saved_events if event["changed"]
            ],
            "skipped_episode_indices": [
                event["episode_index"] for event in saved_events if not event["changed"]
            ],
        }
        failures = [event for event in events if event["status"] == "failed"]
        metadata_errors = [
            error
            for event in events
            if event["status"] == "metadata_error"
            for error in event["errors"]
        ]
        if failures:
            result["failures"] = failures
        if metadata_errors:
            result["metadata_errors"] = metadata_errors
        return result

    def save_annotation_events(
        self, requests: list[tuple[int, list[dict[str, int]]]]
    ):
        episode_indices = [episode_index for episode_index, _ in requests]
        if len(episode_indices) != len(set(episode_indices)):
            raise ValueError("each episode_index may appear only once")

        prepared = []
        for episode_index, ranges in requests:
            episode = self._episode(episode_index)
            length = int(episode["length"])
            validated = validate_ranges(ranges, length)
            mask = contact_mask(validated, length)
            soft_mask = contact_soft_mask(validated, length, self.soft_width)
            prepared.append((episode_index, mask, soft_mask))

        saved = []
        with self.lock, exclusive_dataset_lock(self.meta_dir):
            if prepared:
                with ThreadPoolExecutor(
                    max_workers=min(self.save_workers, len(prepared))
                ) as executor:
                    futures = {
                        executor.submit(
                            self._write_contact_columns,
                            episode_index,
                            mask,
                            soft_mask,
                        ): episode_index
                        for episode_index, mask, soft_mask in prepared
                    }
                    for future in as_completed(futures):
                        episode_index = futures[future]
                        try:
                            changed = future.result()
                        except Exception as error:
                            yield {
                                "episode_index": episode_index,
                                "status": "failed",
                                "error": str(error),
                            }
                            continue
                        saved.append(episode_index)
                        self.contact_episodes.add(episode_index)
                        self.contact_soft_episodes.add(episode_index)
                        yield {
                            "episode_index": episode_index,
                            "status": "saved",
                            "changed": changed,
                        }
            if saved:
                try:
                    self._ensure_contact_features()
                except Exception as error:
                    yield {
                        "status": "metadata_error",
                        "errors": [{"path": str(self.info_path), "error": str(error)}],
                    }

    def _write_contact_columns(
        self,
        episode_index: int,
        is_contact: np.ndarray,
        is_contact_soft: np.ndarray,
    ) -> bool:
        path = self.episode_path(episode_index)
        source = pq.ParquetFile(path)
        if source.metadata.num_rows != len(is_contact) or len(is_contact) != len(is_contact_soft):
            raise ValueError(
                f"Episode {episode_index} parquet has {source.metadata.num_rows} rows, "
                f"expected {len(is_contact)}"
            )
        expected = {
            "is_contact": is_contact,
            "is_contact_soft": is_contact_soft,
        }
        if all(name in source.schema_arrow.names for name in expected):
            current = source.read(columns=list(expected))
            if all(
                np.array_equal(current[name].to_numpy(zero_copy_only=False), values)
                for name, values in expected.items()
            ):
                return False
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
                    (
                        "is_contact",
                        pa.array(is_contact[offset : offset + length], type=pa.bool_()),
                    ),
                    (
                        "is_contact_soft",
                        pa.array(
                            is_contact_soft[offset : offset + length], type=pa.float32()
                        ),
                    ),
                )
                for name, column in columns:
                    if name in table.column_names:
                        table = table.set_column(table.column_names.index(name), name, column)
                    else:
                        table = table.append_column(name, column)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression=compression
                    )
                writer.write_table(table, row_group_size=length)
                offset += length
            if writer is None:
                raise ValueError(f"Episode {episode_index} parquet has no row groups")
            writer.close()
            writer = None
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
            return True
        finally:
            if writer is not None:
                writer.close()
            Path(temporary).unlink(missing_ok=True)

    def _ensure_contact_features(self) -> None:
        required = {}
        if self.contact_episodes:
            required["is_contact"] = IS_CONTACT_FEATURE
        if self.contact_soft_episodes:
            required["is_contact_soft"] = IS_CONTACT_SOFT_FEATURE
        if all(
            self.info.get("features", {}).get(name) == feature
            for name, feature in required.items()
        ):
            return
        updated_info = copy.deepcopy(self.info)
        updated_info.setdefault("features", {}).update(required)
        atomic_write_text(
            self.info_path,
            json.dumps(updated_info, ensure_ascii=False, indent=2) + "\n",
        )
        self.info = updated_info


def validate_ranges(
    ranges: list[dict[str, int]], length: int
) -> list[dict[str, int]]:
    normalized = [
        {"start": int(item["start"]), "end": int(item["end"])} for item in ranges
    ]
    if normalized != sorted(normalized, key=lambda item: (item["start"], item["end"])):
        raise ValueError("contact ranges must be sorted by start frame")
    for index, item in enumerate(normalized):
        if not 0 <= item["start"] <= item["end"] < length:
            raise ValueError(
                f"contact range must satisfy 0 <= start <= end < {length}"
            )
        if index and item["start"] <= normalized[index - 1]["end"]:
            raise ValueError("contact ranges must not overlap")
    return normalized


def contact_mask(ranges: list[dict[str, int]], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.bool_)
    for item in ranges:
        mask[item["start"] : item["end"] + 1] = True
    return mask


def contact_soft_mask(
    ranges: list[dict[str, int]], length: int, soft_width: int
) -> np.ndarray:
    soft = contact_mask(ranges, length).astype(np.float32)
    if soft_width == 0:
        return soft
    weights = 0.5 * (
        1.0 + np.cos(np.pi * np.arange(1, soft_width + 1) / (soft_width + 1))
    )
    for item in ranges:
        left_width = min(soft_width, item["start"])
        if left_width:
            indices = item["start"] - np.arange(1, left_width + 1)
            soft[indices] = np.maximum(soft[indices], weights[:left_width])
        right_width = min(soft_width, length - item["end"] - 1)
        if right_width:
            indices = item["end"] + np.arange(1, right_width + 1)
            soft[indices] = np.maximum(soft[indices], weights[:right_width])
    return soft


def true_ranges(values: list[bool]) -> list[dict[str, int]]:
    mask = np.asarray(values, dtype=np.bool_)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [
        {"start": int(start), "end": int(end - 1)}
        for start, end in changes.reshape(-1, 2)
    ]


def create_app(
    dataset: Path | str,
    preview_fps: float = 5,
    cache_episodes: int = 2,
    soft_width: int = 2,
    horizontal_flip: bool = True,
    preview_scale: float = 0.5,
    save_workers: int = 4,
) -> FastAPI:
    store = DatasetStore(
        root=Path(dataset),
        preview_fps=preview_fps,
        cache_episodes=cache_episodes,
        soft_width=soft_width,
        horizontal_flip=horizontal_flip,
        preview_scale=preview_scale,
        save_workers=save_workers,
    )
    app = FastAPI(title="LeRobot Contact Annotator")

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
            return Response(
                store.frame(episode_index, frame_index), media_type="image/jpeg"
            )
        except (KeyError, IndexError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.post("/api/episodes/{episode_index}/annotation")
    def annotate(
        episode_index: int, request: AnnotationRequest
    ) -> dict[str, Any]:
        try:
            return store.save_annotation(
                episode_index, [item.model_dump() for item in request.ranges]
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/annotations/batch")
    def annotate_batch(request: BatchAnnotationRequest) -> dict[str, Any]:
        try:
            return store.save_annotations([
                (item.episode_index, [value.model_dump() for value in item.ranges])
                for item in request.annotations
            ])
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
            completed = 0
            metadata_errors = []
            requests = [
                (
                    item.episode_index,
                    [value.model_dump() for value in item.ranges],
                )
                for item in request.annotations
            ]
            for event in store.save_annotation_events(requests):
                if event["status"] == "metadata_error":
                    metadata_errors.extend(event["errors"])
                    continue
                completed += 1
                saved += event["status"] == "saved"
                failed += event["status"] == "failed"
                event.update({"completed": completed, "total": total})
                yield json.dumps(event, ensure_ascii=False) + "\n"
            yield json.dumps(
                {
                    "status": "done",
                    "saved": saved,
                    "failed": failed,
                    "total": total,
                    "metadata_errors": metadata_errors,
                }
            ) + "\n"

        return StreamingResponse(
            events(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeRobot Contact Annotator</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#e8edf5;background:#0f131a}*{box-sizing:border-box}body{margin:0}
header{height:59px;padding:14px 22px;background:#171d27;border-bottom:1px solid #2b3442;display:flex;gap:20px;align-items:center}h1{font-size:18px;margin:0}
select,button{font:inherit;background:#222b38;color:#eef3fa;border:1px solid #3b4758;border-radius:7px;padding:8px 11px}button{cursor:pointer}button:hover{background:#2d394a}
.layout{display:grid;grid-template-columns:270px minmax(500px,1fr) 340px;height:calc(100vh - 59px)}aside{padding:14px;overflow:auto;background:#141a23}.episodes{border-right:1px solid #2b3442}.right{border-left:1px solid #2b3442}
.episode{display:flex;width:100%;justify-content:space-between;margin:5px 0}.episode.current{background:#34465f;border-color:#6b8fbd}.done{color:#77d49b}.pending{color:#e5b86b}.dirty{color:#ffad55}
main{padding:18px;display:flex;flex-direction:column;align-items:center;overflow:auto}.viewer{width:min(100%,900px);background:#080a0e;border-radius:10px;overflow:hidden;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center}#frame{max-width:100%;max-height:100%}
.timeline{width:min(100%,900px);margin-top:14px}.range-wrap{position:relative;padding-bottom:18px}.range-wrap input{width:100%;margin:0}.contact-track{position:absolute;left:8px;right:8px;bottom:0;height:15px;pointer-events:none}.contact-region{position:absolute;bottom:0;height:10px;background:#ef6b73;opacity:.55;border-radius:3px}.contact-keyframe{position:absolute;top:0;width:4px;height:15px;transform:translateX(-2px);border-radius:2px;z-index:2}.contact-keyframe.start{background:#63d68b}.contact-keyframe.end{background:#ef6b73}.contact-keyframe.unsaved{opacity:.65;outline:1px dashed #fff}.contact-keyframe.saved{opacity:1}.contact-keyframe.open{background:#ffad55;opacity:1}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}.frame-label{font-variant-numeric:tabular-nums;min-width:175px}.primary{background:#246bce;border-color:#347be0}.end{background:#8b3c43}.save{width:100%;margin-top:14px}
.range{background:#202936;border-radius:7px;padding:9px;margin:7px 0}.range button{float:right;padding:4px 8px}.open{color:#ffad55}.message{min-height:24px;color:#7ed79f;margin-top:10px}
</style></head><body>
<header><h1>LeRobot Contact Annotator</h1><label>Task <select id="task"></select></label><span id="progress"></span></header>
<div class="layout"><aside class="episodes"><b>Episodes</b><div id="episodes"></div></aside>
<main><div class="viewer"><img id="frame" alt="Episode frame"></div><div class="timeline">
<div class="range-wrap"><input id="slider" type="range" min="0" value="0" step="1"><div id="contactTrack" class="contact-track"></div></div>
<div class="controls"><button id="play">▶ Play</button><button id="prev">← Frame</button><button id="next">Frame →</button><span id="frameLabel" class="frame-label"></span><button id="contactToggle" class="primary">Contact Start</button></div>
</div></main><aside class="right"><b>Contact ranges (inclusive)</b><div id="ranges"></div><button id="save" class="primary save">Save All Changes (0)</button><div id="message" class="message"></div></aside></div>
<script>
const $=id=>document.getElementById(id);let tasks=[],episodes=[],detail=null,current=0,ranges=[],openStart=null,timer=null,activeTask=null;
const drafts=new Map(),savedByEpisode=new Map(),annotatedByEpisode=new Map(),edited=new Set();
async function json(url,options){const response=await fetch(url,options),body=await response.json();if(!response.ok)throw new Error(body.detail||response.statusText);return body}
function cloneRanges(values){return (values||[]).map(item=>({...item}))}
function equal(a,b){return JSON.stringify(a||[])===JSON.stringify(b||[])}
function stashCurrent(){if(detail)drafts.set(detail.episode_index,cloneRanges(ranges))}
function needsSave(id){return edited.has(id)&&!equal(drafts.get(id),savedByEpisode.get(id))}
function dirtyIds(){return [...edited].filter(needsSave).sort((a,b)=>a-b)}
function saveIds(){return episodes.map(item=>item.episode_index).filter(id=>annotatedByEpisode.get(id)||edited.has(id))}
function rememberEpisode(episode){const id=episode.episode_index;annotatedByEpisode.set(id,episode.annotated);savedByEpisode.set(id,cloneRanges(episode.ranges));if(!drafts.has(id))drafts.set(id,cloneRanges(episode.ranges))}
async function init(){tasks=await json('/api/tasks');for(const task of tasks){const option=document.createElement('option');option.value=task.task_index;option.textContent=task.task_index+': '+task.task;$('task').appendChild(option)}$('task').onchange=loadTask;if(tasks.length)await loadTask()}
async function loadTask(){if(openStart!==null){$('task').value=activeTask;$('message').textContent='Finish the open contact range before switching tasks.';return}stashCurrent();stop();activeTask=Number($('task').value);episodes=await json('/api/tasks/'+activeTask+'/episodes');for(const episode of episodes)rememberEpisode(episode);renderEpisodes();renderProgress();if(episodes.length)await selectEpisode(episodes[0].episode_index)}
async function selectEpisode(id){if(openStart!==null){$('message').textContent='Finish the open contact range before switching episodes.';return}stashCurrent();stop();const next=await json('/api/episodes/'+id);detail=next;rememberEpisode(next);ranges=cloneRanges(drafts.get(id));openStart=null;current=0;$('slider').max=detail.length-1;showFrame();render();$('message').textContent=''}
function changeEpisode(delta){if(!detail)return;const index=episodes.findIndex(item=>item.episode_index===detail.episode_index),next=episodes[index+delta];if(next)selectEpisode(next.episode_index)}
function renderProgress(){const task=tasks.find(item=>item.task_index===activeTask);$('progress').textContent=task?task.annotated+'/'+task.episodes+' saved · '+dirtyIds().length+' unsaved':'';$('save').textContent='Save All Annotated ('+saveIds().length+')'}
function renderEpisodes(){const box=$('episodes');box.innerHTML='';for(const episode of episodes){const id=episode.episode_index,button=document.createElement('button');button.className='episode'+(detail&&detail.episode_index===id?' current':'');button.onclick=()=>selectEpisode(id);const state=needsSave(id)?'<span class="dirty">Unsaved</span>':(annotatedByEpisode.get(id)?'<span class="done">is_contact ✓</span>':'<span class="pending">Pending</span>');button.innerHTML='<span>Episode '+id+'</span>'+state;box.appendChild(button)}renderProgress()}
function showFrame(){if(!detail)return;$('slider').value=current;$('frame').src='/api/episodes/'+detail.episode_index+'/frames/'+current;$('frameLabel').textContent='Frame '+current+' / '+(detail.length-1)}
function move(delta){if(!detail)return;current=Math.max(0,Math.min(detail.length-1,current+delta));showFrame()}
function play(){if(timer){stop();return}$('play').textContent='⏸ Pause';timer=setInterval(()=>{if(current>=detail.length-1){stop();return}move(detail.preview_stride)},1000/detail.fps*detail.preview_stride)}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Play'}
function changed(){const id=detail.episode_index;drafts.set(id,cloneRanges(ranges));if(equal(ranges,savedByEpisode.get(id)))edited.delete(id);else edited.add(id);render()}
function render(){renderEpisodes();const box=$('ranges');box.innerHTML='';ranges.forEach((range,index)=>{const row=document.createElement('div');row.className='range';row.textContent='Frame '+range.start+'–'+range.end;const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{ranges.splice(index,1);changed()};row.appendChild(remove);box.appendChild(row)});if(openStart!==null){const row=document.createElement('div');row.className='range open';row.textContent='Open contact: frame '+openStart+'–?';box.appendChild(row)}const track=$('contactTrack');track.innerHTML='',saved=savedByEpisode.get(detail.episode_index)||[],max=Math.max(1,detail.length-1);for(const range of ranges){const region=document.createElement('span');region.className='contact-region';region.style.left=(range.start/detail.length*100)+'%';region.style.width=((range.end-range.start+1)/detail.length*100)+'%';track.appendChild(region);const persisted=saved.some(item=>item.start===range.start&&item.end===range.end);for(const [kind,frame] of [['start',range.start],['end',range.end]]){const marker=document.createElement('span');marker.className='contact-keyframe '+kind+' '+(persisted?'saved':'unsaved');marker.style.left=(frame/max*100)+'%';marker.title='Contact '+kind+' · frame '+frame+(persisted?' · saved':' · unsaved');track.appendChild(marker)}}if(openStart!==null){const marker=document.createElement('span');marker.className='contact-keyframe start open';marker.style.left=(openStart/max*100)+'%';marker.title='Contact start · frame '+openStart+' · open';track.appendChild(marker)}const toggle=$('contactToggle');toggle.textContent=openStart===null?'Contact Start':'Contact End';toggle.className=openStart===null?'primary':'end'}
$('contactToggle').onclick=()=>{if(!detail)return;if(openStart===null){const previous=ranges[ranges.length-1];if(previous&&current<=previous.end){$('message').textContent='The new start must be after the previous contact end.';return}openStart=current;$('message').textContent='';changed();return}if(current<openStart){$('message').textContent='Contact end cannot precede contact start.';return}ranges.push({start:openStart,end:current});openStart=null;$('message').textContent='';changed()};
async function saveAll(){
  if(openStart!==null){$('message').textContent='Finish the open contact range before saving.';return}
  stashCurrent();const ids=saveIds();
  if(!ids.length){$('message').textContent='There are no annotated episodes in this task.';return}
  const button=$('save');button.disabled=true;button.textContent='Saving 0 / '+ids.length;
  let summary=null,metadataErrorCount=0;
  try{
    const submitted=new Map(ids.map(id=>[id,cloneRanges(drafts.get(id))]));
    const annotations=ids.map(id=>({episode_index:id,ranges:submitted.get(id)}));
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
          const id=event.episode_index,savedRanges=submitted.get(id)||[];savedByEpisode.set(id,cloneRanges(savedRanges));annotatedByEpisode.set(id,true);
          if(equal(drafts.get(id),savedRanges))edited.delete(id);
          if(detail&&id===detail.episode_index)detail.ranges=cloneRanges(savedRanges);
          metadataErrorCount+=(event.metadata_errors||[]).length;render();button.textContent='Saving '+event.completed+' / '+event.total;
          $('message').textContent='Saved episode '+id+' ('+event.completed+' / '+event.total+').';
        }else if(event.status==='failed'){
          render();button.textContent='Saving '+event.completed+' / '+event.total;
          $('message').textContent='Episode '+event.episode_index+' failed and remains unsaved: '+event.error;
        }else if(event.status==='done'){summary=event;metadataErrorCount+=(event.metadata_errors||[]).length}
      }
      if(chunk.done)break;
    }
    tasks=await json('/api/tasks');episodes=await json('/api/tasks/'+activeTask+'/episodes');
    for(const episode of episodes){annotatedByEpisode.set(episode.episode_index,episode.annotated);if(!edited.has(episode.episode_index)){savedByEpisode.set(episode.episode_index,cloneRanges(episode.ranges));drafts.set(episode.episode_index,cloneRanges(episode.ranges))}}
    render();
    if(summary)$('message').textContent='Saved '+summary.saved+' / '+summary.total+' episode(s).'+(summary.failed?' '+summary.failed+' failed and remain unsaved.':'')+(metadataErrorCount?' Metadata sync reported '+metadataErrorCount+' error(s).':'');
  }catch(error){$('message').textContent=error.message}finally{button.disabled=false;renderProgress()}
}
$('save').onclick=saveAll;
$('slider').oninput=event=>{stop();current=Number(event.target.value);showFrame()};$('play').onclick=play;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
document.addEventListener('keydown',event=>{if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);if(event.key==='ArrowUp'||event.key==='ArrowDown'){event.preventDefault();changeEpisode(event.key==='ArrowUp'?-1:1)}if(event.key===' '){event.preventDefault();play()}});window.onbeforeunload=()=>openStart!==null||dirtyIds().length?'You have unsaved annotations.':undefined;init().catch(error=>$('message').textContent=error.message);
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, help="Local LeRobot v2 dataset directory to modify."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--preview-fps", type=float, default=5)
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--cache-episodes", type=int, default=2)
    parser.add_argument(
        "--soft-width",
        type=int,
        default=2,
        help="Raised-cosine soft-label width on both sides of each contact interval.",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=4,
        help="Number of episode parquet files written concurrently by Save All.",
    )
    parser.add_argument(
        "--horizontal-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Horizontally flip images in the web UI only (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        dataset=args.dataset,
        preview_fps=args.preview_fps,
        cache_episodes=args.cache_episodes,
        soft_width=args.soft_width,
        horizontal_flip=args.horizontal_flip,
        preview_scale=args.preview_scale,
        save_workers=args.save_workers,
    )
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
