from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tools.annotate_lerobot_subtasks as annotator
from tools.annotate_lerobot_subtasks import completion_mask, create_app, true_ranges


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def png_bytes(value: int) -> bytes:
    buffer = io.BytesIO()
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def make_dataset(root: Path) -> None:
    info = {
        "codebase_version": "v2.0",
        "total_episodes": 2,
        "total_frames": 10,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": [4, 6, 3], "names": ["height", "width", "channel"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "stats.json", {})
    write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 7, "task": "annotate me"}])
    write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 3, "tasks": ["annotate me"], "length": 5},
            {"episode_index": 8, "tasks": ["annotate me"], "length": 5},
        ],
    )
    for episode_index in (3, 8):
        table = pa.table(
            {
                "image": [{"bytes": png_bytes(frame * 20), "path": None} for frame in range(5)],
                "frame_index": pa.array(range(5), type=pa.int64()),
                "episode_index": pa.array([episode_index] * 5, type=pa.int64()),
                "index": pa.array(range(5), type=pa.int64()),
                "task_index": pa.array([7] * 5, type=pa.int64()),
            }
        )
        path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=2)


def test_completion_mask_marks_last_n_frames_within_each_subtask() -> None:
    assert completion_mask([4, 7], length=10, completion_frames=2).tolist() == [
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert completion_mask([2], length=5, completion_frames=4).tolist() == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert completion_mask([], length=7, completion_frames=3).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_true_ranges_returns_inclusive_completion_intervals() -> None:
    assert true_ranges([False, True, True, False, True]) == [
        {"start": 1, "end": 2},
        {"start": 4, "end": 4},
    ]
    assert true_ranges([False, False]) == []


def test_http_annotation_writes_full_frame_ids_and_recovers_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    client = TestClient(create_app(root, preview_fps=5, cache_episodes=1, completion_frames=1))
    parquet = root / "data" / "chunk-000" / "episode_000003.parquet"
    original_row_groups = pq.ParquetFile(parquet).num_row_groups
    original_compression = pq.ParquetFile(parquet).metadata.row_group(0).column(0).compression

    page = client.get("/")
    assert page.status_code == 200
    assert 'id="completion"' in page.text
    assert 'completion-region' in page.text
    tasks = client.get("/api/tasks").json()
    assert tasks == [{"task_index": 7, "task": "annotate me", "episodes": 2, "annotated": 0}]
    episodes = client.get("/api/tasks/7/episodes").json()
    assert [row["episode_index"] for row in episodes] == [3, 8]
    detail = client.get("/api/episodes/3").json()
    assert detail["length"] == 5
    assert detail["preview_stride"] == 2
    assert detail["boundaries"] == []
    assert detail["completion_frames"] == 1
    assert detail["completion_ranges"] == []
    frame = client.get("/api/episodes/3/frames/2")
    assert frame.status_code == 200
    assert frame.headers["content-type"] == "image/jpeg"

    response = client.post("/api/episodes/3/annotation", json={"boundaries": [2, 4]})
    assert response.status_code == 200, response.text
    assert response.json()["segments"] == [
        {"subtask_id": 1, "start": 0, "end": 1},
        {"subtask_id": 2, "start": 2, "end": 3},
        {"subtask_id": 3, "start": 4, "end": 4},
    ]
    assert pq.read_table(parquet, columns=["subtask_id"])["subtask_id"].to_pylist() == [1, 1, 2, 2, 3]
    assert pq.read_table(parquet, columns=["is_complete"])["is_complete"].to_pylist() == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert client.get("/api/episodes/3").json()["completion_ranges"] == [
        {"start": 1, "end": 1},
        {"start": 3, "end": 4},
    ]
    assert pq.ParquetFile(parquet).num_row_groups == original_row_groups
    assert pq.ParquetFile(parquet).metadata.row_group(0).column(0).compression == original_compression
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["subtask_id"] == {"dtype": "int64", "shape": [1], "names": None}
    assert "is_complete" not in info["features"]

    restarted = TestClient(create_app(root, preview_fps=5, cache_episodes=1, completion_frames=1))
    assert restarted.get("/api/episodes/3").json()["boundaries"] == [2, 4]
    assert restarted.get("/api/tasks").json()[0]["annotated"] == 1
    assert restarted.post("/api/episodes/3/annotation", json={"boundaries": [3]}).status_code == 200
    assert pq.read_table(parquet, columns=["subtask_id"])["subtask_id"].to_pylist() == [1, 1, 1, 2, 2]
    assert pq.read_table(parquet, columns=["is_complete"])["is_complete"].to_pylist() == [
        False,
        False,
        True,
        False,
        True,
    ]

    (root / "meta" / "subtask_annotations.jsonl").unlink()
    recovered = TestClient(create_app(root, preview_fps=5, cache_episodes=1))
    assert recovered.get("/api/episodes/3").json()["boundaries"] == [3]
    assert recovered.get("/api/tasks").json()[0]["annotated"] == 1


def test_existing_subtask_ids_are_reported_as_needing_is_complete(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    parquet = root / "data/chunk-000/episode_000003.parquet"
    table = pq.read_table(parquet).append_column(
        "subtask_id", pa.array([1, 1, 2, 2, 2], type=pa.int64())
    )
    pq.write_table(table, parquet, row_group_size=2)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["subtask_id"] = {"dtype": "int64", "shape": [1], "names": None}
    write_json(info_path, info)

    client = TestClient(create_app(root))
    episode = client.get("/api/episodes/3").json()

    assert episode["annotated"] is True
    assert episode["completion_annotated"] is False
    row = client.get("/api/tasks/7/episodes").json()[0]
    assert row["completion_annotated"] is False
    assert [item["episode_index"] for item in client.get("/api/completion-pending").json()] == [3]

    with client.stream(
        "POST",
        "/api/annotations/stream",
        json={"annotations": [{"episode_index": 3, "boundaries": [2]}]},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]
    assert events[0]["status"] == "saved"
    assert pq.read_table(parquet, columns=["is_complete"])["is_complete"].to_pylist() == [
        True,
        True,
        True,
        True,
        True,
    ]
    restarted = TestClient(create_app(root))
    assert restarted.get("/api/episodes/3").json()["completion_annotated"] is True


def test_completion_frames_change_queues_recomputation(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    client = TestClient(create_app(root, completion_frames=1))
    assert client.post("/api/episodes/3/annotation", json={"boundaries": [2]}).status_code == 200

    restarted = TestClient(create_app(root, completion_frames=5))

    assert restarted.get("/api/episodes/3").json()["completion_annotated"] is False
    assert [item["episode_index"] for item in restarted.get("/api/completion-pending").json()] == [3]


def test_batch_annotation_writes_multiple_episodes_once(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    client = TestClient(create_app(root))

    response = client.post(
        "/api/annotations/batch",
        json={
            "annotations": [
                {"episode_index": 3, "boundaries": [2]},
                {"episode_index": 8, "boundaries": []},
            ]
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"saved_episode_indices": [3, 8]}
    assert pq.read_table(
        root / "data/chunk-000/episode_000003.parquet", columns=["subtask_id"]
    )["subtask_id"].to_pylist() == [1, 1, 2, 2, 2]
    assert pq.read_table(
        root / "data/chunk-000/episode_000008.parquet", columns=["subtask_id"]
    )["subtask_id"].to_pylist() == [1, 1, 1, 1, 1]
    assert client.get("/api/tasks").json()[0]["annotated"] == 2
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["is_complete"] == {
        "dtype": "bool",
        "shape": [1],
        "names": None,
    }


def test_stream_annotation_reports_each_episode_as_it_finishes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    client = TestClient(create_app(root))

    with client.stream(
        "POST",
        "/api/annotations/stream",
        json={
            "annotations": [
                {"episode_index": 3, "boundaries": [2]},
                {"episode_index": 8, "boundaries": [3]},
            ]
        },
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events == [
        {"episode_index": 3, "status": "saved", "completed": 1, "total": 2},
        {"episode_index": 8, "status": "saved", "completed": 2, "total": 2},
        {"status": "done", "saved": 2, "failed": 0, "total": 2},
    ]


def test_batch_annotation_reports_partial_failure_and_keeps_success(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    failed_parquet = root / "data/chunk-000/episode_000008.parquet"
    original_failed = failed_parquet.read_bytes()
    real_replace = annotator.os.replace

    def fail_second_episode(source, destination):
        if Path(destination) == failed_parquet:
            raise OSError("simulated second episode failure")
        return real_replace(source, destination)

    monkeypatch.setattr(annotator.os, "replace", fail_second_episode)
    client = TestClient(create_app(root))
    response = client.post(
        "/api/annotations/batch",
        json={
            "annotations": [
                {"episode_index": 3, "boundaries": [2]},
                {"episode_index": 8, "boundaries": [3]},
            ]
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["saved_episode_indices"] == [3]
    assert response.json()["failures"][0]["episode_index"] == 8
    assert pq.read_table(
        root / "data/chunk-000/episode_000003.parquet", columns=["subtask_id"]
    )["subtask_id"].to_pylist() == [1, 1, 2, 2, 2]
    assert failed_parquet.read_bytes() == original_failed
    assert client.get("/api/tasks").json()[0]["annotated"] == 1
    annotations = [
        json.loads(line)
        for line in (root / "meta/subtask_annotations.jsonl").read_text().splitlines()
    ]
    assert [row["episode_index"] for row in annotations] == [3]


@pytest.mark.parametrize("metadata_name", ["info.json", "subtask_annotations.jsonl"])
def test_batch_annotation_reports_metadata_failure_after_parquet_commit(
    tmp_path: Path, monkeypatch, metadata_name: str
) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    target = root / "meta" / metadata_name
    real_atomic_write_text = annotator.atomic_write_text

    def fail_target(path, text):
        if Path(path) == target:
            raise OSError("simulated metadata failure")
        return real_atomic_write_text(path, text)

    monkeypatch.setattr(annotator, "atomic_write_text", fail_target)
    client = TestClient(create_app(root))
    response = client.post(
        "/api/annotations/batch",
        json={"annotations": [{"episode_index": 3, "boundaries": [2]}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["saved_episode_indices"] == [3]
    assert response.json()["metadata_errors"][0]["path"] == str(target)
    assert pq.read_table(
        root / "data/chunk-000/episode_000003.parquet", columns=["subtask_id"]
    )["subtask_id"].to_pylist() == [1, 1, 2, 2, 2]

    monkeypatch.undo()
    recovered = TestClient(create_app(root))
    assert recovered.get("/api/episodes/3").json()["boundaries"] == [2]
    info = json.loads((root / "meta/info.json").read_text())
    assert info["features"]["subtask_id"] == {"dtype": "int64", "shape": [1], "names": None}


def test_web_ui_is_english_and_has_batch_save_and_timeline_markers(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)

    html = TestClient(create_app(root)).get("/").text

    assert '<html lang="en">' in html
    assert "Save All Changes" in html
    assert 'id="markers"' in html
    assert "Saved boundary" in html
    assert "Unsaved boundary" in html
    assert "/api/annotations/stream" in html
    assert "const edited=new Set()" in html
    assert "visited=new Set()" not in html
    assert "function dirtyIds(){return [...new Set([...edited,...completionPending])]" in html
    assert "submittedByEpisode" in html
    assert ".episode.current" in html
    assert "button.classList.toggle('current'" in html
    assert "subtask_id ✓" in html
    assert "persisted.className='persisted'" in html
    assert "Needs is_complete" in html
    assert "completionPending" in html
    assert "/api/completion-pending" in html
    assert "boundary-thumbnail" in html
    assert "thumbnail.loading='lazy'" in html
    assert "'/frames/'+value" in html
    assert "保存" not in html
    assert "标注" not in html


def test_http_annotation_rejects_invalid_boundaries_without_writing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    client = TestClient(create_app(root))

    response = client.post("/api/episodes/3/annotation", json={"boundaries": [0, 5]})

    assert response.status_code == 422
    assert "strictly inside" in response.json()["detail"]
    assert "subtask_id" not in pq.read_schema(root / "data/chunk-000/episode_000003.parquet").names
    assert not (root / "meta" / "subtask_annotations.jsonl").exists()


def test_parquet_write_failure_keeps_original_file(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    parquet = root / "data/chunk-000/episode_000003.parquet"
    original = parquet.read_bytes()
    real_replace = annotator.os.replace

    def fail_parquet_replace(source, destination):
        if Path(destination) == parquet:
            raise OSError("simulated atomic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(annotator.os, "replace", fail_parquet_replace)
    client = TestClient(create_app(root), raise_server_exceptions=False)
    response = client.post("/api/episodes/3/annotation", json={"boundaries": [2]})

    assert response.status_code == 500
    assert parquet.read_bytes() == original
    assert "subtask_id" not in pq.read_schema(parquet).names
    assert not (root / "meta/subtask_annotations.jsonl").exists()


def test_http_serves_agent_view_frames_from_mp4(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    make_dataset(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["features"].pop("image")
    info["features"]["observation.images.image"] = {
        "dtype": "video",
        "shape": [4, 6, 3],
        "names": ["height", "width", "rgb"],
    }
    info["total_videos"] = 2
    write_json(root / "meta" / "info.json", info)
    for episode_index in (3, 8):
        path = (
            root
            / "videos"
            / "chunk-000"
            / "observation.images.image"
            / f"episode_{episode_index:06d}.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (6, 4))
        assert writer.isOpened()
        for frame_index in range(5):
            writer.write(np.full((4, 6, 3), frame_index * 30, dtype=np.uint8))
        writer.release()

    client = TestClient(create_app(root))
    frame = client.get("/api/episodes/3/frames/4")

    assert frame.status_code == 200, frame.text
    assert frame.headers["content-type"] == "image/jpeg"
    assert len(frame.content) > 100
