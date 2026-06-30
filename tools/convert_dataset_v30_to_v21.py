#!/usr/bin/env python
"""Convert a local LeRobot v3.0 dataset back to the v2.1 file layout.

The conversion is intentionally local-only:

* v3.0 stores many episodes per parquet/video file.
* v2.1 stores one parquet and one video per episode.

This script expands the v3.0 chunks into v2.1 episode files while preserving
the original feature names and per-episode statistics.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n: int = 1) -> None:
            pass

        def set_postfix(self, *args, **kwargs) -> None:
            pass

        @staticmethod
        def write(message: str) -> None:
            print(message)


DEFAULT_INPUT_ROOT = Path("/data0/luokang/dataset/luokang/lerobot/calvin-abc-lerobot-depth")
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output-root before conversion.")
    parser.add_argument(
        "--video-mode",
        choices=["split", "skip"],
        default="split",
        help="split writes v2.1 per-episode videos; skip only writes metadata/parquet and is not fully loadable.",
    )
    parser.add_argument("--validate", action="store_true", help="Load the converted dataset with LeRobot v2.1.")
    return parser.parse_args()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(data), indent=4, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(as_jsonable(row), ensure_ascii=False) + "\n")


def unflatten(flat: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in flat.items():
        cursor = out
        parts = key.split(sep)
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    return json.loads(info_path.read_text())


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def read_parquet_dir(path: Path) -> pd.DataFrame:
    files = parquet_files(path)
    frames = [
        pd.read_parquet(file)
        for file in tqdm(files, desc=f"read {path.name}", unit="file", leave=False)
    ]
    return pd.concat(frames, ignore_index=True)


def load_tasks(input_root: Path) -> list[dict[str, Any]]:
    task_path = input_root / "meta" / "tasks.parquet"
    if task_path.is_file():
        df = pd.read_parquet(task_path)
    else:
        df = read_parquet_dir(input_root / "meta" / "tasks")

    if "task_index" not in df.columns:
        df["task_index"] = np.arange(len(df), dtype=np.int64)

    if "task" in df.columns:
        task_values = df["task"].tolist()
    else:
        task_values = df.index.tolist()

    rows = [
        {"task_index": int(task_index), "task": str(task)}
        for task_index, task in zip(df["task_index"].tolist(), task_values, strict=True)
    ]
    return sorted(rows, key=lambda item: item["task_index"])


def load_episodes(input_root: Path, max_episodes: int | None) -> pd.DataFrame:
    episodes = read_parquet_dir(input_root / "meta" / "episodes")
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    if max_episodes is not None:
        episodes = episodes.iloc[:max_episodes].copy()
    return episodes


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]


def make_v21_info(info: dict[str, Any], episodes: pd.DataFrame, video_mode: str) -> dict[str, Any]:
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = video_keys_from_info(info)
    total_episodes = int(len(episodes))
    total_frames = int(episodes["length"].sum())
    out = {
        key: value
        for key, value in info.items()
        if key not in {"data_files_size_in_mb", "video_files_size_in_mb"}
    }
    out["codebase_version"] = "v2.1"
    out["total_episodes"] = total_episodes
    out["total_frames"] = total_frames
    out["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    out["chunks_size"] = chunks_size
    out["splits"] = {"train": f"0:{total_episodes}"}
    out["data_path"] = DEFAULT_PARQUET_PATH
    out["video_path"] = DEFAULT_VIDEO_PATH
    out["total_videos"] = total_episodes * len(video_keys) if video_mode == "split" else 0
    return out


def episode_meta_rows(episodes: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in tqdm(episodes.itertuples(index=False), total=len(episodes), desc="episodes meta", unit="ep"):
        rows.append(
            {
                "episode_index": int(getattr(row, "episode_index")),
                "tasks": list(getattr(row, "tasks")),
                "length": int(getattr(row, "length")),
            }
        )
    return rows


def episode_stats_rows(episodes: pd.DataFrame) -> list[dict[str, Any]]:
    stats_cols = [col for col in episodes.columns if col.startswith("stats/")]
    rows = []
    for _, row in tqdm(episodes.iterrows(), total=len(episodes), desc="episode stats", unit="ep"):
        flat = {col.removeprefix("stats/"): as_jsonable(row[col]) for col in stats_cols}
        rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "stats": unflatten(flat),
            }
        )
    return rows


def prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)


def copy_optional_files(input_root: Path, output_root: Path) -> None:
    for name in [".gitattributes", "README.md"]:
        src = input_root / name
        if src.is_file():
            shutil.copy2(src, output_root / name)


def write_metadata(input_root: Path, output_root: Path, info: dict[str, Any], episodes: pd.DataFrame) -> None:
    write_json(output_root / "meta" / "info.json", info)
    write_jsonl(output_root / "meta" / "tasks.jsonl", load_tasks(input_root))
    write_jsonl(output_root / "meta" / "episodes.jsonl", episode_meta_rows(episodes))
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episode_stats_rows(episodes))

    stats_path = input_root / "meta" / "stats.json"
    if stats_path.is_file():
        shutil.copy2(stats_path, output_root / "meta" / "stats.json")


def data_source_path(input_root: Path, info: dict[str, Any], chunk_index: int, file_index: int) -> Path:
    rel = info["data_path"].format(chunk_index=chunk_index, file_index=file_index)
    return input_root / rel


def output_data_path(output_root: Path, chunks_size: int, episode_index: int) -> Path:
    rel = DEFAULT_PARQUET_PATH.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return output_root / rel


def convert_data(input_root: Path, output_root: Path, info: dict[str, Any], episodes: pd.DataFrame) -> None:
    chunks_size = int(info.get("chunks_size", 1000))
    groups = list(episodes.groupby(["data/chunk_index", "data/file_index"], sort=True))

    with tqdm(total=len(episodes), desc="data episodes", unit="ep") as episode_bar:
        for (chunk_index, file_index), group in tqdm(groups, desc="data files", unit="file"):
            src = data_source_path(input_root, info, int(chunk_index), int(file_index))
            episode_bar.set_postfix(file=str(src.relative_to(input_root)))
            table = pq.read_table(src)
            episode_col = table["episode_index"]

            for _, ep in group.sort_values("episode_index").iterrows():
                ep_idx = int(ep["episode_index"])
                ep_table = table.filter(pc.equal(episode_col, ep_idx))
                expected_len = int(ep["length"])
                if ep_table.num_rows != expected_len:
                    raise RuntimeError(
                        f"Episode {ep_idx} expected {expected_len} rows, got {ep_table.num_rows} from {src}"
                    )
                dst = output_data_path(output_root, chunks_size, ep_idx)
                dst.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(ep_table.replace_schema_metadata(None), dst)
                episode_bar.update(1)


def video_source_path(input_root: Path, info: dict[str, Any], video_key: str, chunk_index: int, file_index: int) -> Path:
    rel = info["video_path"].format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
    return input_root / rel


def output_video_path(output_root: Path, chunks_size: int, video_key: str, episode_index: int) -> Path:
    rel = DEFAULT_VIDEO_PATH.format(
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )
    return output_root / rel


AV1_REENCODE_CRF = "26"
AV1_REENCODE_PRESET = "8"
AV1_REENCODE_GOP_SIZE = 2


def open_video_encoder(dst: Path, fps: int, width: int, height: int):
    import av

    dst.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(dst), mode="w")
    stream = container.add_stream("av1", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": AV1_REENCODE_CRF, "preset": AV1_REENCODE_PRESET}
    stream.gop_size = AV1_REENCODE_GOP_SIZE
    return container, stream


def close_video_encoder(container: Any, stream: Any) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def encode_frame(frame: Any, out_container: Any, out_stream: Any, width: int, height: int, frame_index: int, fps: int) -> None:
    frame = frame.reformat(width=width, height=height, format="yuv420p")
    frame.pts = frame_index
    frame.time_base = Fraction(1, fps)
    for packet in out_stream.encode(frame):
        out_container.mux(packet)


def write_episode_video(
    job: dict[str, Any],
    source_frames: list[tuple[float, Any]],
    fps: int,
    width: int,
    height: int,
) -> None:
    length = int(job["length"])
    if len(source_frames) < 2 and length > 1:
        raise RuntimeError(f"Not enough decoded frames for {job['dst']}")

    out_container, out_stream = open_video_encoder(job["dst"], fps, width, height)
    target_idx = 0
    cursor = 1 if len(source_frames) > 1 else 0
    try:
        while target_idx < length:
            target_ts = job["from_timestamp"] + target_idx / fps
            while cursor < len(source_frames) and source_frames[cursor][0] < target_ts:
                cursor += 1

            if cursor >= len(source_frames):
                chosen = source_frames[-1]
            elif cursor == 0:
                chosen = source_frames[0]
            else:
                prev = source_frames[cursor - 1]
                curr = source_frames[cursor]
                chosen = prev if abs(prev[0] - target_ts) <= abs(curr[0] - target_ts) else curr

            encode_frame(chosen[1], out_container, out_stream, width, height, target_idx, fps)
            target_idx += 1
    finally:
        close_video_encoder(out_container, out_stream)


def split_one_video_file(
    src: Path,
    jobs: list[dict[str, Any]],
    fps: int,
    progress: Any | None = None,
) -> None:
    """Split a source video by decoding exact episode frames and re-encoding AV1.

    This matches LeRobot v3's timestamp-based loading strategy: seek to a
    preceding keyframe, decode surrounding frames, choose the closest decoded
    frame for each episode timestamp, and write exactly `length` frames.
    """
    import av

    if not src.is_file():
        raise FileNotFoundError(src)

    jobs = sorted(jobs, key=lambda item: item["from_timestamp"])
    job_index = 0
    frame_slack = 0.5 / fps
    source_frames: list[tuple[float, Any]] = []

    in_container = av.open(str(src), mode="r")
    in_stream = in_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    try:
        for frame in in_container.decode(in_stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time)

            while job_index < len(jobs):
                job = jobs[job_index]
                target_last_ts = job["from_timestamp"] + (int(job["length"]) - 1) / fps
                if timestamp < target_last_ts + frame_slack:
                    break

                write_episode_video(job, source_frames, fps, width, height)
                if progress is not None:
                    progress.update(1)
                job_index += 1
                source_frames = [item for item in source_frames if item[0] >= jobs[job_index]["from_timestamp"] - frame_slack] if job_index < len(jobs) else []

            if job_index >= len(jobs):
                break

            if timestamp >= jobs[job_index]["from_timestamp"] - frame_slack:
                source_frames.append((timestamp, frame))
    finally:
        in_container.close()

    while job_index < len(jobs):
        write_episode_video(jobs[job_index], source_frames, fps, width, height)
        if progress is not None:
            progress.update(1)
        job_index += 1

def convert_videos(input_root: Path, output_root: Path, info: dict[str, Any], episodes: pd.DataFrame) -> None:
    chunks_size = int(info.get("chunks_size", 1000))
    fps = int(info["fps"])
    video_keys = video_keys_from_info(info)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)

    for _, ep in episodes.iterrows():
        ep_idx = int(ep["episode_index"])
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            dst = output_video_path(output_root, chunks_size, video_key, ep_idx)
            grouped[
                (
                    video_key,
                    int(ep[f"{prefix}/chunk_index"]),
                    int(ep[f"{prefix}/file_index"]),
                )
            ].append(
                {
                    "episode_index": ep_idx,
                    "length": int(ep["length"]),
                    "from_timestamp": float(ep[f"{prefix}/from_timestamp"]),
                    "to_timestamp": float(ep[f"{prefix}/to_timestamp"]),
                    "dst": dst,
                }
            )

    group_items = sorted(grouped.items())
    total_segments = sum(len(jobs) for _, jobs in group_items)
    with tqdm(total=total_segments, desc="video episodes", unit="ep") as episode_bar:
        for (video_key, chunk_index, file_index), jobs in tqdm(group_items, desc="video files", unit="file"):
            src = video_source_path(input_root, info, video_key, chunk_index, file_index)
            episode_bar.set_postfix(file=str(src.relative_to(input_root)))
            split_one_video_file(src, jobs, fps=fps, progress=episode_bar)

def validate_dataset(output_root: Path) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("converted/calvin-abc-lerobot-depth-v21", root=output_root, video_backend="pyav")
    print(f"[validate] episodes={ds.num_episodes} frames={len(ds)}")
    sample = ds[0]
    print(f"[validate] first sample keys={sorted(sample.keys())}")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"[validate] {key}: shape={tuple(value.shape)} dtype={getattr(value, 'dtype', None)}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else Path(str(input_root) + "_v21")

    info = load_info(input_root)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected codebase_version v3.0, got {info.get('codebase_version')}")

    episodes = load_episodes(input_root, args.max_episodes)
    if episodes.empty:
        raise ValueError("No episodes selected")

    prepare_output(output_root, args.overwrite)
    copy_optional_files(input_root, output_root)
    v21_info = make_v21_info(info, episodes, args.video_mode)
    write_metadata(input_root, output_root, v21_info, episodes)
    convert_data(input_root, output_root, info, episodes)

    if args.video_mode == "split":
        convert_videos(input_root, output_root, info, episodes)
    else:
        print("[video] skipped; this output is metadata/parquet-only and video features will not be loadable")

    if args.validate:
        validate_dataset(output_root)

    print(f"[done] wrote {output_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
