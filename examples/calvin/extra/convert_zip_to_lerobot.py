"""
Convert a CALVIN zip dataset to LeRobot v2.1 format without extracting it.

Supported CALVIN variants:
- D     -> task_D_D
- ABC   -> task_ABC_D
- ABCD  -> task_ABCD_D
- debug -> calvin_debug_dataset

The output feature names follow the CALVIN LeRobot v2.1 depth dataset schema,
including RGB videos, depth maps, robot state, scene state, absolute actions,
and relative actions.
"""

from __future__ import annotations

import shutil
import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_HOME = Path("/data0/luokang/dataset/luokang/lerobot")

DatasetVariant = Literal["ABC", "ABCD", "D", "debug"]
Split = Literal["training", "validation"]

VARIANT_TO_ROOT: dict[str, str] = {
    "ABC": "task_ABC_D",
    "ABCD": "task_ABCD_D",
    "D": "task_D_D",
    "debug": "calvin_debug_dataset",
}


@dataclass(frozen=True)
class Args:
    zip_path: str
    variant: DatasetVariant = "D"
    repo_id: str | None = None
    dataset_root: str | None = None
    output_root: str | None = None
    fps: int = 30
    splits: Literal["training", "validation", "both"] = "training"
    max_episodes: int | None = None
    push_to_hub: bool = False
    video: bool = False


def _zip_exists(z: zipfile.ZipFile, name: str) -> bool:
    return name in z.namelist()


def _zip_join(root: str, *parts: str) -> str:
    path = "/".join(part.strip("/") for part in (root, *parts) if part.strip("/"))
    return path


def _infer_dataset_root(z: zipfile.ZipFile, args: Args) -> str:
    if args.dataset_root is not None:
        root = args.dataset_root.strip("/")
        probe = _zip_join(root, "training", "lang_annotations", "auto_lang_ann.npy")
        if _zip_exists(z, probe):
            return root
        raise FileNotFoundError(f"Could not find {probe} in {args.zip_path}")

    preferred = VARIANT_TO_ROOT[args.variant]
    probe = _zip_join(preferred, "training", "lang_annotations", "auto_lang_ann.npy")
    if _zip_exists(z, probe):
        return preferred

    suffixes = [
        "/training/lang_annotations/auto_lang_ann.npy",
        "/validation/lang_annotations/auto_lang_ann.npy",
    ]
    candidates = sorted(
        {
            name[: -len(suffix)]
            for name in z.namelist()
            for suffix in suffixes
            if name.endswith(suffix)
        }
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"Multiple CALVIN roots found in zip: {candidates}. "
            "Please pass --dataset-root explicitly."
        )
    raise FileNotFoundError(
        "Could not find */training/lang_annotations/auto_lang_ann.npy "
        f"or */validation/lang_annotations/auto_lang_ann.npy in {args.zip_path}"
    )


def _load_npy_from_zip(z: zipfile.ZipFile, name: str) -> np.ndarray:
    with z.open(name, "r") as f:
        return np.load(f, allow_pickle=True)


def _load_lang_data(z: zipfile.ZipFile, dataset_root: str, split: Split) -> dict:
    path = _zip_join(dataset_root, split, "lang_annotations", "auto_lang_ann.npy")
    return _load_npy_from_zip(z, path).item()


def _load_step_npz_from_zip(
    z: zipfile.ZipFile, dataset_root: str, split: Split, step_id: int
) -> dict[str, np.ndarray]:
    name = _zip_join(dataset_root, split, f"episode_{step_id:07d}.npz")
    with z.open(name, "r") as f:
        npz = np.load(f, allow_pickle=True)
        try:
            return {key: npz[key] for key in npz.files}
        finally:
            npz.close()


def _features(use_videos: bool) -> dict:
    image_dtype = "video" if use_videos else "image"
    return {
        "observation.images.rgb_static": {
            "dtype": image_dtype,
            "shape": (200, 200, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.rgb_gripper": {
            "dtype": image_dtype,
            "shape": (84, 84, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.depths.static": {
            "dtype": "float32",
            "shape": (200, 200),
            "names": ["height", "width"],
        },
        "observation.depths.gripper": {
            "dtype": "float32",
            "shape": (84, 84),
            "names": ["height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (15,),
            "names": {
                "axes": [f"robot_obs_{idx}" for idx in range(15)],
            },
        },
        "observation.scene_state": {
            "dtype": "float32",
            "shape": (24,),
            "names": {
                "axes": [f"scene_obs_{idx}" for idx in range(24)],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
        },
        "action.rel": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
        },
    }


def _frame_from_step(step: dict[str, np.ndarray], task: str) -> dict:
    return {
        "observation.images.rgb_static": step["rgb_static"],
        "observation.images.rgb_gripper": step["rgb_gripper"],
        "observation.depths.static": step["depth_static"].astype(np.float32),
        "observation.depths.gripper": step["depth_gripper"].astype(np.float32),
        "observation.state": step["robot_obs"].astype(np.float32),
        "observation.scene_state": step["scene_obs"].astype(np.float32),
        "action": step["actions"].astype(np.float32),
        "action.rel": step["rel_actions"].astype(np.float32),
        "task": task,
    }


def _selected_splits(args: Args) -> list[Split]:
    if args.splits == "both":
        return ["training", "validation"]
    return [args.splits]


def main(args: Args) -> None:
    with zipfile.ZipFile(args.zip_path, "r") as z:
        dataset_root = _infer_dataset_root(z, args)
        repo_id = args.repo_id or f"{dataset_root}"
        output_root = Path(args.output_root) if args.output_root else LEROBOT_HOME / repo_id

        if output_root.exists():
            shutil.rmtree(output_root)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_root,
            robot_type="franka_panda",
            fps=args.fps,
            features=_features(args.video),
            use_videos=args.video,
            image_writer_threads=10,
            image_writer_processes=5,
        )

        saved_episodes = 0
        selected_splits = _selected_splits(args)
        total_available = 0
        split_lang_data: dict[str, dict] = {}
        for split in selected_splits:
            lang_path = _zip_join(dataset_root, split, "lang_annotations", "auto_lang_ann.npy")
            if not _zip_exists(z, lang_path):
                raise FileNotFoundError(f"Could not find {lang_path} in {args.zip_path}")
            lang_data = _load_lang_data(z, dataset_root, split)
            split_lang_data[split] = lang_data
            total_available += len(lang_data["info"]["indx"])

        total_episodes = min(total_available, args.max_episodes) if args.max_episodes is not None else total_available
        with tqdm(total=total_episodes, desc="episodes", unit="ep") as episode_bar:
            for split in selected_splits:
                lang_data = split_lang_data[split]
                ep_start_end_ids = lang_data["info"]["indx"]
                lang_ann = lang_data["language"]["ann"]

                iterator = enumerate(ep_start_end_ids)
                for i, (start_idx, end_idx) in iterator:
                    if args.max_episodes is not None and saved_episodes >= args.max_episodes:
                        break

                    task = str(lang_ann[i])
                    step_start = int(start_idx)
                    step_end = int(end_idx)
                    step_iter = range(step_start, step_end + 1)
                    episode_bar.set_postfix(split=split, task=task[:32], frames=step_end - step_start + 1)
                    for step_idx in tqdm(
                        step_iter,
                        desc=f"{split} ep {saved_episodes}",
                        unit="frame",
                        leave=False,
                    ):
                        step = _load_step_npz_from_zip(z, dataset_root, split, step_idx)
                        dataset.add_frame(_frame_from_step(step, task))

                    dataset.save_episode()
                    saved_episodes += 1
                    episode_bar.update(1)

                if args.max_episodes is not None and saved_episodes >= args.max_episodes:
                    break

        print(
            {
                "zip_path": args.zip_path,
                "dataset_root": dataset_root,
                "repo_id": repo_id,
                "output_root": str(output_root),
                "fps": args.fps,
                "episodes": saved_episodes,
                "video": args.video,
            }
        )

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["calvin", dataset_root],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANT_TO_ROOT), default="D")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--splits", choices=["training", "validation", "both"], default="training")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--video", action="store_true", help="Encode RGB observations as mp4 videos instead of storing images in parquet.")
    ns = parser.parse_args()
    return Args(
        zip_path=ns.zip_path,
        variant=ns.variant,
        repo_id=ns.repo_id,
        dataset_root=ns.dataset_root,
        output_root=ns.output_root,
        fps=ns.fps,
        splits=ns.splits,
        max_episodes=ns.max_episodes,
        push_to_hub=ns.push_to_hub,
        video=ns.video,
    )


if __name__ == "__main__":
    main(parse_args())
