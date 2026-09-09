from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
        hf_transform_to_torch,
    )
except ModuleNotFoundError:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
        hf_transform_to_torch,
    )

from .transform import Compose


class _MetadataWithoutEpisodeStats(LeRobotDatasetMetadata):
    def load_metadata(self) -> None:
        self.info = lerobot_dataset_module.load_info(self.root)
        lerobot_dataset_module.check_version_compatibility(
            self.repo_id, self._version, lerobot_dataset_module.CODEBASE_VERSION,
        )
        self.tasks, self.task_to_task_index = lerobot_dataset_module.load_tasks(self.root)
        self.episodes = lerobot_dataset_module.load_episodes(self.root)
        self.episodes_stats = {episode_index: {} for episode_index in self.episodes}
        self.stats = {}


_METADATA_CLASS_LOCK = threading.Lock()


class _EpisodeIndexedLeRobotDataset(LeRobotDataset):
    """Keep LeRobot's episode index addressable by original episode id."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        load_episode_stats = kwargs.pop("load_episode_stats", True)
        if load_episode_stats:
            super().__init__(*args, **kwargs)
        else:
            with _METADATA_CLASS_LOCK:
                original = lerobot_dataset_module.LeRobotDatasetMetadata
                lerobot_dataset_module.LeRobotDatasetMetadata = _MetadataWithoutEpisodeStats
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    lerobot_dataset_module.LeRobotDatasetMetadata = original
        if self.episodes:
            compact_index = self.episode_data_index
            index_size = max(self.meta.total_episodes, max(self.episodes) + 1)
            expanded_index = {
                key: torch.zeros(index_size, dtype=value.dtype)
                for key, value in compact_index.items()
            }
            for subset_index, episode_index in enumerate(self.episodes):
                for key, value in expanded_index.items():
                    value[episode_index] = compact_index[key][subset_index]
            self.episode_data_index = expanded_index


class _FeatureOnlyLeRobotDataset(_EpisodeIndexedLeRobotDataset):
    """LeRobot integration hook that skips camera decoding for feature-only training."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        requested = set(self.delta_indices or {})
        requested.update({"episode_index", "frame_index", "task_index", "timestamp", "index"})
        unused = [name for name in self.hf_dataset.column_names if name not in requested]
        if unused:
            self.hf_dataset = self.hf_dataset.remove_columns(unused)

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
    ) -> dict[str, torch.Tensor]:
        return {}

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        grouped_keys: dict[tuple[int, ...], list[str]] = {}
        video_keys = set(self.meta.video_keys)
        for key, indices in query_indices.items():
            if key not in video_keys:
                grouped_keys.setdefault(tuple(indices), []).append(key)

        materialized: dict[str, torch.Tensor] = {}
        for indices, keys in grouped_keys.items():
            selected = self.hf_dataset.select(list(indices)).with_format("numpy")
            for key in keys:
                materialized[key] = torch.from_numpy(selected[key].copy())

        return {key: materialized[key] for key in query_indices if key in materialized}


class _FeatureSubsetLeRobotDataset(_EpisodeIndexedLeRobotDataset):
    """Load only requested parquet columns while retaining video decoding."""

    def __init__(
        self,
        *args: Any,
        feature_keys: tuple[str, ...],
        video_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        self.feature_keys = feature_keys
        self.selected_video_keys = set(video_keys) if video_keys is not None else None
        super().__init__(*args, **kwargs)

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
    ) -> dict[str, torch.Tensor]:
        if self.selected_video_keys is not None:
            query_timestamps = {
                key: value
                for key, value in query_timestamps.items()
                if key in self.selected_video_keys
            }
        return super()._query_videos(query_timestamps, ep_idx)

    def load_hf_dataset(self):
        required = {
            *self.feature_keys,
            "episode_index",
            "frame_index",
            "task_index",
            "timestamp",
            "index",
        }
        if self.episodes is None:
            data_files: Any = str(self.root / "data")
            dataset = load_dataset(
                "parquet", data_dir=data_files, split="train", columns=sorted(required),
            )
        else:
            data_files = [
                str(self.root / self.meta.get_data_file_path(episode_index))
                for episode_index in self.episodes
            ]
            dataset = load_dataset(
                "parquet", data_files=data_files, split="train", columns=sorted(required),
            )
        dataset.set_transform(hf_transform_to_torch)
        return dataset


def make_lerobot_dataset(
    dataset_dir: Path,
    *,
    load_videos: bool = True,
    video_keys: tuple[str, ...] | None = None,
    feature_keys: tuple[str, ...] | None = None,
    load_episode_stats: bool = True,
    **kwargs: Any,
) -> LeRobotDataset:
    dataset_dir = Path(dataset_dir)
    kwargs["load_episode_stats"] = load_episode_stats
    if feature_keys is not None:
        dataset_cls = _FeatureSubsetLeRobotDataset
        kwargs["feature_keys"] = feature_keys
        kwargs["video_keys"] = video_keys
    else:
        dataset_cls = _EpisodeIndexedLeRobotDataset if load_videos else _FeatureOnlyLeRobotDataset
    try:
        return dataset_cls(repo_id=dataset_dir.name, root=dataset_dir, **kwargs)
    except TypeError:
        return dataset_cls(repo_id=str(dataset_dir), **kwargs)


class GlobalBatchSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        *,
        global_batch_size: int,
        local_batch_size: int,
        world_size: int,
        rank: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if not drop_last:
            raise ValueError("GlobalBatchSampler currently requires drop_last=True for stable cross-world-size resume")
        if global_batch_size <= 0 or local_batch_size <= 0:
            raise ValueError(
                f"batch sizes must be positive, got global={global_batch_size}, local={local_batch_size}"
            )
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if global_batch_size % world_size != 0:
            raise ValueError(f"global_batch_size={global_batch_size} must be divisible by world_size={world_size}")
        expected_local = global_batch_size // world_size
        if local_batch_size != expected_local:
            raise ValueError(
                f"local_batch_size must equal global_batch_size/world_size, got {local_batch_size} vs {expected_local}"
            )

        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.local_batch_size = local_batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.skip_batches = 0

    def __iter__(self) -> Iterator[int]:
        dataset_size = len(self.dataset)
        num_global_batches = self._num_global_batches()
        if num_global_batches <= 0:
            return

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(dataset_size, generator=generator)
        else:
            indices = torch.arange(dataset_size)
        indices = indices[: num_global_batches * self.global_batch_size]

        start = self.rank * self.local_batch_size
        end = start + self.local_batch_size
        for global_batch in indices.reshape(num_global_batches, self.global_batch_size)[self.skip_batches:]:
            yield from global_batch[start:end].tolist()

    def __len__(self) -> int:
        return max(0, self._num_global_batches() - self.skip_batches) * self.local_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip_batches(self, skip_batches: int) -> None:
        if skip_batches < 0:
            raise ValueError(f"skip_batches must be non-negative, got {skip_batches}")
        self.skip_batches = skip_batches

    def _num_global_batches(self) -> int:
        return len(self.dataset) // self.global_batch_size


class GenericDataset(Dataset):
    def __init__(self, data_config: Any) -> None:
        self.data_config = data_config
        self.dataset_dir = Path(data_config.dataset_dir)
        self.dataset = make_lerobot_dataset(
            self.dataset_dir,
            load_videos=getattr(data_config, "load_videos", True),
            video_keys=getattr(data_config, "camera_keys", None),
            feature_keys=getattr(data_config, "feature_keys", None),
            load_episode_stats=getattr(data_config, "load_episode_stats", True),
            episodes=data_config.episodes,
            video_backend=data_config.video_backend,
            delta_timestamps=self._build_delta_timestamps(getattr(data_config, "horizon", None)),
        )
        self.indices = self._build_task_indices(getattr(data_config, "tasks", None))
        self.transform = Compose(getattr(data_config, "transforms", ()))

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.indices is not None:
            index = self.indices[index]
        return self.transform(self.dataset[index])

    def _build_delta_timestamps(self, horizon: dict[str, list[int]] | None) -> dict[str, list[float]] | None:
        if not horizon:
            return None

        info_path = self.dataset_dir / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        fps = float(info["fps"])

        return {key: [frame_offset / fps for frame_offset in frame_offsets] for key, frame_offsets in horizon.items()}

    def _build_task_indices(self, tasks: list[int] | None) -> list[int] | None:
        if tasks is None:
            return None

        selected_tasks = set(tasks)
        if hasattr(self.dataset, "hf_dataset") and "task_index" in self.dataset.hf_dataset.column_names:
            task_indices = self.dataset.hf_dataset["task_index"]
        else:
            task_indices = [self.dataset[idx]["task_index"] for idx in range(len(self.dataset))]

        def to_int(value: Any) -> int:
            if hasattr(value, "item"):
                return int(value.item())
            return int(value)

        return [idx for idx, task_index in enumerate(task_indices) if to_int(task_index) in selected_tasks]


class GenericDataLoader:
    def __init__(
        self,
        data_config: Any,
        batch_size: int,
        *,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool | None = None,
        collate_fn: Any | None = None,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_config = data_config
        self.is_distributed = distributed
        self.world_size = world_size if distributed else 1
        self.rank = rank if distributed else 0
        self.seed = seed

        self.dataset = GenericDataset(data_config)
        self.global_batch_size = batch_size
        self.local_batch_size = self._build_local_batch_size(batch_size)
        self.drop_last = drop_last
        self.sampler = self._build_sampler(shuffle=shuffle, drop_last=drop_last)

        if persistent_workers is None:
            persistent_workers = num_workers > 0

        mp_context = "spawn" if num_workers > 0 else None
        self.loader_kwargs = {
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "collate_fn": collate_fn,
        }
        self.loader = TorchDataLoader(
            self.dataset,
            batch_size=self.local_batch_size,
            shuffle=False,
            sampler=self.sampler,
            drop_last=drop_last,
            **self.loader_kwargs,
        )

    def __iter__(self) -> Iterator[Any]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def resume_position(self, skip_batches: int) -> tuple[int, int]:
        if skip_batches < 0:
            raise ValueError(f"skip_batches must be non-negative, got {skip_batches}")
        batches_per_epoch = len(self)
        if batches_per_epoch <= 0:
            raise ValueError("dataloader has no batches; check dataset size and batch_size")
        return divmod(skip_batches, batches_per_epoch)

    def iter_epoch(self, epoch: int, skip_batches: int = 0) -> Iterator[Any]:
        self.set_epoch(epoch)
        self.sampler.set_skip_batches(skip_batches)
        try:
            yield from self.loader
        finally:
            self.sampler.set_skip_batches(0)

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def _build_sampler(self, *, shuffle: bool, drop_last: bool) -> GlobalBatchSampler:
        return GlobalBatchSampler(
            self.dataset,
            global_batch_size=self.global_batch_size,
            local_batch_size=self.local_batch_size,
            world_size=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
        )

    def _build_local_batch_size(self, batch_size: int) -> int:
        if not self.is_distributed:
            return batch_size

        if batch_size % self.world_size != 0:
            raise ValueError(f"batch_size={batch_size} must be divisible by world_size={self.world_size}")
        return batch_size // self.world_size
