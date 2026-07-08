from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .transform import Compose


def make_lerobot_dataset(dataset_dir: Path, **kwargs: Any) -> LeRobotDataset:
    dataset_dir = Path(dataset_dir)
    try:
        return LeRobotDataset(repo_id=dataset_dir.name, root=dataset_dir, **kwargs)
    except TypeError:
        return LeRobotDataset(repo_id=str(dataset_dir), **kwargs)


class EpochRandomSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, *, shuffle: bool, seed: int = 0) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(len(self.dataset)))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class GenericDataset(Dataset):
    def __init__(self, data_config: Any) -> None:
        self.data_config = data_config
        self.dataset_dir = Path(data_config.dataset_dir)
        self.dataset = make_lerobot_dataset(
            self.dataset_dir,
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
        self.world_size = world_size
        self.rank = rank
        self.seed = seed

        self.dataset = GenericDataset(data_config)
        self.global_batch_size = batch_size
        self.sampler = self._build_sampler(shuffle=shuffle, drop_last=drop_last)
        self.local_batch_size = self._build_local_batch_size(batch_size)

        if persistent_workers is None:
            persistent_workers = num_workers > 0

        self.loader = TorchDataLoader(
            self.dataset,
            batch_size=self.local_batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
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
        iterator = iter(self.loader)
        for _ in range(skip_batches):
            try:
                next(iterator)
            except StopIteration:
                return
        yield from iterator

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def _build_sampler(self, *, shuffle: bool, drop_last: bool) -> Sampler[int]:
        if not self.is_distributed:
            return EpochRandomSampler(self.dataset, shuffle=shuffle, seed=self.seed)

        return DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def _build_local_batch_size(self, batch_size: int) -> int:
        if not self.is_distributed:
            return batch_size

        if batch_size % self.world_size != 0:
            raise ValueError(f"batch_size={batch_size} must be divisible by world_size={self.world_size}")
        return batch_size // self.world_size
