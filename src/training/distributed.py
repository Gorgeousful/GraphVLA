"""Distributed training helpers."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class TrainingDistributed:
    def __init__(self, *, enabled: bool = True, backend: str = "nccl") -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        self.enabled = self._setup(enabled=enabled, backend=backend)
        self.rank = torch.distributed.get_rank() if self.enabled else 0
        self.world_size = torch.distributed.get_world_size() if self.enabled else 1

    def _setup(self, *, enabled: bool, backend: str) -> bool:
        if not enabled or not torch.distributed.is_available():
            return torch.distributed.is_available() and torch.distributed.is_initialized()
        if not torch.distributed.is_initialized():
            has_torchrun_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
            if not has_torchrun_env:
                return False
            if self.device.type == "cuda" and backend == "nccl":
                torch.distributed.init_process_group(backend=backend, device_id=self.device)
            else:
                torch.distributed.init_process_group(backend=backend)
        return torch.distributed.get_world_size() > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.distributed.barrier(device_ids=[self.device.index])
        else:
            torch.distributed.barrier()

    def gather_object(self, value: Any) -> list[Any] | None:
        if not self.enabled:
            return [value]
        gathered = [None] * self.world_size if self.is_main_process else None
        torch.distributed.gather_object(value, gathered, dst=0)
        return gathered

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        if not self.enabled:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )

    def reduce_metrics(self, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            item = value.detach().float()
            if self.enabled:
                torch.distributed.all_reduce(item, op=torch.distributed.ReduceOp.AVG)
            reduced[key] = float(item.item())
        return reduced
