"""Training configuration for the task-specific LIBERO ACT baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    resume: bool = True
    max_steps: int = 30_000
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    gradient_clip_norm: float | None = 0.1
    warmup_steps: int = 0
    peak_lr: float = 1e-4
    backbone_lr: float = 1e-5
    decay_steps: int = 1
    decay_lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    ckpt_path: str | Path | None = None
    save_dir: str | Path | None = "examples/libero/result"
    log_interval: int = 100
    save_interval: int = 1_000
    keep_period: int = 0
    seed: int = 42
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    use_ddp: bool = True
    distributed_backend: str = "nccl"
    use_amp: bool = True
    compile_model: bool = False
    log_backend: str | None = "wandb"
    wandb_entity: str | None = "luokang2192-irmv"
    wandb_project: str | None = "GraphVLA"
    wandb_name: str | None = "act-task01-ep10-chunk10"

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LIBERO_TRAINING_CONFIG = TrainingConfig()
