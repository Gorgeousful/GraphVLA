"""Training configuration for LIBERO GraphVLA experiments."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from pathlib import Path


@dataclass
class TrainingConfig:
    # common
    resume: bool = True
    max_steps: int = 30_000
    batch_size: int = 2

    # gradient
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_clip_l2_norm: float | None = 1.0

    # cos
    warmup_steps: int = 1_000
    peak_lr: float = 1e-5
    decay_steps: int = 100_000
    decay_lr: float = 1e-5

    # adamw
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.95)
    
    # ckpt
    ckpt_path: str | Path | None = None    
    save_dir: str | Path | None = "examples/libero/result"
    log_interval: int = 100
    save_interval: int = 1_000
    keep_period: int = 5_000

    # others
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
    wandb_name: str | None = "test"


    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LIBERO_TRAINING_CONFIG = TrainingConfig()
