"""Training configuration for LIBERO GraphVLA experiments."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    max_steps: int = 30_000
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42

    log_interval: int = 100
    save_inerval: int = 1_000
    keep_period: int = 5_000

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LIBERO_TRAINING_CONFIG = TrainingConfig()
