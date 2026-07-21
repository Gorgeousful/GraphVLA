"""Configuration for the entity-centric GraphVLA Flow model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    num_points: int = 32
    history_horizon: int = 19
    future_horizon: int = 10
    condition_dim: int = 384 * 3
    hidden_dim: int = 512
    encoder_layers: int = 8
    flow_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    sample_steps: int = 10
    weights: dict[str, float] = field(default_factory=lambda: {
        "loss_relative": 1.0,
        "loss_metric_z": 1.0,
        "loss_gripper_width": 1.0,
        "loss_complete": 0.5,
    })

    @property
    def min_frame(self) -> int:
        return -self.history_horizon

    @property
    def max_frame(self) -> int:
        return self.future_horizon

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LIBERO_MODEL_CONFIG = ModelConfig()
