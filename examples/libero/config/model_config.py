"""Configuration for the entity-centric GraphVLA Flow model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    num_points: int = 32
    cls_token_num: int = 1
    history_horizon: int = 9
    future_horizon: int = 10
    condition_dim: int = 384
    hidden_dim: int = 512
    encoder_layers: int = 8
    flow_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    sample_steps: int = 10
    flow_mode: str = "point_only"
    action_delta: bool = False
    complete_pos_weight: float = 10.0
    contact_pos_weight: float = 1.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "loss_flow": 1.0,
        "loss_complete": 0.5,
        "loss_contact": 0.5,
    })

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("action_delta")
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
