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
    hidden_dim: int = 256 * 3
    encoder_layers: int = 8
    flow_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    flow_mode: str = "joint"
    action_delta: bool = True
    include_future_object_point: bool = False
    point_num_train_timesteps: int = 1000
    action_num_train_timesteps: int = 1000
    point_sigma_shift: float = 5.0
    action_sigma_shift: float = 1.0
    point_sample_steps: int = 10
    action_sample_steps: int = 10
    complete_pos_weight: float = 10.0
    contact_pos_weight: float = 1.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "loss_point_flow": 1.0,
        "loss_action_flow": 1.0,
        "loss_complete": 0.5,
        "loss_contact": 0.5,
    })

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LIBERO_MODEL_CONFIG = ModelConfig(action_delta=False)
