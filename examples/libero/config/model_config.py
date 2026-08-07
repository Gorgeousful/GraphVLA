"""Configuration for the entity-centric GraphVLA Flow model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.common.schema import validate_actor_point_indices


@dataclass
class ModelConfig:
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    num_points: int = 32
    cls_token_num: int = 4
    history_horizon: int = 9
    future_horizon: int = 10
    condition_dim: int = 384
    hidden_dim: int = 512
    encoder_layers: int = 8
    # 0: register-only global attention; 1: full-history CLS/point/scene attention.
    # global_layer_types: tuple[int, ...] = (1,1,1,1,1,1,1,1)
    global_layer_types: tuple[int, ...] = (0,0,0,0,0,0,0,0)
    # full: unrestricted global attention; role_chain: node tokens follow actor-patient-target edges.
    node_attention_mode: str = "role_chain"
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

    def __post_init__(self) -> None:
        self.actor_point_indices = validate_actor_point_indices(self.actor_point_indices)
        if self.num_points < len(self.actor_point_indices):
            raise ValueError(
                f"num_points must be at least {len(self.actor_point_indices)}, got {self.num_points}"
            )
        self.global_layer_types = tuple(self.global_layer_types)
        if len(self.global_layer_types) != self.encoder_layers:
            raise ValueError(
                f"global_layer_types must contain {self.encoder_layers} entries, "
                f"got {len(self.global_layer_types)}"
            )
        if any(layer_type not in (0, 1) for layer_type in self.global_layer_types):
            raise ValueError(
                "global_layer_types entries must be 0 (register) or 1 (dense), "
                f"got {self.global_layer_types}"
            )
        if self.node_attention_mode not in ("full", "role_chain"):
            raise ValueError(
                "node_attention_mode must be 'full' or 'role_chain', "
                f"got {self.node_attention_mode!r}"
            )

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("action_delta")
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
