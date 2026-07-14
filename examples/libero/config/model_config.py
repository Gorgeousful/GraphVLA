"""Model configuration for LIBERO GraphVLA experiments."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class ModelConfig:
    """Keyword configuration for GraphVLA.src.model.model.PointQueryModel."""
    point_dim: int = 6
    num_points: int = 32
    actor_num_points: int = 3
    set_hidden_dim: int = 384
    set_layers: int = 12
    set_heads: int = 6
    set_mlp_ratio: float = 4.0
    set_register_tokens: int = 0
    condition_dim: int = 384*3

    encoder_hidden_dim: int = 1024
    encoder_layers: int = 24
    encoder_heads: int = 16
    encoder_mlp_ratio: float = 4.0

    decoder_hidden_dim: int = 1024
    decoder_layers: int = 8
    decoder_heads: int = 16
    decoder_mlp_ratio: float = 4.0

    output_dims: dict[str, int] = field(
        default_factory=lambda: {
            "point": 6, 
            "is_complete": 1
        }
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "history_weight": 0.5, # history
            "history_actor": 1.0,
            "history_object": 1.0,
            "future_weight": 1.0, # future
            "future_actor": 1.0,
            "future_object": 0.1,
            "is_complete": 0.5, # complete
        }
    )
    residual_point_dims: tuple[int, ...] = (0, 1, 2, 4) # u,v,d,d_metirc

    num_query_types: int = 0
    num_frame_query_types: int = 0
    max_objects: int = 3
    min_frame: int = -15
    max_frame: int = 16
    attention_pattern: str | None = "interleaved_local_global"
    dropout: float = 0.1
     


    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LIBERO_MODEL_CONFIG = ModelConfig()
