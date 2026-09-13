"""Configuration for the entity-centric GraphVLA Flow model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.common.schema import validate_actor_point_indices


@dataclass
class ModelConfig:
    policy_name: str = "graphpoint"
    # abs_action: future measured world XYZ + axis-angle + gripper (7D).
    # delta_action: recorded LIBERO delta commands + gripper (7D).
    action_mode: str = "points"  # points, abs_action, delta_action
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    num_points: int = 32
    cls_token_num: int = 4
    history_frames: tuple[int, ...] = (-9, -8, -7, -6, -5, -4, -3, -2, -1)
    history_horizon: int | None = None
    future_horizon: int = 10
    condition_dim: int = 384
    hidden_dim: int = 512 # 512
    encoder_layers: int = 8 # 8
    # current: expose only the latest entity CLS tokens; all: expose CLS tokens from every history step.
    encoder_output_type: str = "current"
    # 0: register-only global attention; 1: full-history CLS/point/scene attention.
    # global_layer_types: tuple[int, ...] = (1,1,1,1,1,1,1,1)
    global_layer_types: tuple[int, ...] = (0,0,0,0,0,0,0,0)
    # full: unrestricted global attention; role_chain: node tokens follow actor-patient-target edges.
    node_attention_mode: str = "role_chain"
    flow_layers: int = 6 # 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    sample_steps: int = 10
    action_delta: bool = False  # Derived from action_mode in __post_init__.
    point_coordinate_frame: str = "tcp_relative" # "tcp_relative"
    # A missing target is replaced by the subtask-initial patient, for progress only.
    progresshead_input: list[str] = field(default_factory=lambda: ["patient", "target"])
    gripper_flow_weight: float = 1.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "loss_flow": 1.0,
        "loss_progress": 1.0,
    })

    def __post_init__(self) -> None:
        if self.action_mode not in ("points", "abs_action", "delta_action"):
            raise ValueError("action_mode must be 'points', 'abs_action', or 'delta_action'")
        self.action_delta = self.action_mode == "delta_action"
        if self.policy_name != "graphpoint":
            raise ValueError(
                f"Expected policy_name='graphpoint', got {self.policy_name!r}"
            )
        self.history_frames = tuple(self.history_frames)
        if any(type(frame) is not int or frame >= 0 for frame in self.history_frames):
            raise ValueError(
                f"history_frames must contain only negative integers, got {self.history_frames}"
            )
        if any(left >= right for left, right in zip(self.history_frames, self.history_frames[1:])):
            raise ValueError(
                f"history_frames must be strictly increasing, got {self.history_frames}"
            )
        self.history_horizon = len(self.history_frames)
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
        if self.encoder_output_type not in ("current", "all"):
            raise ValueError(
                "encoder_output_type must be 'current' or 'all', "
                f"got {self.encoder_output_type!r}"
            )
        if self.gripper_flow_weight <= 0:
            raise ValueError(
                f"gripper_flow_weight must be positive, got {self.gripper_flow_weight}"
            )
        if self.point_coordinate_frame not in ("camera", "tcp_absolute", "tcp_relative"):
            raise ValueError(
                "point_coordinate_frame must be 'tcp_absolute' or 'tcp_relative' "
                "('camera' is kept as a legacy alias for 'tcp_absolute'), "
                f"got {self.point_coordinate_frame!r}"
            )

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("policy_name")
        kwargs.pop("action_delta")
        kwargs.pop("history_frames")
        kwargs.pop("point_coordinate_frame")
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
