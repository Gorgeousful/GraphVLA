from __future__ import annotations

from dataclasses import dataclass

import pytest

from examples.libero.config.graphpoint.model_config import ModelConfig
from src.policy.graphpoint.model import GraphFlowModel
from src.policy.registry import build_policy, resolve_policy_name


def test_build_graphpoint_policy() -> None:
    config = ModelConfig(
        actor_point_indices=(0, 1),
        num_points=2,
        cls_token_num=1,
        history_frames=(-1,),
        future_horizon=2,
        condition_dim=8,
        hidden_dim=32,
        encoder_layers=1,
        global_layer_types=(0,),
        flow_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )

    assert isinstance(build_policy(config), GraphFlowModel)


def test_legacy_config_resolves_to_graphpoint() -> None:
    @dataclass
    class LegacyConfig:
        def to_kwargs(self) -> dict:
            return {}

    assert resolve_policy_name(LegacyConfig()) == "graphpoint"


def test_unknown_policy_is_rejected() -> None:
    @dataclass
    class UnknownConfig:
        policy_name: str = "unknown"

        def to_kwargs(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Unsupported policy"):
        build_policy(UnknownConfig())
