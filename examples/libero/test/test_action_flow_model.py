from __future__ import annotations

import pytest
import torch

from src.model.model import ACTION_DIM, GraphFlowModel


def _make_model() -> GraphFlowModel:
    return GraphFlowModel(
        num_points=5,
        history_horizon=1,
        future_horizon=3,
        condition_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        flow_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        sample_steps=2,
    )


def _make_batch(batch_size: int = 2) -> dict:
    return {
        "entity_points": torch.randn(batch_size, 2, 3, 5, 3),
        "entity_point_mask": torch.ones(batch_size, 2, 3, 5, dtype=torch.bool),
        "scene_condition": torch.randn(batch_size, 2, 8),
        "target": {
            "action": torch.randn(batch_size, 3, ACTION_DIM),
            "is_complete": torch.zeros(batch_size, 1),
            "is_contact": torch.ones(batch_size, 1),
        },
    }


def test_forward_uses_sixteen_dimensional_actor_action_target() -> None:
    model = _make_model()
    batch = _make_batch()

    loss, losses = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(losses) == {"loss", "loss_flow", "loss_complete", "loss_contact"}
    assert model.flow.input_projection.in_features == ACTION_DIM
    assert model.flow.output_projection.out_features == ACTION_DIM
    assert "gripper_closedness_history" not in batch


def test_sample_returns_only_action_plan_and_unchanged_head_outputs() -> None:
    model = _make_model().eval()
    batch = _make_batch()
    noise = torch.zeros(2, 3, ACTION_DIM)

    outputs = model.sample(batch, num_steps=2, noise=noise)

    assert set(outputs) == {"action_plan", "is_complete", "is_contact"}
    assert outputs["action_plan"].shape == (2, 3, ACTION_DIM)
    assert outputs["is_complete"].shape == (2, 1)
    assert outputs["is_contact"].shape == (2, 1)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_rejects_seven_dimensional_robot_action_target() -> None:
    model = _make_model()
    batch = _make_batch()
    batch["target"]["action"] = torch.randn(2, 3, 7)

    with pytest.raises(ValueError, match=r"Expected target action .*16"):
        model(batch)
