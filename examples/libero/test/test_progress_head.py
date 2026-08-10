from __future__ import annotations

import torch

from src.model.model import GraphFlowModel


def _model() -> GraphFlowModel:
    return GraphFlowModel(
        actor_point_indices=(0, 1),
        num_points=2,
        cls_token_num=1,
        history_horizon=1,
        future_horizon=2,
        condition_dim=8,
        hidden_dim=32,
        encoder_layers=1,
        global_layer_types=(0,),
        flow_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        sample_steps=2,
        flow_mode="point_only",
    )


def _batch() -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    batch_size, history_steps, future_steps, num_points = 2, 2, 2, 2
    return {
        "entity_points": torch.randn(batch_size, history_steps, 3, num_points, 3),
        "entity_point_mask": torch.ones(
            batch_size, history_steps, 3, num_points, dtype=torch.bool,
        ),
        "scene_condition": torch.randn(batch_size, 2, 8),
        "gripper_closedness_history": torch.zeros(batch_size, history_steps, 1),
        "target": {
            "trajectory": torch.randn(batch_size, future_steps, num_points * 3 + 1),
            "subtask_progress": torch.tensor([[0.25], [0.75]]),
            "is_contact": torch.tensor([[0.0], [1.0]]),
        },
    }


def test_progress_head_forward_backward_and_sample() -> None:
    model = _model()
    batch = _batch()

    loss, losses = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(losses) == {"loss", "loss_flow", "loss_progress", "loss_contact"}
    assert model.progress_head[0].in_features == 2 * model.cls_token_num * 32
    assert any(parameter.grad is not None for parameter in model.progress_head.parameters())

    outputs = model.eval().sample(
        batch,
        noise=torch.zeros(2, 2, 2 * 3 + 1),
    )
    assert set(outputs) == {
        "point_plan", "point_plan_mask", "gripper_plan",
        "subtask_progress", "is_contact",
    }
    assert outputs["subtask_progress"].shape == (2, 1)
    assert torch.all((outputs["subtask_progress"] >= 0.0) & (outputs["subtask_progress"] <= 1.0))
