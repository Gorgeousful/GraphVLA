from __future__ import annotations

import torch

from src.policy.graphpoint.model import GraphFlowModel


def _model(
    *, cls_token_num: int = 1,
    gripper_flow_weight: float = 1.0,
) -> GraphFlowModel:
    return GraphFlowModel(
        actor_point_indices=(0, 1),
        num_points=2,
        cls_token_num=cls_token_num,
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
        gripper_flow_weight=gripper_flow_weight,
    )


def _batch() -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    batch_size, history_steps, future_steps, num_points = 2, 2, 2, 2
    return {
        "entity_points": torch.randn(batch_size, history_steps, 3, num_points, 3),
        "entity_point_mask": torch.ones(
            batch_size, history_steps, 3, num_points, dtype=torch.bool,
        ),
        "scene_condition": torch.stack([
            torch.stack([torch.ones(8), torch.zeros(8)]),
            torch.stack([torch.ones(8), torch.ones(8)]),
        ]),
        "gripper_closedness_history": torch.zeros(batch_size, history_steps, 1),
        "target": {
            "trajectory": torch.randn(batch_size, future_steps, num_points * 3 + 1),
            "subtask_progress": torch.tensor([[0.25], [0.75]]),
        },
    }


def test_progress_head_forward_backward_and_sample() -> None:
    model = _model()
    model.set_gradient_checkpointing(True)
    batch = _batch()

    loss, losses = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(losses) == {
        "loss", "loss_flow", "loss_flow_points", "loss_flow_gripper",
        "loss_progress",
    }
    assert model.progress_head.input_projection.in_features == 2 * model.cls_token_num * 32
    assert any(parameter.grad is not None for parameter in model.progress_head.parameters())
    assert all(
        block.self_norm.task_modulation.weight.grad is not None
        and torch.count_nonzero(block.self_norm.task_modulation.weight.grad) > 0
        for block in model.flow.blocks
    )
    assert all(
        not hasattr(block, "semantic_attention")
        for block in model.flow.blocks
    )
    assert model.progress_head.norm.modulation[-1].weight.grad is not None
    assert torch.count_nonzero(model.progress_head.norm.modulation[-1].weight.grad) > 0

    outputs = model.eval().sample(
        batch,
        noise=torch.zeros(2, 2, 2 * 3 + 1),
    )
    assert set(outputs) == {
        "point_plan", "point_plan_mask", "gripper_plan",
        "subtask_progress",
    }
    assert outputs["subtask_progress"].shape == (2, 1)
    assert torch.all((outputs["subtask_progress"] >= 0.0) & (outputs["subtask_progress"] <= 1.0))


def test_gripper_flow_weight_reweights_only_the_last_trajectory_dimension() -> None:
    point_dimensions = 2 * 3
    for gripper_flow_weight in (1.0, 4.0):
        _, losses = _model(gripper_flow_weight=gripper_flow_weight)(_batch())
        expected = (
            point_dimensions * losses["loss_flow_points"]
            + gripper_flow_weight * losses["loss_flow_gripper"]
        ) / (point_dimensions + gripper_flow_weight)

        torch.testing.assert_close(losses["loss_flow"], expected)
