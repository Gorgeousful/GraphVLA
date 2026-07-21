from __future__ import annotations

import torch

from src.model.model import GraphFlowModel


def make_batch(batch_size: int = 2) -> dict:
    history, horizon, points, condition_dim = 3, 2, 6, 12
    return {
        "entity_points": torch.randn(batch_size, history, 3, points, 3),
        "entity_point_mask": torch.ones(batch_size, history, 3, points, dtype=torch.bool),
        "entity_condition": torch.randn(batch_size, 3, condition_dim),
        "actor_metric_history": torch.randn(batch_size, history, 4, 3),
        "gripper_width_history": torch.rand(batch_size, history, 1),
        "robot_metric_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "target": {
            "relative_plan": torch.randn(batch_size, horizon, 4, 3),
            "metric_z_plan": torch.randn(batch_size, horizon, 4, 1),
            "gripper_width_plan": torch.rand(batch_size, horizon, 1),
            "is_complete": torch.rand(batch_size, 1),
        },
    }


def make_model() -> GraphFlowModel:
    return GraphFlowModel(
        num_points=6, history_horizon=2, future_horizon=2, condition_dim=12,
        hidden_dim=32, encoder_layers=2, flow_layers=2, num_heads=4, dropout=0.0,
    )


def test_flow_model_trains_and_samples_joint_trajectories() -> None:
    model = make_model()
    model.set_gradient_checkpointing(True)
    batch = make_batch()
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "loss", "loss_relative", "loss_metric_z", "loss_gripper_width", "loss_complete"
    }
    outputs = model.sample(batch, num_steps=2)
    assert outputs["relative_plan"].shape == (2, 2, 4, 3)
    assert outputs["metric_z_plan"].shape == (2, 2, 4, 1)
    assert outputs["gripper_width_plan"].shape == (2, 2, 1)
    assert outputs["is_complete"].shape == (2, 1)


def test_complete_head_is_independent_of_actor_and_metric_inputs() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    with torch.no_grad():
        _, relation_a = model._encode(batch)
        score_a = model.complete_head(relation_a.flatten(1))
        batch["entity_points"][:, :, 0].normal_(mean=100.0, std=10.0)
        batch["actor_metric_history"].normal_(mean=100.0, std=10.0)
        batch["gripper_width_history"].fill_(100.0)
        _, relation_b = model._encode(batch)
        score_b = model.complete_head(relation_b.flatten(1))
    torch.testing.assert_close(score_a, score_b)


def test_relative_flow_is_independent_of_robot_private_history() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    relative_state = torch.randn(1, 2, 4, 3)
    time = torch.full((1,), 0.5)
    with torch.no_grad():
        memory, _ = model._encode(batch)
        velocity_a, _ = model.relative_flow(relative_state, time, memory)
        batch["actor_metric_history"].normal_(mean=100.0, std=10.0)
        batch["gripper_width_history"].fill_(100.0)
        velocity_b, _ = model.relative_flow(relative_state, time, memory)
    torch.testing.assert_close(velocity_a, velocity_b)


def test_metric_flow_distinguishes_history_order() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    memory, _ = model._encode(batch)
    relative_state = torch.randn(1, 2, 4, 3)
    metric_state = torch.randn(1, 2, 4, 1)
    width_state = torch.randn(1, 2, 1)
    time = torch.full((1,), 0.5)
    with torch.no_grad():
        _, relative_hidden = model.relative_flow(relative_state, time, memory)
        velocity_a = model.metric_flow(
            metric_state, width_state, time, memory, relative_hidden,
            batch["actor_metric_history"], batch["gripper_width_history"],
        )
        velocity_b = model.metric_flow(
            metric_state, width_state, time, memory, relative_hidden,
            batch["actor_metric_history"].flip(1), batch["gripper_width_history"].flip(1),
        )
    assert not torch.allclose(velocity_a[0], velocity_b[0])


def test_human_only_batch_needs_no_robot_private_fields() -> None:
    model = make_model()
    batch = make_batch(1)
    del batch["actor_metric_history"]
    del batch["gripper_width_history"]
    batch["robot_metric_mask"].fill_(False)
    del batch["target"]["metric_z_plan"]
    del batch["target"]["gripper_width_plan"]

    loss, metrics = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss_metric_z"].item() == 0.0
    assert metrics["loss_gripper_width"].item() == 0.0
    assert all(parameter.grad is not None for parameter in model.metric_flow.parameters())
