from __future__ import annotations

import torch

from src.model.model import GraphFlowModel


def make_batch(batch_size: int = 2) -> dict:
    history, horizon, points, condition_dim = 3, 2, 6, 12
    return {
        "entity_points": torch.randn(batch_size, history, 3, points, 3),
        "entity_point_mask": torch.ones(batch_size, history, 3, points, dtype=torch.bool),
        "scene_condition": torch.randn(batch_size, 2, condition_dim),
        "entity_role_condition": torch.randn(batch_size, 3, condition_dim),
        "actor_metric_history": torch.randn(batch_size, history, 4, 1),
        "gripper_closedness_history": torch.empty(batch_size, history, 1).uniform_(-1.0, 1.0),
        "robot_metric_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "target": {
            "relative_plan": torch.randn(batch_size, horizon, 4, 3),
            "metric_z_plan": torch.randn(batch_size, horizon, 4, 1),
            "gripper_action_plan": torch.empty(batch_size, horizon, 1).uniform_(-1.0, 1.0),
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
        "loss", "loss_relative", "loss_private", "loss_metric_z",
        "loss_gripper_action", "loss_complete"
    }
    torch.testing.assert_close(
        metrics["loss_private"],
        metrics["loss_metric_z"] + metrics["loss_gripper_action"],
    )
    outputs = model.sample(batch, num_steps=2)
    assert outputs["relative_plan"].shape == (2, 2, 4, 3)
    assert outputs["metric_z_plan"].shape == (2, 2, 4, 1)
    assert outputs["gripper_action_plan"].shape == (2, 2, 1)
    assert outputs["is_complete"].shape == (2, 1)


def test_sample_only_restores_current_coordinates_for_delta_model() -> None:
    delta_model = make_model().eval()
    absolute_model = make_model().eval()
    absolute_model.load_state_dict(delta_model.state_dict())
    absolute_model.use_delta = False
    batch = make_batch(1)
    noise = torch.randn(1, 2 * 4 * 3 + 2 * 5)

    delta_outputs = delta_model.sample(batch, num_steps=1, noise=noise.clone())
    absolute_outputs = absolute_model.sample(batch, num_steps=1, noise=noise.clone())

    current_relative = batch["entity_points"][:, -1, 0, :4][:, None]
    current_metric = batch["actor_metric_history"][:, -1][:, None]
    torch.testing.assert_close(
        delta_outputs["relative_plan"],
        absolute_outputs["relative_plan"] + current_relative,
    )
    torch.testing.assert_close(
        delta_outputs["metric_z_plan"],
        absolute_outputs["metric_z_plan"] + current_metric,
    )


def test_relative_flow_uses_one_token_per_future_step() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    state = torch.randn(1, 2, 4, 3)
    time = torch.full((1,), 0.5)
    with torch.no_grad():
        memory, _ = model._encode(batch)
        velocity, hidden = model.relative_flow(state, time, memory)
    assert velocity.shape == (1, 2, 4, 3)
    assert hidden.shape == (1, 2, 32)


def test_private_flow_uses_one_joint_value_token_per_future_step() -> None:
    model = make_model()
    assert model.private_flow.input_projection.in_features == 5
    assert model.private_flow.history_projection.in_features == 5
    assert model.private_flow.output_projection.out_features == 5


def test_all_temporal_order_uses_rope_instead_of_learned_embeddings() -> None:
    model = make_model()
    assert not hasattr(model.encoder, "time_embedding")
    assert not hasattr(model.relative_flow, "horizon_embedding")
    assert not hasattr(model.private_flow, "horizon_embedding")
    assert not hasattr(model.private_flow, "history_time_embedding")


def test_complete_head_uses_final_task_relation_tokens() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    with torch.no_grad():
        memory, relation = model._encode(batch)
    assert memory.shape == (1, 5, 32)
    assert relation.shape == (1, 4, 32)
    torch.testing.assert_close(relation, memory[:, 1:])
    assert model.complete_head[0].in_features == 4 * 32


def test_relative_flow_is_independent_of_robot_private_history() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    relative_state = torch.randn(1, 2, 4, 3)
    time = torch.full((1,), 0.5)
    with torch.no_grad():
        memory, _ = model._encode(batch)
        velocity_a, _ = model.relative_flow(relative_state, time, memory)
        batch["actor_metric_history"].normal_(mean=100.0, std=10.0)
        batch["gripper_closedness_history"].fill_(100.0)
        velocity_b, _ = model.relative_flow(relative_state, time, memory)
    torch.testing.assert_close(velocity_a, velocity_b)


def test_robot_private_flow_distinguishes_history_order() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    memory, _ = model._encode(batch)
    relative_state = torch.randn(1, 2, 4, 3)
    private_state = torch.randn(1, 2, 5)
    time = torch.full((1,), 0.5)
    with torch.no_grad():
        # OpenPI-style AdaRMS residual gates are zero-initialized. Open the
        # cross-attention gates here so this unit test isolates temporal RoPE.
        for block in model.private_flow.blocks:
            block.cross_norm.modulation.bias[2 * 32:].fill_(1.0)
        _, relative_hidden = model.relative_flow(relative_state, time, memory)
        velocity_a = model.private_flow(
            private_state, time, memory, relative_hidden,
            batch["actor_metric_history"], batch["gripper_closedness_history"],
        )
        velocity_b = model.private_flow(
            private_state, time, memory, relative_hidden,
            batch["actor_metric_history"].flip(1), batch["gripper_closedness_history"].flip(1),
        )
    assert velocity_a.shape == (1, 2, 5)
    assert not torch.allclose(velocity_a, velocity_b)


def test_human_only_batch_needs_no_robot_private_fields() -> None:
    model = make_model()
    batch = make_batch(1)
    del batch["actor_metric_history"]
    del batch["gripper_closedness_history"]
    batch["robot_metric_mask"].fill_(False)
    del batch["target"]["metric_z_plan"]
    del batch["target"]["gripper_action_plan"]

    loss, metrics = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["loss_private"].item() == 0.0
    assert metrics["loss_metric_z"].item() == 0.0
    assert metrics["loss_gripper_action"].item() == 0.0
    assert all(parameter.grad is not None for parameter in model.private_flow.parameters())


def test_object_point_permutation_does_not_change_encoder_memory() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    with torch.no_grad():
        memory_a, relation_a = model._encode(batch)
        permutation = torch.randperm(batch["entity_points"].shape[3])
        for entity_index in (1, 2):
            batch["entity_points"][:, :, entity_index] = batch["entity_points"][:, :, entity_index, permutation]
            batch["entity_point_mask"][:, :, entity_index] = batch["entity_point_mask"][:, :, entity_index, permutation]
        memory_b, relation_b = model._encode(batch)
    torch.testing.assert_close(memory_a, memory_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(relation_a, relation_b, atol=1e-5, rtol=1e-5)
