from __future__ import annotations

import torch

from src.model.model import GraphFlowModel, TRAJECTORY_DIM


def make_batch(batch_size: int = 2) -> dict:
    history, horizon, points, condition_dim = 3, 2, 6, 12
    return {
        "entity_points": torch.randn(batch_size, history, 3, points, 3),
        "entity_point_mask": torch.ones(batch_size, history, 3, points, dtype=torch.bool),
        "scene_condition": torch.randn(batch_size, 2, condition_dim),
        "entity_role_condition": torch.randn(batch_size, 3, condition_dim),
        "gripper_closedness_history": torch.empty(batch_size, history, 1).uniform_(-1.0, 1.0),
        "target": {
            "trajectory": torch.randn(batch_size, horizon, TRAJECTORY_DIM),
            "is_complete": torch.rand(batch_size, 1),
        },
    }


def make_model() -> GraphFlowModel:
    return GraphFlowModel(
        num_points=6, history_horizon=2, future_horizon=2, condition_dim=12,
        hidden_dim=32, encoder_layers=2, flow_layers=2, num_heads=4, dropout=0.0,
    )


def test_joint_flow_trains_and_samples_xyz_action_trajectory() -> None:
    model = make_model()
    model.set_gradient_checkpointing(True)
    batch = make_batch()
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "loss_flow", "loss_complete"}
    outputs = model.sample(batch, num_steps=2)
    assert outputs["gripper_points_xyz_plan"].shape == (2, 2, 3, 3)
    assert outputs["gripper_action_plan"].shape == (2, 2, 1)
    assert outputs["is_complete"].shape == (2, 1)


def test_flow_concatenates_history_and_future_ten_dimensional_tokens() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    state = torch.randn(1, 2, TRAJECTORY_DIM)
    history_state = model._actor_history(batch)
    with torch.no_grad():
        memory, _ = model._encode(batch)
        velocity = model.flow(state, torch.full((1,), 0.5), memory, history_state)
    assert history_state.shape == (1, 3, 10)
    assert velocity.shape == (1, 2, 10)
    assert model.flow.input_projection.in_features == 10
    assert model.flow.output_projection.out_features == 10
    assert not hasattr(model.flow, "closedness_projection")

    mask = model.flow.self_attention_mask
    assert mask[:3, :3].all()
    assert not mask[:3, 3:].any()
    assert mask[3:].all()


def test_future_flow_queries_use_actor_xyz_and_closedness_history() -> None:
    model = make_model().eval()
    batch = make_batch(1)
    state = torch.randn(1, 2, TRAJECTORY_DIM)
    with torch.no_grad():
        memory, _ = model._encode(batch)
        for block in model.flow.blocks:
            block.self_norm.modulation.bias[2 * 32:].fill_(1.0)
        history_a = model._actor_history(batch)
        history_b = history_a.clone()
        history_b[:, :, :9] += 1.0
        history_b[:, :, 9:] = history_b[:, :, 9:].flip(1)
        velocity_a = model.flow(state, torch.full((1,), 0.5), memory, history_a)
        velocity_b = model.flow(state, torch.full((1,), 0.5), memory, history_b)
    assert not torch.allclose(velocity_a, velocity_b)


def test_complete_head_keeps_patient_target_and_two_scene_tokens() -> None:
    model = make_model().eval()
    with torch.no_grad():
        memory, relation = model._encode(make_batch(1))
    assert memory.shape == (1, 5, 32)
    assert relation.shape == (1, 4, 32)
    torch.testing.assert_close(relation, memory[:, 1:])
    assert model.complete_head[0].in_features == 4 * 32


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
