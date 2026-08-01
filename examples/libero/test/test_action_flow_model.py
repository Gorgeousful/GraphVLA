from __future__ import annotations

import pytest
import torch

from src.common.schema import ACTION_DIM, ACTOR_NUM_POINTS
from src.model.model import GraphFlowModel


def _make_model(*, flow_mode: str = "joint", include_objects: bool = True) -> GraphFlowModel:
    return GraphFlowModel(
        num_points=4,
        history_horizon=1,
        future_horizon=3,
        condition_dim=8,
        hidden_dim=48,
        encoder_layers=1,
        flow_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        flow_mode=flow_mode,
        include_future_object_point=include_objects,
        point_num_train_timesteps=32,
        action_num_train_timesteps=32,
        point_sample_steps=2,
        action_sample_steps=2,
    )


def _make_batch(batch_size: int = 2, num_points: int = 4) -> dict:
    history_mask = torch.ones(batch_size, 2, 3, num_points, dtype=torch.bool)
    history_mask[:, :, 0, ACTOR_NUM_POINTS:] = False
    target_mask = torch.ones(batch_size, 3, 3, num_points, dtype=torch.bool)
    target_mask[:, :, 0, ACTOR_NUM_POINTS:] = False
    target_mask[0, :, 2] = False
    return {
        "entity_points": torch.randn(batch_size, 2, 3, num_points, 3),
        "entity_point_mask": history_mask,
        "scene_condition": torch.randn(batch_size, 2, 8),
        "target": {
            "action": torch.randn(batch_size, 3, ACTION_DIM),
            "points": torch.randn(batch_size, 3, 3, num_points, 3),
            "point_mask": target_mask,
            "is_complete": torch.zeros(batch_size, 1),
            "is_contact": torch.ones(batch_size, 1),
        },
    }


@pytest.mark.parametrize("flow_mode", ["joint", "point_then_action"])
def test_joint_point_action_forward_backward(flow_mode: str) -> None:
    model = _make_model(flow_mode=flow_mode)
    batch = _make_batch()
    loss, losses = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(losses) == {
        "loss", "loss_point_flow", "loss_action_flow", "loss_complete", "loss_contact",
    }
    assert model.flow.point_projection.in_features == 3
    assert model.flow.action_projection.in_features == ACTION_DIM


@pytest.mark.parametrize(("include_objects", "points_per_step"), [(False, 3), (True, 11)])
def test_sample_returns_point_and_action_plans(include_objects: bool, points_per_step: int) -> None:
    model = _make_model(include_objects=include_objects).eval()
    outputs = model.sample(
        _make_batch(),
        point_noise=torch.zeros(2, 3, points_per_step, 3),
        action_noise=torch.zeros(2, 3, ACTION_DIM),
    )
    assert set(outputs) == {
        "action_plan", "point_plan", "point_plan_mask", "is_complete", "is_contact",
    }
    assert outputs["action_plan"].shape == (2, 3, ACTION_DIM)
    assert outputs["point_plan"].shape == (2, 3, points_per_step, 3)
    assert outputs["point_plan_mask"].shape == (2, 3, points_per_step)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_point_then_action_mask_reads_clean_points_but_not_clean_actions() -> None:
    model = _make_model(flow_mode="point_then_action")
    point_length = model.future_horizon * model.flow.points_per_step
    action_length = model.future_horizon
    point_valid = torch.ones(1, 2 * point_length, dtype=torch.bool)
    action_valid = torch.ones(1, 2 * action_length, dtype=torch.bool)
    point_mask, action_mask = model.flow._attention_masks(
        "point_then_action", point_valid, action_valid, point_length, action_length,
    )
    full = torch.cat([point_mask, action_mask], dim=1)[0]
    action_noisy_query = 2 * point_length
    point_clean_key = point_length
    action_clean_key = 2 * point_length + action_length
    point_noisy_query = 0
    action_noisy_key = 2 * point_length
    assert full[action_noisy_query, point_clean_key]
    assert not full[action_noisy_query, action_clean_key]
    assert not full[point_noisy_query, action_noisy_key]


def test_point_loss_averages_roles_before_averaging_points() -> None:
    model = _make_model(include_objects=True)
    shape = (1, 3, model.flow.points_per_step, 3)
    prediction = torch.zeros(shape)
    target = torch.zeros(shape)
    actor_slots = model.flow.point_role_ids == 0
    prediction[:, :, actor_slots] = 1.0
    loss = model._point_flow_loss(
        prediction, target, torch.ones(shape[:-1], dtype=torch.bool), torch.ones(1, 3),
    )
    torch.testing.assert_close(loss, torch.tensor(1.0 / 3.0))


def test_encoder_uses_layer_specific_rope_and_keeps_object_points_unordered() -> None:
    torch.manual_seed(0)
    model = _make_model().eval()
    encoder = model.encoder
    assert encoder.local_blocks[0].attention.axis_dims is None
    assert encoder.global_blocks[0].attention.axis_dims == (6, 6)
    assert not any(
        hasattr(encoder, name)
        for name in (
            "actor_keypoint_embedding", "role_type_embedding", "scene_type_embedding",
            "action_projection", "degree_projection", "null_degree_token",
        )
    )
    assert model.flow.layers[0].point_block.history_attention.axis_dims == (6, 6)
    assert model.flow.layers[0].point_block.semantic_attention.axis_dims is None

    batch = _make_batch()
    points = batch["entity_points"]
    mask = batch["entity_point_mask"]
    memory, positions, relation = encoder(points, mask)
    assert memory.shape == (points.shape[0], points.shape[1] * 3, 48)
    assert positions.shape == (points.shape[1] * 3, 2)
    assert relation.shape == (points.shape[0], 2, 48)
    assert model.flow.encode_scene(batch["scene_condition"]).shape == (points.shape[0], 2, 48)
    assert model.complete_head[0].in_features == 4 * 48
    assert model.contact_head[0].in_features == 48

    object_permutation = torch.tensor([2, 0, 3, 1])
    permuted_objects = points.clone()
    permuted_objects[:, :, 1] = points[:, :, 1, object_permutation]
    permuted_memory, _, _ = encoder(permuted_objects, mask)
    torch.testing.assert_close(memory, permuted_memory, atol=1e-6, rtol=1e-5)

    permuted_actor = points.clone()
    permuted_actor[:, :, 0, :ACTOR_NUM_POINTS] = points[
        :, :, 0, torch.tensor([2, 0, 1])
    ]
    actor_memory, _, _ = encoder(permuted_actor, mask)
    assert not torch.allclose(memory, actor_memory)


def test_future_point_rope_assigns_slots_only_to_actor_keypoints() -> None:
    flow = _make_model(include_objects=True).flow
    torch.testing.assert_close(
        flow.point_slot_ids[:ACTOR_NUM_POINTS],
        torch.arange(ACTOR_NUM_POINTS, dtype=torch.float32),
    )
    assert torch.count_nonzero(flow.point_slot_ids[ACTOR_NUM_POINTS:]) == 0
    assert set(flow.point_role_ids.tolist()) == {0, 1, 2}


def test_rejects_mismatched_joint_sampling_steps() -> None:
    with pytest.raises(ValueError, match="matching point/action sample steps"):
        GraphFlowModel(hidden_dim=48, num_heads=4, point_sample_steps=2, action_sample_steps=3)
