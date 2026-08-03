from __future__ import annotations

import pytest
import torch

from src.common.schema import ACTION_DIM, ACTOR_POINT_INDICES
from src.model.flow_matching import FlowMatchScheduler
from src.model.model import GraphFlowModel


def _make_model(
    *,
    flow_mode: str = "joint",
    history_horizon: int = 1,
    correlated_sigma_sampling: bool = False,
    shared_horizon_sigma_sampling: bool = False,
) -> GraphFlowModel:
    return GraphFlowModel(
        num_points=4,
        history_horizon=history_horizon,
        future_horizon=3,
        condition_dim=8,
        hidden_dim=48,
        encoder_layers=1,
        flow_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        flow_mode=flow_mode,
        point_num_train_timesteps=32,
        action_num_train_timesteps=32,
        correlated_sigma_sampling=correlated_sigma_sampling,
        shared_horizon_sigma_sampling=shared_horizon_sigma_sampling,
        point_sample_steps=2,
        action_sample_steps=2,
    )


def _make_batch(batch_size: int = 2, num_points: int = 4, history_steps: int = 2) -> dict:
    history_mask = torch.ones(batch_size, history_steps, 3, num_points, dtype=torch.bool)
    history_mask[:, :, 0, len(ACTOR_POINT_INDICES):] = False
    target_mask = torch.ones(batch_size, 3, 3, num_points, dtype=torch.bool)
    target_mask[:, :, 0, len(ACTOR_POINT_INDICES):] = False
    target_mask[0, :, 2] = False
    return {
        "entity_points": torch.randn(batch_size, history_steps, 3, num_points, 3),
        "entity_point_mask": history_mask,
        "scene_condition": torch.randn(batch_size, 2, 8),
        "target": {
            "action": torch.randn(batch_size, 3, ACTION_DIM),
            "points": torch.randn(batch_size, 3, 3, num_points, 3),
            "point_mask": target_mask,
            "is_complete": torch.zeros(batch_size, 1),
            "is_contact_future": torch.ones(batch_size, 3),
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
    assert model.flow.point_projection.in_features == len(ACTOR_POINT_INDICES) * 3 + 1
    assert model.flow.action_projection.in_features == ACTION_DIM

@pytest.mark.parametrize("flow_mode", ["joint", "point_then_action"])
def test_paired_point_target_contains_flattened_actor_and_gripper(
    monkeypatch: pytest.MonkeyPatch, flow_mode: str,
) -> None:
    captured = []
    original = FlowMatchScheduler.sample_training

    def capture_target(self, target, **kwargs):
        captured.append(target.detach().clone())
        return original(self, target, **kwargs)

    monkeypatch.setattr(FlowMatchScheduler, "sample_training", capture_target)
    batch = _make_batch()
    _make_model(flow_mode=flow_mode)(batch)

    expected = torch.cat([
        batch["target"]["points"][:, :, 0, :len(ACTOR_POINT_INDICES)].flatten(2),
        batch["target"]["action"][..., -1:],
    ], dim=-1)
    assert len(captured) == 2
    torch.testing.assert_close(captured[0], expected)


@pytest.mark.parametrize(
    ("flow_mode", "active_loss", "inactive_loss", "feature_dim"),
    [
        ("action_only", "loss_action_flow", "loss_point_flow", ACTION_DIM),
        (
            "point_only",
            "loss_point_flow",
            "loss_action_flow",
            len(ACTOR_POINT_INDICES) * 3 + 1,
        ),
    ],
)
def test_single_stream_forward_uses_only_selected_branch(
    flow_mode: str, active_loss: str, inactive_loss: str, feature_dim: int,
) -> None:
    model = _make_model(flow_mode=flow_mode)
    model.set_gradient_checkpointing()
    batch = _make_batch()
    if flow_mode == "action_only":
        batch["target"].pop("points")
        batch["target"].pop("point_mask")

    loss, losses = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(losses) == {
        "loss", "loss_point_flow", "loss_action_flow", "loss_complete", "loss_contact",
    }
    assert losses[active_loss] > 0
    assert losses[inactive_loss] == 0
    assert model.flow.stream == flow_mode.removesuffix("_only")
    assert model.flow.projection.in_features == feature_dim
    assert not hasattr(model.flow, "point_projection")
    assert not hasattr(model.flow, "action_projection")
    assert any(parameter.grad is not None for parameter in model.flow.parameters())


def test_point_only_flow_target_flattens_actor_points_and_gripper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    original = FlowMatchScheduler.sample_training

    def capture_target(self, target, **kwargs):
        captured.append(target.detach().clone())
        return original(self, target, **kwargs)

    monkeypatch.setattr(FlowMatchScheduler, "sample_training", capture_target)
    model = _make_model(flow_mode="point_only")
    batch = _make_batch()
    model(batch)

    expected = torch.cat([
        batch["target"]["points"][:, :, 0, :len(ACTOR_POINT_INDICES)].flatten(2),
        batch["target"]["action"][..., -1:],
    ], dim=-1)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)


def test_point_only_gripper_is_supervised_when_contact_loss_is_disabled() -> None:
    model = _make_model(flow_mode="point_only")
    model.weights["loss_contact"] = 0.0

    loss, losses = model(_make_batch())
    loss.backward()

    assert losses["loss_point_flow"] > 0
    gripper_output_grad = model.flow.output.weight.grad[-1]
    assert torch.isfinite(gripper_output_grad).all()
    assert gripper_output_grad.abs().sum() > 0


@pytest.mark.parametrize("flow_mode", ["action_only", "point_only"])
def test_single_stream_shared_horizon_sigma(
    monkeypatch: pytest.MonkeyPatch, flow_mode: str,
) -> None:
    timestep_ids = []
    original = FlowMatchScheduler.sample_training

    def capture_timestep_ids(self, target, **kwargs):
        ids = kwargs.get("timestep_ids")
        timestep_ids.append(None if ids is None else ids.clone())
        return original(self, target, **kwargs)

    monkeypatch.setattr(FlowMatchScheduler, "sample_training", capture_timestep_ids)
    model = _make_model(
        flow_mode=flow_mode,
        correlated_sigma_sampling=True,
        shared_horizon_sigma_sampling=True,
    )
    model(_make_batch())

    assert len(timestep_ids) == 1
    assert timestep_ids[0] is not None
    assert torch.all(timestep_ids[0] == timestep_ids[0][:, :1])


def test_single_stream_sample_returns_only_selected_plan() -> None:
    batch = _make_batch()

    action_model = _make_model(flow_mode="action_only").eval()
    action_outputs = action_model.sample(
        batch, action_noise=torch.zeros(2, 3, ACTION_DIM),
    )
    assert set(action_outputs) == {"action_plan", "is_complete", "contact_profile"}
    assert action_outputs["action_plan"].shape == (2, 3, ACTION_DIM)

    point_model = _make_model(flow_mode="point_only").eval()
    point_feature_dim = len(ACTOR_POINT_INDICES) * 3 + 1
    point_outputs = point_model.sample(
        batch, point_noise=torch.zeros(2, 3, point_feature_dim),
    )
    assert set(point_outputs) == {
        "point_plan", "point_plan_mask", "gripper_plan", "is_complete", "contact_profile",
    }
    assert point_outputs["point_plan"].shape == (2, 3, len(ACTOR_POINT_INDICES), 3)
    assert point_outputs["point_plan_mask"].shape == (2, 3, len(ACTOR_POINT_INDICES))
    assert point_outputs["gripper_plan"].shape == (2, 3)
    assert all(torch.isfinite(value).all() for value in action_outputs.values())
    assert all(torch.isfinite(value).all() for value in point_outputs.values())


@pytest.mark.parametrize(
    ("shared_horizon", "correlated"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_training_sigma_sampling_modes(
    monkeypatch: pytest.MonkeyPatch, shared_horizon: bool, correlated: bool,
) -> None:
    timestep_ids = []
    original = FlowMatchScheduler.sample_training

    def capture_timestep_ids(self, target, **kwargs):
        ids = kwargs.get("timestep_ids")
        timestep_ids.append(None if ids is None else ids.clone())
        return original(self, target, **kwargs)

    monkeypatch.setattr(FlowMatchScheduler, "sample_training", capture_timestep_ids)
    model = _make_model(
        correlated_sigma_sampling=correlated,
        shared_horizon_sigma_sampling=shared_horizon,
    )
    model(_make_batch())

    point_ids, action_ids = timestep_ids
    if not shared_horizon and not correlated:
        assert point_ids is None and action_ids is None
        return
    assert point_ids is not None and action_ids is not None
    if shared_horizon:
        assert torch.all(point_ids == point_ids[:, :1])
        assert torch.all(action_ids == action_ids[:, :1])
    if correlated:
        torch.testing.assert_close(point_ids, action_ids)


def test_sample_returns_actor_point_and_action_plans() -> None:
    model = _make_model().eval()
    outputs = model.sample(
        _make_batch(),
        point_noise=torch.zeros(2, 3, len(ACTOR_POINT_INDICES) * 3 + 1),
        action_noise=torch.zeros(2, 3, ACTION_DIM),
    )
    assert set(outputs) == {
        "action_plan", "point_plan", "point_plan_mask", "gripper_plan",
        "is_complete", "contact_profile",
    }
    assert outputs["action_plan"].shape == (2, 3, ACTION_DIM)
    assert outputs["contact_profile"].shape == (2, 3)
    assert outputs["point_plan"].shape == (2, 3, len(ACTOR_POINT_INDICES), 3)
    assert outputs["point_plan_mask"].shape == (2, 3, len(ACTOR_POINT_INDICES))
    assert outputs["gripper_plan"].shape == (2, 3)
    assert all(torch.isfinite(value).all() for value in outputs.values())
@pytest.mark.parametrize("history_horizon", [0, 1, 9])
def test_contact_head_pools_any_history_length(history_horizon: int) -> None:
    model = _make_model(history_horizon=history_horizon)
    batch = _make_batch(history_steps=history_horizon + 1)
    history_memory, _, _, _ = model._encode(batch)
    logits = model._contact_logits(history_memory)
    assert model.contact_query.shape == (1, 1, 48)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_point_then_action_mask_reads_clean_points_without_clean_actions() -> None:
    model = _make_model(flow_mode="point_then_action")
    point_length = model.future_horizon
    action_length = model.future_horizon
    point_valid = torch.ones(1, 2 * point_length, dtype=torch.bool)
    action_valid = torch.ones(1, action_length, dtype=torch.bool)
    point_mask, action_mask = model.flow._attention_masks(
        "point_then_action", point_valid, action_valid, point_length, action_length,
    )
    full = torch.cat([point_mask, action_mask], dim=1)[0]
    action_noisy_query = 2 * point_length
    point_clean_key = point_length
    point_noisy_query = 0
    action_noisy_key = 2 * point_length
    assert full[action_noisy_query, point_clean_key]
    assert not full[point_noisy_query, action_noisy_key]
    assert full.shape == (2 * point_length + action_length,) * 2



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
    assert model.contact_head[-1].out_features == model.future_horizon

    object_permutation = torch.tensor([2, 0, 3, 1])
    permuted_objects = points.clone()
    permuted_objects[:, :, 1] = points[:, :, 1, object_permutation]
    permuted_memory, _, _ = encoder(permuted_objects, mask)
    torch.testing.assert_close(memory, permuted_memory, atol=1e-6, rtol=1e-5)

    permuted_actor = points.clone()
    actor_permutation = torch.arange(len(ACTOR_POINT_INDICES)).roll(1)
    permuted_actor[:, :, 0, :len(ACTOR_POINT_INDICES)] = points[
        :, :, 0, actor_permutation
    ]
    actor_memory, _, _ = encoder(permuted_actor, mask)
    assert not torch.allclose(memory, actor_memory)


def test_future_point_flow_uses_one_actor_group_token_per_step() -> None:
    flow = _make_model().flow
    assert flow.point_projection.in_features == len(ACTOR_POINT_INDICES) * 3 + 1
    assert flow.point_output.out_features == len(ACTOR_POINT_INDICES) * 3 + 1
    assert flow._point_positions(torch.device("cpu")).shape == (flow.horizon, 3)

def test_rejects_mismatched_joint_sampling_steps() -> None:
    with pytest.raises(ValueError, match="matching point/action sample steps"):
        GraphFlowModel(hidden_dim=48, num_heads=4, point_sample_steps=2, action_sample_steps=3)
