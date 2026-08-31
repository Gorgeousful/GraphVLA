from __future__ import annotations

import pytest
import torch

from src.model.model import GraphFlowModel
from src.model.temporal import AdaRMSNorm


def _model(
    *, encoder_output_type: str = "current", cls_token_num: int = 1,
    gripper_flow_weight: float = 1.0,
    semantic_injection_mode: str = "encoder_only",
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
        encoder_output_type=encoder_output_type,
        global_layer_types=(0,),
        flow_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        sample_steps=2,
        gripper_flow_weight=gripper_flow_weight,
        semantic_injection_mode=semantic_injection_mode,
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
        },
    }


def test_progress_head_forward_backward_and_sample() -> None:
    model = _model()
    batch = _batch()

    loss, losses = model(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(losses) == {
        "loss", "loss_flow", "loss_flow_points", "loss_flow_gripper",
        "loss_progress",
    }
    assert model.progress_head[0].in_features == (2 * model.cls_token_num + 4) * 32
    assert any(parameter.grad is not None for parameter in model.progress_head.parameters())

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


def _direct_flow_output(
    model: GraphFlowModel,
    scene_condition: torch.Tensor,
) -> torch.Tensor:
    batch_size = scene_condition.shape[0]
    trajectory_dim = model.flow.trajectory_dim
    shape_memory = torch.randn(1, 5, 32).expand(batch_size, -1, -1).clone()
    center_memory = torch.randn(1, 3, 32).expand(batch_size, -1, -1).clone()
    return model.flow(
        state=torch.zeros(batch_size, 2, trajectory_dim),
        time=torch.full((batch_size,), 0.5),
        shape_memory=shape_memory,
        center_memory=center_memory,
        history_state=torch.zeros(batch_size, 2, trajectory_dim),
        shape_positions=torch.zeros(5, dtype=torch.long),
        center_positions=torch.zeros(3, dtype=torch.long),
        scene_condition=scene_condition,
    )


def test_encoder_only_flow_has_no_direct_semantic_parameters_or_effect() -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode="encoder_only").eval()
    on_condition = torch.ones(1, 2, 8)
    right_condition = on_condition.clone()
    right_condition[:, 1] = -1.0
    shape_memory = torch.randn(1, 5, 32)
    center_memory = torch.randn(1, 3, 32)
    trajectory_dim = model.flow.trajectory_dim

    def output(scene_condition: torch.Tensor) -> torch.Tensor:
        return model.flow(
            torch.zeros(1, 2, trajectory_dim),
            torch.full((1,), 0.5),
            shape_memory,
            center_memory,
            torch.zeros(1, 2, trajectory_dim),
            torch.zeros(5, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            scene_condition,
        )

    torch.testing.assert_close(output(on_condition), output(right_condition))
    assert model.flow.action_projection is None
    assert model.flow.degree_projection is None
    assert all(block.action_adapter is None for block in model.flow.blocks)
    assert all(block.degree_adapter is None for block in model.flow.blocks)


def test_flow_adarms_only_direct_degree_injection_changes_velocity_and_receives_gradients() -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode="flow_adarms_only").eval()
    for module in model.flow.modules():
        if isinstance(module, AdaRMSNorm):
            torch.nn.init.normal_(module.modulation.weight, std=0.02)

    scene_condition = torch.ones(2, 2, 8)
    scene_condition[1, 1] = -1.0
    output = _direct_flow_output(model, scene_condition)

    assert not torch.allclose(output[0], output[1])
    output.sum().backward()
    assert model.flow.degree_projection is not None
    assert model.flow.degree_projection.weight.grad is not None
    assert model.flow.blocks[0].degree_adapter is not None
    assert model.flow.blocks[0].degree_adapter.weight.grad is not None


def test_flow_adarms_only_uses_learned_null_degree_condition() -> None:
    model = _model(semantic_injection_mode="flow_adarms_only")
    scene_condition = torch.randn(2, 2, 8)
    scene_condition[:, 1] = 0.0

    _, degree_condition = model.flow._semantic_conditions(scene_condition, batch_size=2)

    assert degree_condition is not None
    assert model.flow.null_degree_condition is not None
    torch.testing.assert_close(
        degree_condition,
        model.flow.null_degree_condition.expand(2, -1),
    )


def test_flow_adarms_only_removes_encoder_scene_tokens_and_semantic_parameters() -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode="flow_adarms_only").eval()
    batch = _batch()
    changed_condition_batch = dict(batch)
    changed_condition_batch["scene_condition"] = -batch["scene_condition"]

    memory, centers, relation_shape, relation_center = model._encode(batch)
    changed_memory, changed_centers, changed_relation_shape, changed_relation_center = (
        model._encode(changed_condition_batch)
    )

    assert model.encoder.scene_token_count == 0
    assert model.encoder.action_projection is None
    assert model.encoder.degree_projection is None
    assert memory.shape == (2, 3, 32)
    assert centers.shape == (2, 3, 32)
    assert relation_shape.shape == (2, 2, 32)
    assert relation_center.shape == (2, 2, 32)
    torch.testing.assert_close(memory, changed_memory)
    torch.testing.assert_close(centers, changed_centers)
    torch.testing.assert_close(relation_shape, changed_relation_shape)
    torch.testing.assert_close(relation_center, changed_relation_center)


def test_shape_memory_is_translation_invariant_while_center_memory_moves() -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode="flow_adarms_only").eval()
    batch = _batch()
    translated_batch = dict(batch)
    translation = torch.tensor([0.7, -0.4, 0.2]).view(1, 1, 1, 1, 3)
    translated_batch["entity_points"] = batch["entity_points"] + translation

    shape, center, relation_shape, relation_center = model._encode(batch)
    moved_shape, moved_center, moved_relation_shape, moved_relation_center = model._encode(
        translated_batch
    )

    torch.testing.assert_close(shape, moved_shape, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(relation_shape, moved_relation_shape, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(center, moved_center)
    assert not torch.allclose(relation_center, moved_relation_center)


def test_progress_uses_action_and_degree_conditions() -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode="flow_adarms_only").eval()
    batch = _batch()
    _, _, relation_shape, relation_center = model._encode(batch)
    changed_condition = batch["scene_condition"].clone()
    changed_condition[:, 1] *= -1.0

    progress = model._progress(relation_shape, relation_center, batch["scene_condition"])
    changed_progress = model._progress(relation_shape, relation_center, changed_condition)

    assert not torch.allclose(progress, changed_progress)


def test_flow_reads_center_before_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(semantic_injection_mode="flow_adarms_only").eval()
    block = model.flow.blocks[0]
    calls: list[str] = []
    center_forward = block.center_cross_attention.forward
    shape_forward = block.shape_cross_attention.forward

    def capture_center(*args, **kwargs):
        calls.append("center")
        return center_forward(*args, **kwargs)

    def capture_shape(*args, **kwargs):
        calls.append("shape")
        return shape_forward(*args, **kwargs)

    monkeypatch.setattr(block.center_cross_attention, "forward", capture_center)
    monkeypatch.setattr(block.shape_cross_attention, "forward", capture_shape)
    _direct_flow_output(model, torch.randn(1, 2, 8))

    assert calls == ["center", "shape"]
