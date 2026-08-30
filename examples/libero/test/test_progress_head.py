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
        flow_mode="point_only",
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
            "is_contact": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
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
        "loss_progress", "loss_contact",
    }
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
    assert outputs["is_contact"].shape == (2, 2)
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


@pytest.mark.parametrize("semantic_injection_mode", ["encoder_only", "flow_adarms_only"])
def test_all_history_contact_head_uses_one_query_per_cls(
    semantic_injection_mode: str,
) -> None:
    model = _model(
        encoder_output_type="all",
        cls_token_num=4,
        semantic_injection_mode=semantic_injection_mode,
    )
    batch = _batch()

    loss, _ = model(batch)
    loss.backward()
    memory, relation_local = model._encode(batch)
    logits = model._contact_logits(memory, relation_local)

    assert model.contact_query is not None
    assert model.contact_query.shape == (1, 4, 32)
    assert logits.shape == (2, 2)
    assert model.contact_query.grad is not None
    assert model.contact_head[0].in_features == 4 * 32


def _direct_flow_output(
    model: GraphFlowModel,
    scene_condition: torch.Tensor,
) -> torch.Tensor:
    batch_size = scene_condition.shape[0]
    trajectory_dim = model.flow.trajectory_dim
    memory = torch.randn(1, 5, 32).expand(batch_size, -1, -1).clone()
    return model.flow(
        state=torch.zeros(batch_size, 2, trajectory_dim),
        time=torch.full((batch_size,), 0.5),
        memory=memory,
        history_state=torch.zeros(batch_size, 2, trajectory_dim),
        memory_positions=torch.zeros(5, dtype=torch.long),
        scene_condition=scene_condition,
    )


def test_encoder_only_flow_has_no_direct_semantic_parameters_or_effect() -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode="encoder_only").eval()
    on_condition = torch.ones(1, 2, 8)
    right_condition = on_condition.clone()
    right_condition[:, 1] = -1.0
    memory = torch.randn(1, 5, 32)
    trajectory_dim = model.flow.trajectory_dim

    def output(scene_condition: torch.Tensor) -> torch.Tensor:
        return model.flow(
            torch.zeros(1, 2, trajectory_dim),
            torch.full((1,), 0.5),
            memory,
            torch.zeros(1, 2, trajectory_dim),
            torch.zeros(5, dtype=torch.long),
            scene_condition,
        )

    torch.testing.assert_close(output(on_condition), output(right_condition))
    assert model.flow.action_projection is None
    assert model.flow.degree_projection is None
    assert all(block.action_adapter is None for block in model.flow.blocks)
    assert all(block.degree_adapter is None for block in model.flow.blocks)


@pytest.mark.parametrize("mode", ["flow_adarms", "flow_adarms_only"])
def test_flow_adarms_direct_degree_injection_changes_velocity_and_receives_gradients(
    mode: str,
) -> None:
    torch.manual_seed(0)
    model = _model(semantic_injection_mode=mode).eval()
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


def test_flow_adarms_uses_learned_null_degree_condition() -> None:
    model = _model(semantic_injection_mode="flow_adarms")
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

    memory, relation_local = model._encode(batch)
    changed_memory, changed_relation_local = model._encode(changed_condition_batch)

    assert model.encoder.scene_token_count == 0
    assert model.encoder.action_projection is None
    assert model.encoder.degree_projection is None
    assert memory.shape == (2, 3, 32)
    assert relation_local.shape == (2, 2, 32)
    torch.testing.assert_close(memory, changed_memory)
    torch.testing.assert_close(relation_local, changed_relation_local)
