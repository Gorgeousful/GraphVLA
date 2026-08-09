from __future__ import annotations

import pytest
import torch

from examples.libero.config.model_config import ModelConfig
from src.common.schema import ACTION_DIM
from src.model.model import GraphFlowModel
from src.dataset.transform import CustomTransform


@pytest.mark.parametrize("actor_point_indices", [(0, 1, 2, 5), (0, 1, 2, 3, 4, 5)])
def test_build_model_input_builds_dynamic_point_trajectory(actor_point_indices: tuple[int, ...]) -> None:
    num_frames = 4
    gripper = torch.zeros(num_frames, 6, 3)
    gripper[:, :, 0] = torch.arange(6)
    action = torch.arange(num_frames * ACTION_DIM, dtype=torch.float32).reshape(num_frames, ACTION_DIM)
    transform = CustomTransform(
        mode="build_model_input",
        extra={
            "use_soft": False,
            "actor_point_indices": actor_point_indices,
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "camera_xyz": {
                        "q01": [0.0, 0.0, 0.0],
                        "q99": [1.0, 1.0, 1.0],
                    },
                },
            },
        },
    )
    transform._build_scene_condition = lambda _structure, *, device: torch.zeros(2, 8, device=device)
    data = {
        "node_points_xyz": torch.randn(num_frames, 2, 6, 3),
        "gripper_points_xyz": gripper,
        "valid_node_mask": torch.ones(num_frames, 2, dtype=torch.bool),
        "subtask_node_mask": torch.ones(num_frames, 2, dtype=torch.bool),
        "subtaskstructure": {"nodes": [{"role": "patient"}, {"role": "target"}]},
        "history_horizon": 1,
        "future_horizon": 2,
        "action": action,
        "is_complete": torch.zeros(num_frames),
        "is_contact": torch.ones(num_frames),
    }
    result = transform.build_model_input(data)
    torch.testing.assert_close(
        result["entity_points"][:, 0, :len(actor_point_indices)],
        gripper[:2, actor_point_indices],
    )
    expected_points = gripper[2:4, actor_point_indices]
    expected_trajectory = torch.cat(
        [expected_points.flatten(1), action[2:4, -1:]], dim=-1,
    )
    torch.testing.assert_close(result["target"]["trajectory"], expected_trajectory)
    assert result["target"]["trajectory"].shape == (2, len(actor_point_indices) * 3 + 1)
    assert result["gripper_closedness_history"].shape == (2, 1)
    assert result["target"]["is_contact"].shape == (1,)


@pytest.mark.parametrize("actor_point_indices", [(0, 1, 2, 5), (0, 1, 2, 3, 4, 5)])
def test_model_config_controls_actor_dimensions(actor_point_indices: tuple[int, ...]) -> None:
    config = ModelConfig(
        actor_point_indices=actor_point_indices,
        num_points=6,
        hidden_dim=48,
        encoder_layers=1,
        flow_layers=1,
        num_heads=4,
        condition_dim=8,
    )
    model = GraphFlowModel(**config.to_kwargs())

    trajectory_dim = len(actor_point_indices) * 3 + 1
    assert model.actor_point_indices == actor_point_indices
    assert model.encoder.actor_keypoint_embedding.num_embeddings == len(actor_point_indices)
    assert model.flow.input_projection.in_features == trajectory_dim
    assert model.flow.output_projection.out_features == trajectory_dim


def test_model_config_validates_encoder_output_type() -> None:
    with pytest.raises(ValueError, match="encoder_output_type"):
        ModelConfig(encoder_output_type="invalid")


@pytest.mark.parametrize(
    ("encoder_output_type", "memory_tokens", "expected_positions"),
    [
        ("current", 8, [0] * 8),
        ("all", 14, [-1] * 6 + [0] * 8),
    ],
)
def test_encoder_output_type_controls_model_memory(
    encoder_output_type: str,
    memory_tokens: int,
    expected_positions: list[int],
) -> None:
    model = GraphFlowModel(
        actor_point_indices=(0, 1, 2, 3),
        num_points=4,
        cls_token_num=2,
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
    ).eval()
    memory, relation_local = model.encoder(
        torch.randn(2, 2, 3, 4, 3),
        torch.ones(2, 2, 3, 4, dtype=torch.bool),
        torch.randn(2, 2, 8),
    )

    assert memory.shape == (2, memory_tokens, 32)
    assert relation_local.shape == (2, 4, 32)
    assert model._memory_positions(memory).tolist() == expected_positions


def test_build_model_output_unnormalizes_action_and_points() -> None:
    action_q01 = [float(index) for index in range(ACTION_DIM)]
    action_q99 = [float(index + 2) for index in range(ACTION_DIM)]
    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "camera_action": {
                        "mean": action_q01, "std": [1.0] * ACTION_DIM,
                        "q01": action_q01, "q99": action_q99,
                    },
                    "camera_xyz": {
                        "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                        "q01": [0.0, 2.0, 4.0], "q99": [2.0, 4.0, 6.0],
                    },
                },
            },
            "use_quantiles": True,
            "quantile_to_neg_one_one": True,
            "action_field": "camera_action",
        },
    )
    data = {"outputs": {
        "action_plan": torch.zeros(1, 2, ACTION_DIM),
        "point_plan": torch.zeros(1, 2, 3, 3),
        "gripper_plan": torch.zeros(1, 2),
    }}
    result = transform.build_model_output(data)
    expected_action = torch.tensor([index + 1.0 for index in range(ACTION_DIM)])
    torch.testing.assert_close(result["outputs"]["action_plan"][0, 0], expected_action)
    torch.testing.assert_close(
        result["outputs"]["point_plan"][0, 0, 0], torch.tensor([1.0, 3.0, 5.0]),
    )
    torch.testing.assert_close(result["outputs"]["gripper_plan"], torch.full((1, 2), 7.0))


def test_build_model_output_uses_configured_absolute_action_stats() -> None:
    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "absolute_camera_action": {
                        "q01": [0.0] * ACTION_DIM,
                        "q99": [2.0] * ACTION_DIM,
                    },
                },
            },
            "action_field": "absolute_camera_action",
        },
    )
    data = {"outputs": {"action_plan": torch.zeros(1, 1, ACTION_DIM)}}
    result = transform.build_model_output(data)
    torch.testing.assert_close(result["outputs"]["action_plan"], torch.ones(1, 1, ACTION_DIM))
