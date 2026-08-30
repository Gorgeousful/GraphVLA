from __future__ import annotations

import json

import pytest
import torch

from examples.libero.config.model_config import ModelConfig
from src.common.schema import ACTION_DIM, GRIPPER_TCP_POINT_INDEX
from src.model.model import GraphFlowModel
from src.dataset.transform import (
    CenterOnCurrentTCP,
    CustomTransform,
    Normalize,
    RandomCollapseNodePoints,
    SubtaskBoundryPadding,
)


def test_subtask_boundary_padding_clamps_contact_targets() -> None:
    data = {
        "history_horizon": 1,
        "subtask_id": torch.tensor([0, 0, 1, 1]),
        "is_contact": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "is_contact_soft": torch.tensor([0.1, 0.9, 0.2, 0.3]),
    }

    result = SubtaskBoundryPadding()(data)

    torch.testing.assert_close(result["is_contact"], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(result["is_contact_soft"], torch.tensor([0.1, 0.9, 0.9, 0.9]))


def test_center_on_current_tcp_uses_one_origin_for_the_entire_window() -> None:
    gripper = torch.arange(4 * 6 * 3, dtype=torch.float32).reshape(4, 6, 3)
    node_points = torch.arange(4 * 2 * 2 * 3, dtype=torch.float32).reshape(4, 2, 2, 3)
    valid_node_mask = torch.ones(4, 2, dtype=torch.bool)
    valid_node_mask[0, 1] = False
    origin = gripper[1, GRIPPER_TCP_POINT_INDEX].clone()
    original_gripper = gripper.clone()
    original_nodes = node_points.clone()

    result = CenterOnCurrentTCP()({
        "history_horizon": 1,
        "node_points_xyz": node_points,
        "gripper_points_xyz": gripper,
        "valid_node_mask": valid_node_mask,
    })

    torch.testing.assert_close(result["tcp_origin"], origin)
    torch.testing.assert_close(result["gripper_points_xyz"], original_gripper - origin)
    torch.testing.assert_close(
        result["gripper_points_xyz"][1, GRIPPER_TCP_POINT_INDEX], torch.zeros(3),
    )
    torch.testing.assert_close(result["node_points_xyz"][2, 0], original_nodes[2, 0] - origin)
    torch.testing.assert_close(result["node_points_xyz"][0, 1], torch.zeros(2, 3))


def test_random_collapse_node_points_collapses_selected_nodes_across_window() -> None:
    node_points = torch.arange(3 * 3 * 4 * 3, dtype=torch.float32).reshape(3, 3, 4, 3)
    original = node_points.clone()
    subtask_node_mask = torch.tensor([
        [False, False, False],
        [True, False, True],
        [False, False, False],
    ])

    result = RandomCollapseNodePoints(probability=1.0)({
        "history_horizon": 1,
        "node_points_xyz": node_points,
        "subtask_node_mask": subtask_node_mask,
    })

    expected_centers = original.mean(dim=2, keepdim=True).expand_as(original)
    torch.testing.assert_close(result["node_points_xyz"][:, 0], expected_centers[:, 0])
    torch.testing.assert_close(result["node_points_xyz"][:, 2], expected_centers[:, 2])
    torch.testing.assert_close(result["node_points_xyz"][:, 1], original[:, 1])


def test_normalize_loads_stats_from_json_path(tmp_path) -> None:
    path = tmp_path / "norm_stats_suite.json"
    path.write_text(json.dumps({
        "level": "suite",
        "norm_stats": {
            "tcp_relative_xyz": {
                "q01": [-1.0, -2.0, -3.0],
                "q99": [1.0, 2.0, 3.0],
            },
        },
    }))
    transform = Normalize(
        norm_stats=path,
        field_map={"points": "tcp_relative_xyz"},
    )

    result = transform({"points": torch.zeros(2, 3)})

    torch.testing.assert_close(result["points"], torch.zeros(2, 3))


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
        "subtask_progress": torch.linspace(0.0, 1.0, num_frames),
        "is_contact": torch.tensor([1.0, 1.0, 0.0, 1.0]),
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
    torch.testing.assert_close(result["target"]["subtask_progress"], torch.tensor([1.0 / 3.0]))
    torch.testing.assert_close(
        result["target"]["is_contact"], torch.tensor([0.0, 1.0]),
    )


@pytest.mark.parametrize("actor_point_indices", [(0, 1, 2, 5), (0, 1, 2, 3, 4, 5)])
def test_model_config_controls_actor_dimensions(actor_point_indices: tuple[int, ...]) -> None:
    config = ModelConfig(
        actor_point_indices=actor_point_indices,
        num_points=6,
        hidden_dim=48,
        encoder_layers=1,
        global_layer_types=(0,),
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
    assert "point_coordinate_frame" not in config.to_kwargs()


def test_model_config_validates_encoder_output_type() -> None:
    with pytest.raises(ValueError, match="encoder_output_type"):
        ModelConfig(encoder_output_type="invalid")


@pytest.mark.parametrize("point_coordinate_frame", ["tcp_relative", "tcp_absolute", "camera"])
def test_model_config_accepts_point_coordinate_frames(point_coordinate_frame: str) -> None:
    assert ModelConfig(point_coordinate_frame=point_coordinate_frame).point_coordinate_frame == point_coordinate_frame


def test_model_config_rejects_invalid_point_coordinate_frame() -> None:
    with pytest.raises(ValueError, match="point_coordinate_frame"):
        ModelConfig(point_coordinate_frame="invalid")


def test_model_config_validates_gripper_flow_weight() -> None:
    with pytest.raises(ValueError, match="gripper_flow_weight"):
        ModelConfig(gripper_flow_weight=0.0)


def test_model_config_derives_history_horizon_from_frames() -> None:
    config = ModelConfig(history_frames=[-20, -10, -5, -2, -1], history_horizon=99)

    assert config.history_horizon == 5
    assert "history_frames" not in config.to_kwargs()


@pytest.mark.parametrize("history_frames", [[-1, -2], [-2, -2], [-2, 0], [-2, 1]])
def test_model_config_rejects_invalid_history_frames(history_frames: list[int]) -> None:
    with pytest.raises(ValueError, match="history_frames"):
        ModelConfig(history_frames=history_frames)


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


def test_build_model_output_restores_tcp_relative_points_to_camera_coordinates() -> None:
    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "tcp_relative_xyz": {
                        "q01": [-1.0, -2.0, -3.0],
                        "q99": [1.0, 2.0, 3.0],
                    },
                },
            },
            "point_stats_field": "tcp_relative_xyz",
            "point_coordinate_frame": "tcp_relative",
        },
    )
    data = {
        "outputs": {"point_plan": torch.zeros(1, 2, 3, 3)},
        "batch": {"tcp_origin": torch.tensor([[0.1, 0.2, 0.3]])},
    }

    result = transform.build_model_output(data)

    torch.testing.assert_close(
        result["outputs"]["point_plan"],
        torch.tensor([0.1, 0.2, 0.3]).expand(1, 2, 3, 3),
    )


def test_build_model_output_keeps_tcp_absolute_points_in_camera_coordinates() -> None:
    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "camera_xyz": {
                        "q01": [0.0, 2.0, 4.0],
                        "q99": [2.0, 4.0, 6.0],
                    },
                },
            },
            "point_stats_field": "camera_xyz",
            "point_coordinate_frame": "tcp_absolute",
        },
    )

    result = transform.build_model_output({
        "outputs": {"point_plan": torch.zeros(1, 2, 3, 3)},
    })

    torch.testing.assert_close(
        result["outputs"]["point_plan"],
        torch.tensor([1.0, 3.0, 5.0]).expand(1, 2, 3, 3),
    )


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
