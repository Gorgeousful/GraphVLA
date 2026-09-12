from __future__ import annotations

import json

import pytest
import torch

from examples.libero.config.model_config import ModelConfig
from src.common.schema import ACTION_DIM, GRIPPER_TCP_POINT_INDEX
from src.policy.graphpoint.model import GraphFlowModel
from src.dataset.transform import (
    CenterOnCurrentTCP,
    CustomTransform,
    Normalize,
    RandomCollapseNodePoints,
)


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


@pytest.mark.parametrize("use_numpy", [False, True])
def test_random_collapse_node_points_collapses_patient_when_target_is_disabled(
    use_numpy: bool,
) -> None:
    node_points = torch.tensor([
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0], [22.0, 0.0, 0.0], [24.0, 0.0, 0.0], [26.0, 0.0, 0.0]],
        ],
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
            [[20.0, 0.0, 0.0], [22.0, 0.0, 0.0], [24.0, 0.0, 0.0], [26.0, 0.0, 0.0]],
        ],
    ])
    original = node_points.clone()
    gripper_points = torch.zeros(2, 6, 3)
    gripper_points[0, GRIPPER_TCP_POINT_INDEX, 0] = 0.5
    gripper_points[1, GRIPPER_TCP_POINT_INDEX, 0] = 5.5
    subtask_node_mask = torch.tensor([[True, False, True], [True, False, True]])
    data = {
        "history_horizon": 1,
        "node_points_xyz": node_points,
        "gripper_points_xyz": gripper_points,
        "subtask_node_mask": subtask_node_mask,
        "subtaskstructure": {"nodes": [
            {"role": "actor"}, {"role": "patient"}, {"role": "target"},
        ]},
    }
    if use_numpy:
        data = {
            key: value.numpy() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }

    result = RandomCollapseNodePoints(
        probability=1.0,
        target_probability=0.0,
        patient_nearest_points=2,
    )(data)

    collapsed = torch.as_tensor(result["node_points_xyz"])
    expected_patient = torch.tensor([1.0, 5.0])[:, None, None].expand(-1, 4, 3).clone()
    expected_patient[..., 1:] = 0.0
    torch.testing.assert_close(collapsed[:, 0], expected_patient)
    torch.testing.assert_close(collapsed[:, 2], original[:, 2])
    torch.testing.assert_close(collapsed[:, 1], original[:, 1])


def test_random_collapse_node_points_can_collapse_target_independently() -> None:
    node_points = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    original = node_points.clone()
    result = RandomCollapseNodePoints(
        probability=0.0,
        target_probability=1.0,
    )({
        "history_horizon": 1,
        "node_points_xyz": node_points,
        "gripper_points_xyz": torch.zeros(2, 6, 3),
        "subtask_node_mask": torch.ones(2, 2, dtype=torch.bool),
        "subtaskstructure": {"nodes": [
            {"role": "actor"}, {"role": "patient"}, {"role": "target"},
        ]},
    })

    expected_target = original[:, 1].mean(dim=1, keepdim=True).expand(-1, 4, -1)
    torch.testing.assert_close(result["node_points_xyz"][:, 0], original[:, 0])
    torch.testing.assert_close(result["node_points_xyz"][:, 1], expected_target)


def test_random_collapse_node_points_keeps_legacy_shared_probability() -> None:
    node_points = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    result = RandomCollapseNodePoints(probability=1.0)({
        "history_horizon": 1,
        "node_points_xyz": node_points,
        "gripper_points_xyz": torch.zeros(2, 6, 3),
        "subtask_node_mask": torch.ones(2, 2, dtype=torch.bool),
        "subtaskstructure": {"nodes": [
            {"role": "actor"}, {"role": "patient"}, {"role": "target"},
        ]},
    })

    expected_target = node_points[:, 1].mean(dim=1, keepdim=True).expand(-1, 4, -1)
    torch.testing.assert_close(result["node_points_xyz"][:, 1], expected_target)


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
@pytest.mark.parametrize("action_mode", ["points", "abs_action", "delta_action"])
def test_build_model_input_builds_dynamic_point_trajectory(actor_point_indices: tuple[int, ...], action_mode: str) -> None:
    num_frames = 4
    gripper = torch.zeros(num_frames, 6, 3)
    gripper[:, :, 0] = torch.arange(6)
    action = torch.arange(num_frames * ACTION_DIM, dtype=torch.float32).reshape(num_frames, ACTION_DIM)
    action[:, -1] = torch.linspace(-1, 1, num_frames)
    transform = CustomTransform(
        mode="build_model_input",
        extra={
            "action_mode": action_mode,
            "actor_point_indices": actor_point_indices,
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "camera_xyz": {
                        "q01": [0.0, 0.0, 0.0],
                        "q99": [1.0, 1.0, 1.0],
                    },
                    "action": {"q01": [0.0] * 7, "q99": [2.0] * 7},
                    "state": {"q01": [0.0] * 8, "q99": [2.0] * 8},
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
        "state": torch.arange(num_frames * 8, dtype=torch.float32).reshape(num_frames, 8) / 10,
        "subtask_progress": torch.linspace(0.0, 1.0, num_frames),
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
    if action_mode == "delta_action":
        expected_trajectory = action[1:3] - 1
    elif action_mode == "abs_action":
        expected_trajectory = torch.cat([
            data["state"][2:4, :3] - 1,
            data["state"][2:4, 3:6] / torch.pi,
            action[1:3, -1:],
        ], dim=-1)
    torch.testing.assert_close(result["target"]["trajectory"], expected_trajectory, atol=2e-5, rtol=1e-5)
    assert result["target"]["trajectory"].shape == (2, len(actor_point_indices) * 3 + 1 if action_mode == "points" else 7)
    assert result["gripper_closedness_history"].shape == (2, 1)
    torch.testing.assert_close(result["target"]["subtask_progress"], torch.tensor([1.0 / 3.0]))
    if action_mode != "points":
        output_transform = CustomTransform(mode="build_model_output", extra={
            **transform.extra,
            "action_field": "action" if action_mode == "delta_action" else None,
        })
        restored = output_transform({
            "outputs": {"action_plan": result["target"]["trajectory"][None]},
        })["outputs"]["action_plan"][0]
        expected = action[1:3] if action_mode == "delta_action" else torch.cat([
            data["state"][2:4, :6], action[1:3, -1:],
        ], dim=-1)
        torch.testing.assert_close(restored, expected)


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
        ("current", 6, [0] * 6),
        ("all", 12, [-1] * 6 + [0] * 6),
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
    memory, relation_local, semantic_memory = model.encoder(
        torch.randn(2, 2, 3, 4, 3),
        torch.ones(2, 2, 3, 4, dtype=torch.bool),
        torch.randn(2, 2, 8),
    )

    assert memory.shape == (2, memory_tokens, 32)
    assert relation_local.shape == (2, 4, 32)
    assert semantic_memory.shape == (2, 2, 32)
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


def test_build_model_output_keeps_raw_gripper_when_action_stats_are_disabled() -> None:
    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {"level": "suite", "norm_stats": {}},
            "action_field": None,
        },
    )
    gripper_plan = torch.tensor([[-1.0, 1.0]])

    result = transform.build_model_output({"outputs": {"gripper_plan": gripper_plan.clone()}})

    torch.testing.assert_close(result["outputs"]["gripper_plan"], gripper_plan)


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
