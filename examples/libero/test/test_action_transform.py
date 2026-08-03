from __future__ import annotations

import torch

from src.common.schema import ACTION_DIM, ACTOR_POINT_INDICES
from src.dataset.transform import CustomTransform


def test_build_model_input_builds_dynamic_point_trajectory() -> None:
    num_frames = 4
    gripper = torch.zeros(num_frames, 6, 3)
    gripper[:, :, 0] = torch.arange(6)
    action = torch.arange(num_frames * ACTION_DIM, dtype=torch.float32).reshape(num_frames, ACTION_DIM)
    transform = CustomTransform(
        mode="build_model_input",
        extra={
            "use_soft": False,
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
        "node_points_xyz": torch.randn(num_frames, 2, 4, 3),
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
        result["entity_points"][:, 0, :len(ACTOR_POINT_INDICES)],
        gripper[:2, ACTOR_POINT_INDICES],
    )
    expected_points = gripper[2:4, ACTOR_POINT_INDICES]
    expected_trajectory = torch.cat(
        [expected_points.flatten(1), action[2:4, -1:]], dim=-1,
    )
    torch.testing.assert_close(result["target"]["trajectory"], expected_trajectory)
    assert result["target"]["trajectory"].shape == (2, len(ACTOR_POINT_INDICES) * 3 + 1)
    assert result["gripper_closedness_history"].shape == (2, 1)
    assert result["target"]["is_contact"].shape == (1,)


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
