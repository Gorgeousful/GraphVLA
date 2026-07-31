from __future__ import annotations

import torch

from src.common.schema import ACTION_DIM, ACTOR_POINT_INDICES
from src.dataset.transform import CustomTransform


def test_build_model_input_selects_root_and_fingertips_and_future_actions() -> None:
    num_frames = 4
    gripper = torch.zeros(num_frames, 6, 3)
    gripper[:, :, 0] = torch.arange(6)
    action = torch.arange(num_frames * ACTION_DIM, dtype=torch.float32).reshape(
        num_frames, ACTION_DIM
    )
    transform = CustomTransform(mode="build_model_input", extra={"use_soft": False})
    transform._build_scene_condition = lambda _structure, *, device: torch.zeros(
        2, 8, device=device
    )
    data = {
        "node_points_xyz": torch.randn(num_frames, 2, 4, 3),
        "gripper_points_xyz": gripper,
        "valid_node_mask": torch.ones(num_frames, 2, dtype=torch.bool),
        "subtask_node_mask": torch.ones(num_frames, 2, dtype=torch.bool),
        "subtaskstructure": {
            "nodes": [{"role": "patient"}, {"role": "target"}],
        },
        "history_horizon": 1,
        "future_horizon": 2,
        "action": action,
        "is_complete": torch.zeros(num_frames),
        "is_contact": torch.ones(num_frames),
    }

    result = transform.build_model_input(data)

    expected_actor = gripper[:2, ACTOR_POINT_INDICES]
    torch.testing.assert_close(result["entity_points"][:, 0, :3], expected_actor)
    torch.testing.assert_close(result["target"]["action"], action[2:4])
    assert result["target"]["action"].shape == (2, ACTION_DIM)
    assert "gripper_closedness_history" not in result
    assert "trajectory" not in result["target"]


def test_build_model_output_unnormalizes_camera_action() -> None:
    q01 = [float(index) for index in range(ACTION_DIM)]
    q99 = [float(index + 2) for index in range(ACTION_DIM)]
    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "camera_action": {
                        "mean": q01,
                        "std": [1.0] * ACTION_DIM,
                        "q01": q01,
                        "q99": q99,
                    },
                },
            },
            "use_quantiles": True,
            "quantile_to_neg_one_one": True,
        },
    )
    data = {"outputs": {"action_plan": torch.zeros(1, 2, ACTION_DIM)}}

    result = transform.build_model_output(data)

    expected = torch.tensor([index + 1.0 for index in range(ACTION_DIM)])
    torch.testing.assert_close(result["outputs"]["action_plan"][0, 0], expected)
