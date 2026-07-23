from __future__ import annotations

import torch

from src.dataset.transform import CustomTransform, Normalize


class LightweightModelInputTransform(CustomTransform):
    def _build_scene_condition(self, subtaskstructure, *, device):
        return torch.zeros(2, 3, device=device)

    def _build_entity_role_condition(self, *, device):
        return torch.zeros(3, 3, device=device)


def test_build_model_input_uses_absolute_xyz_and_future_aligned_action() -> None:
    frames, nodes, points = 4, 2, 4
    node_xyz = torch.arange(frames * nodes * points * 3, dtype=torch.float32).reshape(
        frames, nodes, points, 3
    )
    gripper_points_xyz = torch.arange(frames * 3 * 3, dtype=torch.float32).reshape(frames, 3, 3)
    data = {
        "node_points_xyz": node_xyz,
        "valid_node_mask": torch.tensor([[True, True], [True, True], [True, False], [True, False]]),
        "subtask_node_mask": torch.ones(frames, nodes, dtype=torch.bool),
        "gripper_points_xyz": gripper_points_xyz,
        "state": torch.tensor([[0.0] * 8, [0.0] * 6 + [0.02, -0.02], [0.0] * 6 + [0.04, -0.04], [0.0] * 8]),
        "action": torch.tensor([[0.0] * 6 + [-1.0], [0.0] * 6 + [-1.0], [0.0] * 6 + [1.0], [0.0] * 6 + [-1.0]]),
        "is_complete": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "history_horizon": 1,
        "future_horizon": 2,
        "subtaskstructure": {"nodes": [{"role": role} for role in ("actor", "patient", "target")]},
    }
    output = LightweightModelInputTransform(mode="build_model_input")(data)
    assert output["entity_points"].shape == (2, 3, 4, 3)
    assert output["entity_point_mask"].shape == (2, 3, 4)
    torch.testing.assert_close(output["entity_points"][:, 0, :3], gripper_points_xyz[:2])
    torch.testing.assert_close(output["gripper_closedness_history"], torch.tensor([[1.0], [0.0]]))
    assert output["target"]["trajectory"].shape == (2, 10)
    torch.testing.assert_close(output["target"]["trajectory"][:, :9], gripper_points_xyz[2:].flatten(1))
    torch.testing.assert_close(output["target"]["trajectory"][:, 9:], torch.tensor([[1.0], [-1.0]]))
    torch.testing.assert_close(output["target"]["is_complete"], torch.tensor([1.0]))


def test_node_level_mask_expands_to_all_points() -> None:
    data = {
        "node_points_xyz": torch.ones(3, 2, 4, 3),
        "valid_node_mask": torch.tensor([[True, False], [True, False], [True, False]]),
        "subtask_node_mask": torch.ones(3, 2, dtype=torch.bool),
        "gripper_points_xyz": torch.ones(3, 3, 3),
        "state": torch.zeros(3, 8),
        "action": torch.zeros(3, 7),
        "is_complete": torch.zeros(3),
        "history_horizon": 1,
        "future_horizon": 1,
        "subtaskstructure": {"nodes": [{"role": "actor"}, {"role": "patient"}, {"role": "target"}]},
    }
    output = LightweightModelInputTransform(mode="build_model_input")(data)
    assert output["entity_point_mask"][:, 1].all()
    assert not output["entity_point_mask"][:, 2].any()


def test_xyz_normalization_resets_masked_nodes_to_zero() -> None:
    data = {
        "node_points_xyz": torch.zeros(1, 2, 4, 3),
        "valid_node_mask": torch.tensor([[True, False]]),
    }
    transform = Normalize(
        norm_stats={"camera_xyz": {"q01": [-1.0, -1.0, 0.5], "q99": [1.0, 1.0, 1.5]}},
        field_map={"node_points_xyz": "camera_xyz"},
    )
    output = transform(data)
    torch.testing.assert_close(
        output["node_points_xyz"][0, 0, :, 2], torch.full((4,), -2.0), atol=3e-6, rtol=0.0
    )
    assert not output["node_points_xyz"][0, 1].any()
