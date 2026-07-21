from __future__ import annotations

import torch

from src.dataset.transform import CustomTransform


class LightweightModelInputTransform(CustomTransform):
    def _build_conditions(
        self,
        subtaskstructure,
        *,
        object_roles,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(2, 3, device=device), torch.zeros(1, 3, device=device)


def test_build_model_input_uses_history_as_memory_and_queries_only_future_actor() -> None:
    num_frames = 4
    node_points_track = torch.zeros(num_frames, 2, 2, 3)
    node_points_track[..., 0] = 64.0
    node_points_track[..., 1] = 64.0
    node_points_track[..., 2] = 1.0
    gripper_uv = torch.tensor(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0], [60.0, 60.0]]
    ).repeat(num_frames, 1, 1)
    depth_rel = torch.zeros(num_frames, 256 * 256)
    for point_index, (u, v) in enumerate(gripper_uv[0].long()):
        depth_rel[:, v * 256 + u] = 0.1 * (point_index + 1)
    action = torch.zeros(num_frames, 7)
    action[:, -1] = torch.tensor([1.0, 0.0, 0.25, 0.75])
    data = {
        "node_points_track": node_points_track,
        "node_points_mask": torch.ones(num_frames, 2, dtype=torch.bool),
        "depths.depth_rel": depth_rel,
        "gripper_uv": gripper_uv,
        "gripper_d": torch.full((num_frames, 6), 0.25),
        "gripper_openness": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "action": action,
        "is_complete": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "history_horizon": 1,
        "future_horizon": 2,
        "subtaskstructure": {
            "nodes": [
                {"role": "actor"},
                {"role": "patient"},
                {"role": "target"},
            ]
        },
    }

    output = LightweightModelInputTransform(mode="build_model_input")(data)

    assert output["point_feats"].shape == (2, 2, 2, 8)
    assert output["actor_feats"].shape == (2, 1, 4, 8)
    torch.testing.assert_close(
        output["actor_feats"][0, 0, :, 2],
        torch.tensor([0.1, 0.2, 0.3, 0.45]),
    )
    torch.testing.assert_close(output["point_feats"][..., 6:], torch.zeros(2, 2, 2, 2))
    torch.testing.assert_close(
        output["actor_feats"][:, 0, :, 6],
        torch.tensor([[0.1] * 4, [0.2] * 4]),
    )
    torch.testing.assert_close(output["actor_feats"][..., 7], torch.ones(2, 1, 4))

    torch.testing.assert_close(output["object_id"], torch.zeros(8, dtype=torch.long))
    torch.testing.assert_close(
        output["point_id"],
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    torch.testing.assert_close(output["frame_id"], torch.tensor([1] * 4 + [2] * 4))
    torch.testing.assert_close(output["frame_query_frame_id"], torch.tensor([0]))
    torch.testing.assert_close(output["target"]["is_complete"], torch.tensor([1.0]))
    torch.testing.assert_close(output["actor_query_frame_id"], torch.tensor([1, 2]))
    torch.testing.assert_close(
        output["target"]["gripper_openness"],
        torch.tensor([[0.3], [0.4]]),
    )
    torch.testing.assert_close(output["target"]["gripper_action"], torch.tensor([[0.5], [-0.5]]))
    assert "gripper_openness_mask" not in output["target"]
    assert "gripper_action_mask" not in output["target"]
