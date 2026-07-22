from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from src.dataset.transform import CustomTransform


class LightweightModelInputTransform(CustomTransform):
    def _build_scene_condition(self, subtaskstructure, *, device):
        return torch.zeros(2, 3, device=device)

    def _build_entity_role_condition(self, *, device):
        return torch.zeros(3, 3, device=device)

    def _camera_intrinsic(self, data, *, device, dtype):
        return torch.tensor([[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)


def test_ensure_bge_preserves_existing_transform_options(monkeypatch) -> None:
    class Loader:
        @staticmethod
        def from_pretrained(_):
            return SimpleNamespace(eval=lambda: None, config=SimpleNamespace(hidden_size=3))

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=Loader, AutoTokenizer=Loader),
    )
    transform = CustomTransform(mode="build_model_input", extra={"use_delta": False})
    transform._ensure_bge()
    assert transform.extra["use_delta"] is False
    assert transform.extra["bge_dim"] == 3


def test_build_model_input_produces_entity_rays_and_residual_flow_targets() -> None:
    frames = 4
    tracks = torch.zeros(frames, 2, 4, 3)
    tracks[..., :2] = 128.0
    tracks[..., 2] = 1.0
    uv = torch.tensor([
        [128.0, 128.0], [138.0, 128.0], [118.0, 128.0],
        [128.0, 138.0], [128.0, 118.0], [128.0, 148.0],
    ]).repeat(frames, 1, 1)
    uv[2:, :, 0] += 10.0
    depth = torch.zeros(frames, 256 * 256)
    for frame in range(frames):
        for index, (u, v) in enumerate(uv[frame].long()):
            depth[frame, v * 256 + u] = 0.1 * (index + 1)
    data = {
        "node_points_track": tracks,
        "node_points_mask": torch.ones(frames, 2, dtype=torch.bool),
        "depths.depth_rel": depth,
        "gripper_uv": uv,
        "gripper_d": torch.arange(frames, dtype=torch.float32)[:, None].expand(-1, 6),
        "gripper_openness": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "action": torch.tensor([
            [0.0] * 6 + [1.0],
            [0.0] * 7,
            [0.0] * 6 + [1.0],
            [0.0] * 7,
        ]),
        "is_complete": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "history_horizon": 1,
        "future_horizon": 2,
        "subtaskstructure": {"nodes": [{"role": role} for role in ("actor", "patient", "target")]},
    }
    transform = LightweightModelInputTransform(mode="build_model_input")
    output = transform(data)
    assert output["entity_points"].shape == (2, 3, 4, 3)
    assert output["entity_point_mask"].shape == (2, 3, 4)
    assert output["entity_role_condition"].shape == (3, 3)
    assert output["actor_metric_history"].shape == (2, 4, 1)
    torch.testing.assert_close(output["gripper_closedness_history"], torch.tensor([[0.8], [0.6]]))
    torch.testing.assert_close(output["entity_points"][0, 0, 0, :2], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(output["target"]["relative_plan"][:, :, 0], torch.full((2, 4), 0.1))
    torch.testing.assert_close(output["target"]["metric_z_plan"][:, :, 0], torch.tensor([[1.0] * 4, [2.0] * 4]))
    # The first command is action[t], which drives the first t+1 geometry target.
    torch.testing.assert_close(output["target"]["gripper_action_plan"], torch.tensor([[1.0], [-1.0]]))
    torch.testing.assert_close(output["target"]["is_complete"], torch.tensor([1.0]))

    absolute_output = LightweightModelInputTransform(
        mode="build_model_input", extra={"use_delta": False}
    )(data)
    torch.testing.assert_close(
        absolute_output["target"]["relative_plan"],
        output["target"]["relative_plan"] + output["entity_points"][-1, 0, :4],
    )
    torch.testing.assert_close(
        absolute_output["target"]["metric_z_plan"],
        output["target"]["metric_z_plan"] + output["actor_metric_history"][-1],
    )

    scaled_output = LightweightModelInputTransform(
        mode="build_model_input", extra={"use_delta": False, "ray_scale": 2.0}
    )(data)
    torch.testing.assert_close(
        scaled_output["entity_points"][..., :2],
        absolute_output["entity_points"][..., :2] * 2.0,
    )
    torch.testing.assert_close(
        scaled_output["target"]["relative_plan"][..., :2],
        absolute_output["target"]["relative_plan"][..., :2] * 2.0,
    )
    torch.testing.assert_close(
        scaled_output["target"]["relative_plan"][..., 2:],
        absolute_output["target"]["relative_plan"][..., 2:],
    )

    data["node_points_track"].zero_()
    data["node_points_mask"][:, 1] = False
    masked_output = transform(data)
    assert masked_output["entity_point_mask"][:, 1].all()
    assert not masked_output["entity_point_mask"][:, 2].any()
