from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from script.server import EmbodimentAdapter


@pytest.mark.parametrize(("predicted_action", "expected_action"), [(0.8, 1.0), (-0.8, -1.0)])
def test_to_action_recovers_ray_depth_pose_and_direct_command(predicted_action, expected_action) -> None:
    captured = {}
    class Robot:
        def project_ray_depth_to_gripper(self, ray_depth, *, gripper_width, **_):
            captured["ray_depth"] = ray_depth
            captured["width"] = gripper_width
            return np.asarray([0.0] * 6 + [gripper_width], dtype=np.float64)
    adapter = EmbodimentAdapter(future_horizon=3, robot_cls=lambda **_: Robot())
    relative_step = [[0.0,0.0,0.1],[0.1,0.0,0.1],[-0.1,0.0,0.1],[0.0,0.1,0.1]]
    metric_step = [[0.5],[0.5],[0.5],[0.5]]
    outputs = {
        "relative_plan": [[relative_step, relative_step, relative_step]],
        "metric_z_plan": [[metric_step, metric_step, metric_step]],
        "gripper_action_plan": [[[predicted_action], [predicted_action], [predicted_action]]],
    }
    request = {
        "observation.state": [0.0] * 6 + [0.02, -0.02],
        "camera.extrinsics": np.eye(4).tolist(),
    }
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0)
    actions, widths = adapter.to_action(outputs, {}, request, session)
    assert [action[6] for action in actions] == pytest.approx([expected_action] * 3)
    assert widths == pytest.approx([0.04] * 3)
    assert captured["ray_depth"].shape == (4, 3)
    assert captured["width"] == pytest.approx(0.04)


def test_gripper_command_uses_session_hysteresis() -> None:
    class Robot:
        def project_ray_depth_to_gripper(self, ray_depth, *, gripper_width, **_):
            return np.asarray([0.0] * 6 + [gripper_width], dtype=np.float64)

    adapter = EmbodimentAdapter(future_horizon=1, robot_cls=lambda **_: Robot())
    session = SimpleNamespace(benchmark="libero", gripper_command=-1.0)
    relative_step = [[0.0, 0.0, 0.1]] * 4
    metric_step = [[0.5]] * 4
    request = {
        "observation.state": [0.0] * 6 + [0.02, -0.02],
        "camera.extrinsics": np.eye(4).tolist(),
    }

    def command_for(predicted_action: float) -> float:
        outputs = {
            "relative_plan": [[relative_step]],
            "metric_z_plan": [[metric_step]],
            "gripper_action_plan": [[[predicted_action]]],
        }
        actions, _ = adapter.to_action(outputs, {}, request, session)
        return actions[0][6]

    assert command_for(0.1) == pytest.approx(-1.0)
    assert command_for(0.8) == pytest.approx(1.0)
    assert command_for(0.0) == pytest.approx(1.0)
    assert session.gripper_command == pytest.approx(1.0)
