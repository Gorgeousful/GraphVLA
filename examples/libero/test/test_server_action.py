from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from script.server import EmbodimentAdapter


@pytest.mark.parametrize(("predicted_action", "expected_action"), [(0.8, 1.0), (-0.8, -1.0)])
def test_to_action_recovers_xyz_pose_and_direct_command(predicted_action, expected_action) -> None:
    captured = {}

    class Robot:
        def project_xyz_to_gripper(self, xyz, *, gripper_width, return_residual, **_):
            captured["xyz"] = xyz
            captured["width"] = gripper_width
            assert return_residual
            return np.asarray([0.0] * 6 + [gripper_width], dtype=np.float64), 0.001

    adapter = EmbodimentAdapter(future_horizon=3, robot_cls=lambda **_: Robot())
    xyz_step = [[0.0,0.0,0.5],[0.1,0.0,0.5],[-0.1,0.0,0.5]]
    outputs = {
        "gripper_points_xyz_plan": [[xyz_step, xyz_step, xyz_step]],
        "gripper_action_plan": [[[predicted_action], [predicted_action], [predicted_action]]],
    }
    request = {
        "observation.state": [0.0] * 6 + [0.02, -0.02],
        "camera.extrinsics": np.eye(4).tolist(),
    }
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0, frame_index=0)
    actions, widths = adapter.to_action(outputs, {}, request, session)
    assert [action[6] for action in actions] == pytest.approx([expected_action] * 3)
    assert widths == pytest.approx([0.04] * 3)
    assert captured["xyz"].shape == (3, 3)
    assert captured["width"] == pytest.approx(0.04)


def test_gripper_command_uses_session_deadband_hysteresis() -> None:
    class Robot:
        def project_xyz_to_gripper(self, xyz, *, gripper_width, **_):
            return np.asarray([0.0] * 6 + [gripper_width], dtype=np.float64), 0.0

    adapter = EmbodimentAdapter(future_horizon=1, robot_cls=lambda **_: Robot())
    session = SimpleNamespace(benchmark="libero", gripper_command=-1.0, frame_index=0)
    request = {
        "observation.state": [0.0] * 6 + [0.02, -0.02],
        "camera.extrinsics": np.eye(4).tolist(),
    }

    def command_for(predicted_action: float) -> float:
        outputs = {
            "gripper_points_xyz_plan": [[[[0.0,0.0,0.5],[0.1,0.0,0.5],[-0.1,0.0,0.5]]]],
            "gripper_action_plan": [[[predicted_action]]],
        }
        actions, _ = adapter.to_action(outputs, {}, request, session)
        return actions[0][6]

    assert command_for(0.1) == pytest.approx(-1.0)
    assert command_for(0.8) == pytest.approx(1.0)
    assert command_for(0.0) == pytest.approx(1.0)


def test_release_actions_hold_latest_observed_pose_and_open_gripper() -> None:
    adapter = EmbodimentAdapter(future_horizon=10, robot_cls=lambda **_: None)
    session = SimpleNamespace(gripper_command=1.0)
    latest_state = np.asarray([0.1, -0.2, 0.3, 0.2, -0.1, 0.4, 0.01, -0.01])
    request = {
        "observation.state": np.stack([
            np.zeros(8, dtype=np.float64),
            latest_state,
        ]).tolist(),
    }

    actions = adapter.release_actions(request, session, chunk_len=3)
    expected = np.zeros(7, dtype=np.float64)
    expected[:6] = latest_state[:6]
    expected = adapter._to_libero_pose(expected)
    expected[6] = -1.0

    assert np.asarray(actions) == pytest.approx(np.repeat(expected[None], 3, axis=0))
    assert session.gripper_command == pytest.approx(-1.0)
