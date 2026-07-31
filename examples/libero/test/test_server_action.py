from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from script.server import EmbodimentAdapter, InputPreprocessor, ObservationFrame


def test_state_projection_passes_gripper_width_and_returns_six_points() -> None:
    captured = {}

    class Robot:
        def project_gripper_to_xyz(self, *, tcp_state, extrinsic, gripper_width):
            captured["tcp_state"] = tcp_state
            captured["extrinsic"] = extrinsic
            captured["gripper_width"] = gripper_width
            return np.zeros((6, 3), dtype=np.float32)

    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.robot = Robot()
    frame = ObservationFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        metric_depth=np.ones((2, 2), dtype=np.float32),
        state=np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02, -0.03]),
        intrinsic=np.eye(3),
        extrinsic=np.eye(4),
    )

    points = preprocessor._state_to_gripper_points_xyz(frame)

    assert points.shape == (6, 3)
    assert captured["gripper_width"] == pytest.approx(0.05)
    assert np.asarray(captured["tcp_state"]) == pytest.approx(frame.state[:6])


def test_online_model_input_selects_root_and_fingertips_without_closedness() -> None:
    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.num_points = 4
    preprocessor.norm_stats = {
        "camera_xyz": {
            "q01": [0.0, 0.0, 0.0],
            "q99": [1.0, 1.0, 1.0],
        },
    }
    gripper = np.zeros((6, 3), dtype=np.float32)
    gripper[:, 0] = np.arange(6, dtype=np.float32) / 10.0
    frame = {
        "tracks": np.zeros((0, 4, 3), dtype=np.float32),
        "metric_depth": np.ones((2, 2), dtype=np.float32),
        "gripper_points_xyz": gripper,
        "intrinsic": np.eye(3, dtype=np.float32),
    }
    subtask = {"nodes": [], "action_type": "lift", "action_degree": None}
    session = SimpleNamespace(
        taskstructure={"subtasks": [subtask]},
        subtask_index=0,
        active_object_indices=[],
    )

    model_input = preprocessor._build_model_input(session, [frame, frame], subtask)

    actor = np.asarray(model_input["entity_points"], dtype=np.float32)[0, :, 0, :3]
    expected = (gripper[[0, 3, 4]] * 2.0 - 1.0)[None]
    np.testing.assert_allclose(actor, np.repeat(expected, 2, axis=0), atol=1e-6)
    assert "gripper_closedness_history" not in model_input


class _TrajectoryRobot:
    def __init__(self, **_):
        pass

    def project_actor_xyz_to_gripper(self, points, **kwargs):
        self.points = np.asarray(points)
        self.kwargs = kwargs
        return np.zeros(7, dtype=np.float64), 0.0


def _request() -> dict:
    return {
        "camera.extrinsics": np.eye(4).tolist(),
        "observation.state": [0.0] * 6 + [0.02, -0.02],
    }


def test_actor_trajectory_is_fitted_and_gripper_value_is_preserved() -> None:
    points = np.arange(9, dtype=np.float64).reshape(3, 3)
    trajectory = np.concatenate([points.reshape(-1), [0.25]])
    adapter = EmbodimentAdapter(future_horizon=1, robot_cls=_TrajectoryRobot)
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0)

    actions = adapter.to_action(
        {"action_plan": [[trajectory]]},
        _request(),
        session,
    )

    np.testing.assert_allclose(adapter.robot.points, points)
    assert adapter.robot.kwargs["gripper_width"] == pytest.approx(0.04)
    assert actions[0][6] == pytest.approx(0.25)


def test_gripper_command_uses_session_deadband_hysteresis() -> None:
    adapter = EmbodimentAdapter(
        future_horizon=1, robot_cls=_TrajectoryRobot, action_mode="discrete"
    )
    session = SimpleNamespace(benchmark="libero", gripper_command=-1.0)

    def command_for(predicted_action: float) -> float:
        trajectory = [0.0] * 9 + [predicted_action]
        return adapter.to_action(
            {"action_plan": [[trajectory]]}, _request(), session
        )[0][6]

    assert command_for(0.1) == pytest.approx(-1.0)
    assert command_for(0.8) == pytest.approx(1.0)
    assert command_for(0.0) == pytest.approx(1.0)


def test_continuous_gripper_action_is_clipped() -> None:
    adapter = EmbodimentAdapter(future_horizon=1, robot_cls=_TrajectoryRobot)
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0)
    predicted = [0.0] * 9 + [1.5]

    action = adapter.to_action(
        {"action_plan": [[predicted]]},
        _request(),
        session,
    )[0]

    assert action[6] == pytest.approx(1.0)


def test_release_actions_are_zero_delta_and_open_gripper() -> None:
    adapter = EmbodimentAdapter(future_horizon=10, robot_cls=_TrajectoryRobot)
    session = SimpleNamespace(gripper_command=1.0)

    actions = adapter.release_actions(session, chunk_len=3)

    expected = np.zeros(7, dtype=np.float32)
    expected[6] = -1.0
    np.testing.assert_allclose(actions, np.repeat(expected[None], 3, axis=0))
    assert session.gripper_command == pytest.approx(-1.0)
