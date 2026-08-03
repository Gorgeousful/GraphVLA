from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from script.server import (
    EmbodimentAdapter,
    InputPreprocessor,
    ObservationFrame,
    TopLevelTaskPlanner,
)
from src.common.schema import ACTOR_POINT_INDICES


def test_contact_profile_text_is_limited_to_executed_actions() -> None:
    text = TopLevelTaskPlanner._contact_profile_text(
        {"contact_profile": [[0.1, 0.2, 0.3, 0.4]]}, limit=2,
    )

    assert text == "contact_score[2]=[0.100, 0.200]"


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


def test_online_model_input_selects_configured_rigid_actor_points() -> None:
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

    actor = np.asarray(model_input["entity_points"], dtype=np.float32)[0, :, 0, :len(ACTOR_POINT_INDICES)]
    expected = (gripper[list(ACTOR_POINT_INDICES)] * 2.0 - 1.0)[None]
    np.testing.assert_allclose(actor, np.repeat(expected, 2, axis=0), atol=1e-6)
    assert "gripper_closedness_history" not in model_input


def test_camera_action_is_rotated_to_world_frame() -> None:
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    camera_action = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25]
    adapter = EmbodimentAdapter(future_horizon=1)
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0)

    actions = adapter.to_action(
        {"action_plan": [[camera_action]]},
        {"camera.extrinsics": extrinsic.tolist()},
        session,
    )

    np.testing.assert_allclose(actions[0][:3], [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(actions[0][3:6], [-1.0, 0.0, 0.0], atol=1e-7)
    assert actions[0][6] == pytest.approx(0.25)


def test_gripper_command_uses_session_deadband_hysteresis() -> None:
    adapter = EmbodimentAdapter(future_horizon=1, action_mode="discrete")
    session = SimpleNamespace(benchmark="libero", gripper_command=-1.0)
    request = {"camera.extrinsics": np.eye(4).tolist()}

    def command_for(predicted_action: float) -> float:
        action = [0.0] * 6 + [predicted_action]
        return adapter.to_action({"action_plan": [[action]]}, request, session)[0][6]

    assert command_for(0.1) == pytest.approx(-1.0)
    assert command_for(0.8) == pytest.approx(1.0)
    assert command_for(0.0) == pytest.approx(1.0)


def test_continuous_delta_action_is_clipped() -> None:
    adapter = EmbodimentAdapter(future_horizon=1)
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0)
    predicted = [2.0, -2.0, 0.5, 1.5, -1.5, 0.0, 1.5]

    action = adapter.to_action(
        {"action_plan": [[predicted]]},
        {"camera.extrinsics": np.eye(4).tolist()},
        session,
    )[0]

    assert action == pytest.approx([1.0, -1.0, 0.5, 1.0, -1.0, 0.0, 1.0])


def test_release_actions_are_zero_delta_and_open_gripper() -> None:
    adapter = EmbodimentAdapter(future_horizon=10)
    session = SimpleNamespace(gripper_command=1.0)

    actions = adapter.release_actions(session, chunk_len=3)

    expected = np.zeros(7, dtype=np.float32)
    expected[6] = -1.0
    np.testing.assert_allclose(actions, np.repeat(expected[None], 3, axis=0))
    assert session.gripper_command == pytest.approx(-1.0)


def test_absolute_camera_pose_is_transformed_to_world_without_clipping() -> None:
    camera_to_world = R.from_euler("z", 90, degrees=True).as_matrix()
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = camera_to_world
    extrinsic[:3, 3] = [1.0, 2.0, 3.0]
    camera_rotation = R.from_euler("x", 30, degrees=True)
    camera_action = [2.0, 0.0, 0.0, *camera_rotation.as_rotvec(), 0.25]
    adapter = EmbodimentAdapter(future_horizon=1, action_delta=False)
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0)

    action = np.asarray(adapter.to_action(
        {"action_plan": [[camera_action]]},
        {"camera.extrinsics": extrinsic.tolist()},
        session,
    )[0])

    np.testing.assert_allclose(action[:3], [1.0, 4.0, 3.0], atol=1e-6)
    expected_rotation = R.from_matrix(camera_to_world @ camera_rotation.as_matrix()).as_rotvec()
    np.testing.assert_allclose(action[3:6], expected_rotation, atol=1e-6)
    assert action[6] == pytest.approx(0.25)


def test_point_only_plan_recovers_absolute_actions_and_joint_gripper() -> None:
    captured = {"points": [], "widths": []}

    class Robot:
        def __init__(self, *, embodiment, with_fingers):
            captured["init"] = (embodiment, with_fingers)

        def project_actor_xyz_to_gripper(
            self, points, *, gripper_width, extrinsic, return_residual,
        ):
            points = np.asarray(points, dtype=np.float64)
            captured["points"].append(points)
            captured["widths"].append(gripper_width)
            pose = np.asarray([
                *points.mean(axis=0),
                0.1,
                0.2,
                0.3,
                gripper_width,
            ])
            return pose, 1e-4

    point_plan = np.zeros((1, 2, len(ACTOR_POINT_INDICES), 3), dtype=np.float32)
    point_plan[0, 0, :len(ACTOR_POINT_INDICES)] = [0.1, 0.2, 1.0]
    point_plan[0, 1, :len(ACTOR_POINT_INDICES)] = [0.4, 0.5, 1.2]
    outputs = {
        "point_plan": point_plan,
        "point_plan_mask": np.ones(point_plan.shape[:-1], dtype=bool),
        "gripper_plan": [[-0.8, 0.8]],
    }
    request = {
        "camera.extrinsics": np.eye(4).tolist(),
        "observation.state": [[0.0] * 6 + [0.02, -0.03]],
    }
    session = SimpleNamespace(benchmark="libero", gripper_command=0.0, frame_index=4)
    adapter = EmbodimentAdapter(
        future_horizon=2,
        action_delta=False,
        flow_mode="point_only",
        robot_cls=Robot,
    )

    actions = np.asarray(adapter.to_action(outputs, request, session))

    np.testing.assert_allclose(actions[:, :3], [[0.1, 0.2, 1.0], [0.4, 0.5, 1.2]])
    np.testing.assert_allclose(actions[:, 3:6], [[0.1, 0.2, 0.3]] * 2)
    np.testing.assert_allclose(actions[:, 6], [-0.8, 0.8])
    assert captured["init"] == ("franka_panda", True)
    assert all(points.shape == (len(ACTOR_POINT_INDICES), 3) for points in captured["points"])
    assert captured["widths"] == pytest.approx([0.05, 0.05])


def test_point_only_server_rejects_delta_and_missing_robot_geometry() -> None:
    class Robot:
        pass

    with pytest.raises(ValueError, match="requires action_delta=False"):
        EmbodimentAdapter(
            future_horizon=2,
            flow_mode="point_only",
            robot_cls=Robot,
        )
    with pytest.raises(ValueError, match="requires robot_cls"):
        EmbodimentAdapter(
            future_horizon=2,
            action_delta=False,
            flow_mode="point_only",
        )


def test_absolute_release_holds_current_tcp_pose() -> None:
    adapter = EmbodimentAdapter(future_horizon=3, action_delta=False)
    session = SimpleNamespace(gripper_command=1.0)
    state = [0.4, -0.2, 1.3, 0.1, 0.2, 1.4, 0.02, -0.02]

    actions = adapter.release_actions(session, 2, {"observation.state": [state]})

    expected = np.asarray(state[:6] + [-1.0], dtype=np.float32)
    np.testing.assert_allclose(actions, np.repeat(expected[None], 2, axis=0), atol=1e-7)
