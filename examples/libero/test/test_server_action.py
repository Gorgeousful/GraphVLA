from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from script.server import (
    EmbodimentAdapter,
    InputPreprocessor,
    ObservationFrame,
    TopLevelTaskPlanner,
)
ACTOR_POINT_INDICES = (0, 1, 2, 5)


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
    preprocessor.actor_point_indices = ACTOR_POINT_INDICES
    preprocessor.actor_num_points = len(ACTOR_POINT_INDICES)
    preprocessor.point_coordinate_frame = "tcp_absolute"
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
    closedness = np.asarray(model_input["gripper_closedness_history"], dtype=np.float32)
    assert closedness.shape == (1, 2, 1)
    expected_openness = np.clip(np.linalg.norm(gripper[3] - gripper[4]) / 0.08, 0.0, 1.0)
    expected_closedness = 1.0 - 2.0 * expected_openness
    np.testing.assert_allclose(closedness, expected_closedness, atol=1e-6)


def test_online_model_input_centers_entire_history_on_current_tcp() -> None:
    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.num_points = 6
    preprocessor.actor_point_indices = (0, 1, 2, 3, 4, 5)
    preprocessor.actor_num_points = 6
    preprocessor.point_coordinate_frame = "tcp_relative"
    preprocessor.norm_stats = {
        "tcp_relative_xyz": {
            "q01": [-1.0, -1.0, -1.0],
            "q99": [1.0, 1.0, 1.0],
        },
    }
    first_gripper = np.zeros((6, 3), dtype=np.float32)
    current_gripper = np.zeros((6, 3), dtype=np.float32)
    first_gripper[:, 0] = np.arange(6, dtype=np.float32) + 8.0
    current_gripper[:, 0] = np.arange(6, dtype=np.float32) + 10.0
    frames = [
        {
            "tracks": np.zeros((0, 6, 3), dtype=np.float32),
            "metric_depth": np.ones((2, 2), dtype=np.float32),
            "gripper_points_xyz": gripper,
            "intrinsic": np.eye(3, dtype=np.float32),
        }
        for gripper in (first_gripper, current_gripper)
    ]
    subtask = {"nodes": [], "action_type": "lift", "action_degree": None}
    session = SimpleNamespace(
        taskstructure={"subtasks": [subtask]}, subtask_index=0, active_object_indices=[],
    )

    model_input = preprocessor._build_model_input(session, frames, subtask)

    origin = current_gripper[5]
    np.testing.assert_allclose(model_input["tcp_origin"], origin[None], atol=1e-6)
    actor = np.asarray(model_input["entity_points"], dtype=np.float32)[0, :, 0]
    np.testing.assert_allclose(actor[0], first_gripper - origin, atol=5e-6)
    np.testing.assert_allclose(actor[1, 5], np.zeros(3), atol=1e-6)


def test_sparse_history_window_selects_configured_offsets_and_pads_start() -> None:
    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.history_frames = (-5, -2, -1)
    preprocessor.history_horizon = len(preprocessor.history_frames)
    session = SimpleNamespace(feature_history=[])

    for frame_index in range(7):
        preprocessor._append_feature_history(session, {"frame_index": frame_index})

    assert [frame["frame_index"] for frame in preprocessor._feature_window(session)] == [1, 4, 5, 6]

    session.feature_history = [{"frame_index": 0}, {"frame_index": 1}]
    assert [frame["frame_index"] for frame in preprocessor._feature_window(session)] == [0, 0, 0, 1]


def test_object_names_are_deduplicated_and_sampled_masks_expand() -> None:
    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.segmenter = "sam3"
    preprocessor.num_points = 4
    nodes = [
        {"name": "plate", "role": "patient"},
        {"name": "box", "role": "patient"},
        {"name": "plate", "role": "target"},
    ]

    unique_nodes, inverse = preprocessor._unique_nodes_by_name(nodes)
    masks = [np.eye(6, dtype=bool), np.zeros((6, 6), dtype=bool)]
    unique_tracks = preprocessor._tracks_from_masks(masks, (6, 6), len(unique_nodes))
    tracks = unique_tracks[inverse]

    assert [node["name"] for node in unique_nodes] == ["plate", "box"]
    assert inverse == [0, 1, 0]
    np.testing.assert_array_equal(tracks[0], tracks[2])
    np.testing.assert_array_equal(tracks[0, :, 2], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(tracks[1], np.zeros((4, 3), dtype=np.float32))


def test_tracker_update_expands_unique_tracks_to_duplicate_nodes() -> None:
    class Segmenter:
        def predict(self, image, *, anchor_frame):
            assert anchor_frame is False
            return []

    class Tracker:
        def track(self, image, *, anchor_frame):
            assert anchor_frame is False
            return {
                "points": np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32),
                "visibles": np.ones((2, 1), dtype=np.float32),
            }

    preprocessor = object.__new__(InputPreprocessor)
    preprocessor.sam_only = False
    session = SimpleNamespace(
        object_segmenter=Segmenter(),
        point_tracker=Tracker(),
        object_nodes=[{"name": "plate"}, {"name": "box"}, {"name": "plate"}],
        object_to_unique_indices=[0, 1, 0],
        tracked_points=None,
    )
    frame = SimpleNamespace(image=np.zeros((6, 6, 3), dtype=np.uint8))

    preprocessor._update_perception(session, frame)

    assert session.tracked_points.shape == (3, 1, 3)
    np.testing.assert_array_equal(session.tracked_points[0], session.tracked_points[2])
    np.testing.assert_array_equal(session.tracked_points[1, 0], [3.0, 4.0, 1.0])


def test_point_only_plan_recovers_absolute_actions_and_continuous_gripper() -> None:
    captured = {"points": [], "widths": []}

    class Robot:
        def __init__(self, *, embodiment, with_fingers):
            captured["init"] = (embodiment, with_fingers)

        def project_actor_xyz_to_gripper(
            self, points, *, gripper_width, extrinsic, actor_point_indices, return_residual=False,
        ):
            points = np.asarray(points, dtype=np.float64)
            captured["points"].append(points)
            captured["widths"].append(gripper_width)
            captured.setdefault("actor_point_indices", []).append(actor_point_indices)
            pose = np.asarray([
                *points.mean(axis=0),
                0.1,
                0.2,
                0.3,
                gripper_width,
            ])
            return (pose, 1e-4) if return_residual else pose

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
    session = SimpleNamespace(benchmark="libero", frame_index=4)
    adapter = EmbodimentAdapter(
        actor_point_indices=ACTOR_POINT_INDICES,
        future_horizon=2,
        action_delta=False,
        robot_cls=Robot,
    )

    actions = np.asarray(adapter.to_action(outputs, request, session))

    np.testing.assert_allclose(actions[:, :3], [[0.1, 0.2, 1.0], [0.4, 0.5, 1.2]])
    np.testing.assert_allclose(actions[:, 3:6], [[0.1, 0.2, 0.3]] * 2)
    np.testing.assert_allclose(actions[:, 6], [-0.8, 0.8])
    assert captured["init"] == ("franka_panda", True)
    assert all(points.shape == (len(ACTOR_POINT_INDICES), 3) for points in captured["points"])
    assert captured["widths"] == pytest.approx([0.05, 0.05])
    assert captured["actor_point_indices"] == [ACTOR_POINT_INDICES, ACTOR_POINT_INDICES]


def test_point_only_server_rejects_delta_and_missing_robot_geometry() -> None:
    class Robot:
        pass

    with pytest.raises(ValueError, match="requires action_delta=False"):
        EmbodimentAdapter(
            actor_point_indices=ACTOR_POINT_INDICES,
            future_horizon=2,
            action_delta=True,
            robot_cls=Robot,
        )
    with pytest.raises(ValueError, match="requires robot_cls"):
        EmbodimentAdapter(
            actor_point_indices=ACTOR_POINT_INDICES,
            future_horizon=2,
            action_delta=False,
        )


def test_absolute_release_opens_gripper_then_holds_lifted_tcp() -> None:
    adapter = EmbodimentAdapter(
        actor_point_indices=ACTOR_POINT_INDICES,
        future_horizon=3,
        robot_cls=object,
    )
    session = SimpleNamespace()
    state = [0.4, -0.2, 1.3, 0.1, 0.2, 1.4, 0.02, -0.02]

    actions = adapter.release_actions(session, 2, {"observation.state": [state]})

    expected = np.asarray([
        [0.4, -0.2, 1.325, 0.1, 0.2, 1.4, -1.0],
        [0.4, -0.2, 1.350, 0.1, 0.2, 1.4, -1.0],
        [0.4, -0.2, 1.350, 0.1, 0.2, 1.4, -1.0],
        [0.4, -0.2, 1.350, 0.1, 0.2, 1.4, -1.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(actions, expected, atol=1e-7)


def test_delta_release_opens_gripper_then_holds_position() -> None:
    adapter = EmbodimentAdapter(
        actor_point_indices=ACTOR_POINT_INDICES,
        future_horizon=3,
        action_delta=True,
        robot_cls=object,
    )

    actions = adapter.release_actions(SimpleNamespace(), 2)

    expected = np.asarray([
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(actions, expected, atol=1e-7)
