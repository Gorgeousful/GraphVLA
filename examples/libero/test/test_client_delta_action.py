from __future__ import annotations

import numpy as np
import pytest

from examples.libero.eval.client import (
    InferenceClient,
    _draw_response_points,
    _dummy_action,
    _future_score,
    _to_libero_action,
)


def test_wait_action_is_zero_delta_with_open_gripper() -> None:
    np.testing.assert_array_equal(
        _dummy_action(),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    )


def test_wait_action_holds_current_absolute_pose() -> None:
    observation = {
        "robot0_eef_pos": np.asarray([0.4, -0.1, 0.2]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
    }

    np.testing.assert_allclose(
        _dummy_action(observation, action_delta=False),
        np.asarray([0.4, -0.1, 0.2, 0.0, 0.0, np.pi / 2.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_absolute_hand_action_is_adapted_to_libero_site_frame() -> None:
    action = np.asarray([0.4, -0.1, 0.2, 0.0, 0.0, np.pi / 2.0, -1.0])

    np.testing.assert_allclose(
        _to_libero_action(action, action_delta=False),
        np.asarray([0.4, -0.1, 0.2, 0.0, 0.0, 0.0, -1.0]),
        atol=1e-6,
    )


def test_delta_action_passes_through_libero_adapter() -> None:
    action = np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0])

    np.testing.assert_allclose(_to_libero_action(action, action_delta=True), action)


def test_client_accepts_finite_seven_dimensional_action_chunk() -> None:
    client = object.__new__(InferenceClient)
    action = [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0]]

    assert client._validated_action_chunk({"action": action}) == action

    with pytest.raises(ValueError, match="must be 7-D"):
        client._validated_action_chunk({"action": [[0.0] * 6]})


def test_future_score_selects_action_aligned_contact_frame() -> None:
    response = {"is_contact": [[0.1, 0.4, 0.8]]}

    assert _future_score(response, "is_contact", 1) == pytest.approx(0.1)
    assert _future_score(response, "is_contact", 3) == pytest.approx(0.8)
    assert _future_score(response, "is_contact", 4) is None


def test_progress_visualization_uses_server_threshold() -> None:
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    response = {"subtask_progress": [[0.87]]}

    green = _draw_response_points(
        image, response, None, mode="prediction", progress_threshold=0.85,
    )
    white = _draw_response_points(
        image, response, None, mode="prediction", progress_threshold=0.9,
    )

    green_pixels = (green[..., 1] > green[..., 0]) & (green[..., 1] > green[..., 2])
    white_green_pixels = (white[..., 1] > white[..., 0]) & (white[..., 1] > white[..., 2])
    assert np.any(green_pixels)
    assert not np.any(white_green_pixels)


def test_prediction_visualization_projects_only_valid_point_plan() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    response = {
        "point_plan": [[[
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 1.0],
            [0.0, 0.2, 1.0],
        ]]],
        "point_plan_mask": [[[True, False, True]]],
        "action_plan": [[[1.0] * 7]],
    }
    intrinsic = np.asarray([
        [50.0, 0.0, 50.0],
        [0.0, 50.0, 50.0],
        [0.0, 0.0, 1.0],
    ])

    rendered = _draw_response_points(
        image, response, 1, mode="prediction", progress_threshold=0.85, intrinsic=intrinsic,
    )

    np.testing.assert_array_equal(rendered[50, 50], np.asarray([255, 80, 40]))
    np.testing.assert_array_equal(rendered[50, 60], np.zeros(3, dtype=np.uint8))
    assert rendered[60, 50].any()


def test_tracking_visualization_distinguishes_inactive_and_invisible_points() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    response = {
        "tracking_point": [[10.0, 10.0, 1.0], [20.0, 10.0, 0.0], [30.0, 10.0, 1.0]],
        "tracking_object_id": [0, 0, 0],
        "tracking_point_active": [True, True, False],
    }

    rendered = _draw_response_points(
        image, response, 1, mode="tracking", progress_threshold=0.85,
    )

    np.testing.assert_array_equal(rendered[10, 10], np.asarray([255, 80, 40]))
    np.testing.assert_array_equal(rendered[10, 20], np.asarray([255, 194, 180]))
    np.testing.assert_array_equal(rendered[10, 30], np.asarray([145, 145, 145]))
