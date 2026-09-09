from __future__ import annotations

import sys

import numpy as np
import pytest

from examples.libero.eval.client import (
    EpisodeScheduler,
    InferenceClient,
    _draw_response_points,
    _dummy_action,
    _merge_task_results,
    parse_args,
    _to_libero_action,
)


def test_tasks_cli_uses_space_separated_integers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["client", "--tasks", "1", "6", "4"])

    assert parse_args().tasks == [1, 6, 4]


def test_tasks_cli_rejects_comma_separated_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["client", "--tasks", "1,6,4"])

    with pytest.raises(SystemExit):
        parse_args()


def test_episode_scheduler_prefers_distinct_tasks_then_balances() -> None:
    scheduler = EpisodeScheduler([(1, 10), (2, 20)], episodes_per_task=3)

    jobs = [scheduler.acquire() for _ in range(4)]

    assert [(job.task_id, job.episode_idx) for job in jobs] == [
        (10, 0), (20, 0), (10, 1), (20, 1),
    ]
    for job in jobs:
        scheduler.release(job)

    scheduler = EpisodeScheduler([(1, 10), (2, 20), (3, 30)], episodes_per_task=2)
    first = scheduler.acquire()
    second = scheduler.acquire()
    scheduler.release(first)

    third = scheduler.acquire()

    assert (first.task_id, second.task_id, third.task_id) == (10, 20, 30)


def test_merge_task_results_restores_task_and_episode_order() -> None:
    def partial(task_id: int, episode_id: int, success: bool) -> dict[str, object]:
        episode = {
            "episode_id": episode_id,
            "success": success,
            "server_success": success,
            "progress": float(success),
        }
        return {
            "task_id": task_id,
            "task_desc": f"task {task_id}",
            "total_goals": 1,
            "success_rate": float(success),
            "server_success_rate": float(success),
            "progress_rate": float(success),
            "num_episodes": 1,
            "episodes": [episode],
        }

    merged = _merge_task_results(
        [partial(20, 1, False), partial(10, 0, True), partial(20, 0, True)],
        [10, 20],
    )

    assert [result["task_id"] for result in merged] == [10, 20]
    assert [episode["episode_id"] for episode in merged[1]["episodes"]] == [0, 1]
    assert merged[1]["success_rate"] == 0.5


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
