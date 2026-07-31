from __future__ import annotations

import numpy as np
import pytest

from examples.libero.eval.client import InferenceClient, _draw_response_points, _hold_action


def test_wait_action_holds_absolute_pose_with_open_gripper() -> None:
    obs = {
        "robot0_eef_pos": np.asarray([0.1, -0.2, 0.3]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
    }

    action = _hold_action(obs)

    np.testing.assert_allclose(action[:3], obs["robot0_eef_pos"], atol=1e-7)
    assert np.isfinite(action[3:6]).all()
    assert action[6] == pytest.approx(-1.0)


def test_client_accepts_finite_seven_dimensional_action_chunk() -> None:
    client = object.__new__(InferenceClient)
    action = [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0]]

    assert client._validated_action_chunk({"action": action}) == action

    with pytest.raises(ValueError, match="must be 7-D"):
        client._validated_action_chunk({"action": [[0.0] * 6]})


def test_prediction_visualization_projects_sixteen_dimensional_actor_points() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    intrinsic = np.asarray([[20.0, 0.0, 32.0], [0.0, 20.0, 32.0], [0.0, 0.0, 1.0]])
    trajectory = [
        0.0, 0.0, 1.0,
        0.6, 0.0, 1.0,
        -0.6, 0.0, 1.0,
        0.0, 0.6, 1.0,
        0.0, -0.6, 1.0,
        -1.0,
    ]

    drawn = _draw_response_points(
        image,
        {"action_plan": [[trajectory]]},
        1,
        mode="prediction",
        intrinsic=intrinsic,
    )

    assert tuple(drawn[32, 32]) == (255, 80, 40)
    assert tuple(drawn[32, 44]) == (60, 60, 255)
    assert tuple(drawn[32, 20]) == (255, 144, 30)
    assert tuple(drawn[44, 32]) == (50, 205, 50)
    assert tuple(drawn[20, 32]) == (0, 215, 255)
