from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from script.server import EmbodimentAdapter


@pytest.mark.parametrize(
    ("predicted_openness", "predicted_action"),
    [
        (0.0, 1.0),
        (1.0, -1.0),
        (0.5, 0.25),
    ],
)
def test_to_action_uses_predicted_gripper_head(
    predicted_openness: float,
    predicted_action: float,
) -> None:
    class Robot:
        def project_uvd_to_gripper(self, *_: object, gripper_width: float, **__: object) -> np.ndarray:
            return np.asarray([0.0] * 6 + [gripper_width], dtype=np.float64)

    adapter = EmbodimentAdapter(future_horizon=1, robot_cls=lambda **_: Robot())
    outputs = {
        "point": [[[10.0, 10.0, 0.5, 1.0], [11.0, 10.0, 0.5, 1.0], [12.0, 10.0, 0.5, 1.0]]],
        "metric_depth": [[[0.5], [0.5], [0.5]]],
        "gripper_openness": [[[predicted_openness]]],
        "gripper_action": [[[predicted_action]]],
    }
    model_input = {
        "object_id": [[0, 0, 0]],
        "point_id": [[0, 1, 2]],
        "frame_id": [[1, 1, 1]],
        "actor_query_frame_id": [[1]],
    }
    request = {
        "camera.intrinsics": np.eye(3).tolist(),
        "camera.extrinsics": np.eye(4).tolist(),
    }

    actions, widths = adapter.to_action(
        outputs,
        model_input,
        request,
        SimpleNamespace(benchmark="libero"),
    )

    assert actions[0][6] == pytest.approx(predicted_action)
    assert widths[0] == pytest.approx(predicted_openness * 0.08)
