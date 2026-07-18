from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from script.server import EmbodimentAdapter


@pytest.mark.parametrize(
    ("opening_width", "expected_action"),
    [
        (-0.01, 1.0),
        (0.00, 1.0),
        (0.02, 1.0),
        (0.039, 1.0),
        (0.04, -1.0),
        (0.06, -1.0),
        (0.08, -1.0),
        (0.10, -1.0),
    ],
)
def test_to_action_binarizes_gripper_width(opening_width: float, expected_action: float) -> None:
    class Robot:
        def project_uvd_to_gripper(self, *_: object, **__: object) -> np.ndarray:
            return np.asarray([0.0] * 6 + [opening_width], dtype=np.float64)

    adapter = EmbodimentAdapter(future_horizon=1, robot_cls=lambda **_: Robot())
    outputs = {
        "point": [[
            [10.0, 10.0, 0.5, 1.0],
            [11.0, 10.0, 0.5, 1.0],
            [12.0, 10.0, 0.5, 1.0],
        ]],
        "metric_depth": [[[0.5], [0.5], [0.5]]],
    }
    model_input = {
        "object_id": [[0, 0, 0]],
        "point_id": [[0, 1, 2]],
        "frame_id": [[1, 1, 1]],
    }
    request = {
        "camera.intrinsics": np.eye(3).tolist(),
        "camera.extrinsics": np.eye(4).tolist(),
    }

    actions, _ = adapter.to_action(
        outputs,
        model_input,
        request,
        SimpleNamespace(benchmark="libero"),
    )

    assert actions[0][6] == pytest.approx(expected_action)
