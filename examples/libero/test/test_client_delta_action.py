from __future__ import annotations

import numpy as np
import pytest

from examples.libero.eval.client import InferenceClient, _dummy_action


def test_wait_action_is_zero_delta_with_open_gripper() -> None:
    np.testing.assert_array_equal(
        _dummy_action(),
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    )


def test_client_accepts_finite_seven_dimensional_action_chunk() -> None:
    client = object.__new__(InferenceClient)
    action = [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0]]

    assert client._validated_action_chunk({"action": action}) == action

    with pytest.raises(ValueError, match="must be 7-D"):
        client._validated_action_chunk({"action": [[0.0] * 6]})
