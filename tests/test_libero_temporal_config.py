from examples.libero.config.data_config import (
    LIBERO_DATA_CONFIG,
    LIBERO_FUTURE_HORIZON,
    LIBERO_HISTORY_HORIZON,
    LIBERO_HORIZON,
)
from examples.libero.config.model_config import LIBERO_MODEL_CONFIG


def test_libero_uses_ten_input_frames_and_ten_future_frames() -> None:
    assert LIBERO_HISTORY_HORIZON == 9
    assert LIBERO_FUTURE_HORIZON == 10
    expected = list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON + 1))
    assert LIBERO_HORIZON["node_points_xyz"] == expected
    assert LIBERO_HORIZON["valid_node_mask"] == expected
    assert LIBERO_HORIZON["subtask_node_mask"] == expected
    assert LIBERO_HORIZON["gripper_points_xyz"] == expected
    assert LIBERO_HORIZON["state"] == expected
    assert LIBERO_HORIZON["actions"] == expected
    assert LIBERO_MODEL_CONFIG.history_horizon == LIBERO_HISTORY_HORIZON
    assert LIBERO_MODEL_CONFIG.future_horizon == LIBERO_FUTURE_HORIZON


def test_libero_training_uses_only_derived_features() -> None:
    assert LIBERO_DATA_CONFIG.load_videos is False
    assert "relative_plan" not in repr(LIBERO_DATA_CONFIG.transforms)
    assert "ray_scale" not in repr(LIBERO_DATA_CONFIG.transforms)


def test_loss_weights_are_configurable_with_expected_defaults() -> None:
    assert LIBERO_MODEL_CONFIG.weights == {"loss_flow": 1.0, "loss_complete": 0.5}
