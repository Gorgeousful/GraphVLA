from examples.libero.config.data_config import (
    LIBERO_DATA_CONFIG,
    LIBERO_FUTURE_HORIZON,
    LIBERO_HISTORY_HORIZON,
    LIBERO_HORIZON,
)
from examples.libero.config.model_config import LIBERO_MODEL_CONFIG


def test_libero_uses_twenty_input_frames_and_ten_future_frames() -> None:
    assert LIBERO_HISTORY_HORIZON == 19
    assert LIBERO_FUTURE_HORIZON == 10
    assert LIBERO_HORIZON["gripper_uvd"] == list(range(-LIBERO_HISTORY_HORIZON, LIBERO_FUTURE_HORIZON + 1))
    assert LIBERO_MODEL_CONFIG.history_horizon == LIBERO_HISTORY_HORIZON
    assert LIBERO_MODEL_CONFIG.future_horizon == LIBERO_FUTURE_HORIZON


def test_libero_training_does_not_load_unused_videos() -> None:
    assert LIBERO_DATA_CONFIG.load_videos is False


def test_libero_data_and_model_share_delta_configuration() -> None:
    model_input_transform = LIBERO_DATA_CONFIG.transforms[-1]
    assert model_input_transform.extra["use_delta"] is LIBERO_MODEL_CONFIG.use_delta


def test_libero_data_and_model_share_ray_scale() -> None:
    model_input_transform = LIBERO_DATA_CONFIG.transforms[-1]
    model_output_transform = LIBERO_DATA_CONFIG.out_transforms[-1]
    assert model_input_transform.extra["ray_scale"] == LIBERO_MODEL_CONFIG.ray_scale
    assert model_output_transform.extra["ray_scale"] == LIBERO_MODEL_CONFIG.ray_scale
