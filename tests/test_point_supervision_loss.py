from __future__ import annotations

import math

import torch

from examples.libero.config.model_config import ModelConfig
from src.dataset.transform import CustomTransform
from src.model.model import PointQueryModel


class FixedPointModel(PointQueryModel):
    def __init__(self, prediction: torch.Tensor) -> None:
        torch.nn.Module.__init__(self)
        self.prediction = torch.nn.Parameter(prediction)
        self.weights = {}
        self.actor_num_points = 3
        self.use_future_point_residual = False
        self.residual_point_dims = ()

    def encode(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.prediction.new_zeros((self.prediction.shape[0], 1, 1))

    def decode(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"point": self.prediction}


def make_actor_batch(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "point_feats": torch.empty(prediction.shape[0], 0),
        "actor_feats": torch.empty(prediction.shape[0], 0),
        "object_id": torch.zeros(prediction.shape[:2], dtype=torch.long),
        "point_id": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "frame_id": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long),
        "target": {
            "point": target,
            "point_mask": torch.ones(prediction.shape[:2], dtype=torch.bool),
        },
    }


def test_point_supervision_uses_l1_for_geometry_and_bce_for_visibility() -> None:
    prediction = torch.zeros(1, 6, 5)
    target = torch.zeros(1, 6, 6)
    target[..., :3] = 1.0
    target[..., 3] = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
    batch = make_actor_batch(prediction, target)
    model = FixedPointModel(prediction)

    _, metrics = model(
        batch,
        weights={
            "point_regression": 1.0,
            "visibility": 1.0,
            "metric_depth": 0.0,
            "gripper_width": 0.0,
        },
    )

    torch.testing.assert_close(metrics["loss_history_actor_point_regression"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["loss_future_actor_point_regression"], torch.tensor(1.0))
    torch.testing.assert_close(
        metrics["loss_history_actor_visibility"],
        torch.tensor(math.log(2.0)),
    )
    torch.testing.assert_close(
        metrics["loss_future_actor_visibility"],
        torch.tensor(math.log(2.0)),
    )


def test_metric_depth_uses_target_depth_mask_without_predicting_it() -> None:
    prediction = torch.zeros(1, 6, 5)
    target = torch.zeros(1, 6, 6)
    target[..., 4] = torch.tensor([[2.0, 100.0, 4.0, 6.0, 100.0, 8.0]])
    target[..., 5] = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
    batch = make_actor_batch(prediction, target)
    model = FixedPointModel(prediction)

    loss, metrics = model(
        batch,
        weights={
            "point_regression": 0.0,
            "visibility": 0.0,
            "metric_depth": 1.0,
            "gripper_width": 0.0,
        },
    )
    loss.backward()

    torch.testing.assert_close(metrics["loss_history_actor_metric_depth"], torch.tensor(3.0))
    torch.testing.assert_close(metrics["loss_future_actor_metric_depth"], torch.tensor(7.0))
    assert model.prediction.grad is not None
    torch.testing.assert_close(
        model.prediction.grad[..., 4][target[..., 5] == 0],
        torch.zeros(2),
    )


def test_gripper_width_uses_left_right_uv_distance_for_each_actor_frame() -> None:
    prediction = torch.zeros(1, 6, 5)
    target = torch.zeros(1, 6, 6)
    target[:, (1, 4), 0] = -0.5
    target[:, (2, 5), 0] = 0.5
    batch = make_actor_batch(prediction, target)
    model = FixedPointModel(prediction)

    _, metrics = model(
        batch,
        weights={
            "point_regression": 0.0,
            "visibility": 0.0,
            "metric_depth": 0.0,
            "gripper_width": 1.0,
        },
    )

    torch.testing.assert_close(metrics["loss_history_actor_gripper_width"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["loss_future_actor_gripper_width"], torch.tensor(1.0))
    assert "loss_history_object_gripper_width" not in metrics
    assert "loss_future_object_gripper_width" not in metrics


def test_model_outputs_five_point_values_and_sigmoids_visibility() -> None:
    config = ModelConfig()
    assert config.point_dim == 6
    assert config.output_dims["point"] == 5

    transform = CustomTransform(
        mode="build_model_output",
        extra={
            "norm_stats": {
                "level": "suite",
                "norm_stats": {
                    "depths.depth_rel": {"q01": [0.0], "q99": [1.0]},
                    "gripper_d": {"q01": [0.0], "q99": [1.0]},
                },
            },
            "use_quantiles": True,
            "quantile_to_neg_one_one": True,
        },
    )
    point = torch.zeros(1, 1, 5)
    data = {
        "outputs": {"point": point},
        "batch": {"object_id": torch.zeros(1, 1, dtype=torch.long)},
    }

    transform(data)

    assert data["outputs"]["point"].shape[-1] == 5
    torch.testing.assert_close(data["outputs"]["point"][..., 3], torch.tensor([[0.5]]))
