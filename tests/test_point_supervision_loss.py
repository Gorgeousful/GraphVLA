from __future__ import annotations

import math

import torch

from examples.libero.config.model_config import ModelConfig
from src.dataset.transform import CustomTransform
from src.model.model import PointQueryModel


class FixedPointModel(PointQueryModel):
    def __init__(
        self,
        point_prediction: torch.Tensor,
        metric_prediction: torch.Tensor | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.point_prediction = torch.nn.Parameter(point_prediction)
        if metric_prediction is not None:
            self.metric_prediction = torch.nn.Parameter(metric_prediction)
        else:
            self.register_parameter("metric_prediction", None)
        self.weights = {}
        self.actor_num_points = 3
        self.use_future_point_residual = False

    def encode(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.point_prediction.new_zeros((self.point_prediction.shape[0], 1, 1))

    def decode(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {"point": self.point_prediction}
        if self.metric_prediction is not None:
            outputs["metric_depth"] = self.metric_prediction
        return outputs


def make_actor_batch(
    point_prediction: torch.Tensor,
    point_target: torch.Tensor,
    *,
    metric_target: torch.Tensor | None = None,
    metric_mask: torch.Tensor | None = None,
) -> dict:
    target = {
        "point": point_target,
        "point_mask": torch.ones(point_prediction.shape[:2], dtype=torch.bool),
    }
    if metric_target is not None:
        target["metric_depth"] = metric_target
    if metric_mask is not None:
        target["metric_depth_mask"] = metric_mask
    return {
        "point_feats": torch.empty(point_prediction.shape[0], 0),
        "actor_feats": torch.empty(point_prediction.shape[0], 0),
        "object_id": torch.zeros(point_prediction.shape[:2], dtype=torch.long),
        "point_id": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "frame_id": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long),
        "target": target,
    }


def test_point_supervision_uses_l1_for_geometry_and_bce_for_visibility() -> None:
    prediction = torch.zeros(1, 6, 4)
    target = torch.zeros(1, 6, 4)
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


def test_metric_depth_uses_separate_target_and_mask() -> None:
    point_prediction = torch.zeros(1, 6, 4)
    metric_prediction = torch.zeros(1, 6, 1)
    point_target = torch.zeros(1, 6, 4)
    metric_target = torch.tensor([[[2.0], [100.0], [4.0], [6.0], [100.0], [8.0]]])
    metric_mask = torch.tensor([[True, False, True, True, False, True]])
    batch = make_actor_batch(
        point_prediction,
        point_target,
        metric_target=metric_target,
        metric_mask=metric_mask,
    )
    model = FixedPointModel(point_prediction, metric_prediction)

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
    assert model.metric_prediction is not None
    assert model.metric_prediction.grad is not None
    torch.testing.assert_close(
        model.metric_prediction.grad[..., 0][~metric_mask],
        torch.zeros(2),
    )


def test_metric_depth_mask_applies_to_object_queries() -> None:
    point_prediction = torch.zeros(1, 9, 4)
    metric_prediction = torch.zeros(1, 9, 1)
    metric_target = torch.tensor(
        [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [2.0], [100.0], [4.0]]]
    )
    metric_mask = torch.tensor(
        [[False, False, False, False, False, False, True, False, True]]
    )
    batch = {
        "point_feats": torch.empty(1, 0),
        "actor_feats": torch.empty(1, 0),
        "object_id": torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1]]),
        "point_id": torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2]]),
        "frame_id": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1]]),
        "target": {
            "point": torch.zeros(1, 9, 4),
            "point_mask": torch.ones(1, 9, dtype=torch.bool),
            "metric_depth": metric_target,
            "metric_depth_mask": metric_mask,
        },
    }
    model = FixedPointModel(point_prediction, metric_prediction)

    _, metrics = model(
        batch,
        weights={
            "point_regression": 0.0,
            "visibility": 0.0,
            "metric_depth": 1.0,
            "gripper_width": 0.0,
            "future_object": 0.0,
        },
    )

    torch.testing.assert_close(metrics["loss_future_object_metric_depth"], torch.tensor(3.0))


def test_gripper_width_uses_left_right_uv_distance_for_each_actor_frame() -> None:
    prediction = torch.zeros(1, 6, 4)
    target = torch.zeros(1, 6, 4)
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


def test_model_outputs_shared_points_and_separate_metric_depth() -> None:
    config = ModelConfig()
    assert config.point_dim == 6
    assert config.output_dims["point"] == 4
    assert config.output_dims["metric_depth"] == 1

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
    point = torch.zeros(1, 1, 4)
    metric_depth = torch.zeros(1, 1, 1)
    data = {
        "outputs": {"point": point, "metric_depth": metric_depth},
        "batch": {"object_id": torch.zeros(1, 1, dtype=torch.long)},
    }

    transform(data)

    assert data["outputs"]["point"].shape[-1] == 4
    torch.testing.assert_close(data["outputs"]["point"][..., 3], torch.tensor([[0.5]]))
    torch.testing.assert_close(data["outputs"]["metric_depth"], torch.tensor([[[0.5]]]))
