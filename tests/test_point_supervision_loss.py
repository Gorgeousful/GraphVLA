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


class FixedFrameModel(PointQueryModel):
    def __init__(self, prediction: torch.Tensor) -> None:
        torch.nn.Module.__init__(self)
        self.prediction = torch.nn.Parameter(prediction)
        self.weights = {}

    def encode(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.prediction.new_zeros((self.prediction.shape[0], 1, 1))

    def decode_frame(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"is_complete": self.prediction}


class FixedObjectModel(PointQueryModel):
    def __init__(self, openness_prediction: torch.Tensor, action_prediction: torch.Tensor) -> None:
        torch.nn.Module.__init__(self)
        self.openness_prediction = torch.nn.Parameter(openness_prediction)
        self.action_prediction = torch.nn.Parameter(action_prediction)
        self.weights = {}

    def encode(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.openness_prediction.new_zeros((self.openness_prediction.shape[0], 1, 1))

    def decode_object(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "gripper_openness": self.openness_prediction,
            "gripper_action": self.action_prediction,
        }


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
        "frame_id": torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long),
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
        },
    )

    torch.testing.assert_close(metrics["loss_future_actor_point_regression"], torch.tensor(1.0))
    torch.testing.assert_close(
        metrics["loss_future_actor_visibility"],
        torch.tensor(math.log(2.0)),
    )
    assert not any("history" in name or "object" in name for name in metrics)


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
        },
    )
    loss.backward()

    torch.testing.assert_close(metrics["loss_future_actor_metric_depth"], torch.tensor(5.0))
    assert model.metric_prediction is not None
    assert model.metric_prediction.grad is not None
    torch.testing.assert_close(
        model.metric_prediction.grad[..., 0][~metric_mask],
        torch.zeros(2),
    )


def test_point_loss_exposes_only_future_actor_metrics() -> None:
    point_prediction = torch.zeros(1, 6, 4)
    metric_prediction = torch.zeros(1, 6, 1)
    batch = {
        "point_feats": torch.empty(1, 0),
        "actor_feats": torch.empty(1, 0),
        "object_id": torch.tensor([[0, 0, 0, 0, 0, 0]]),
        "point_id": torch.tensor([[0, 1, 2, 0, 1, 2]]),
        "frame_id": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "target": {
            "point": torch.zeros(1, 6, 4),
            "point_mask": torch.ones(1, 6, dtype=torch.bool),
            "metric_depth": torch.zeros(1, 6, 1),
            "metric_depth_mask": torch.zeros(1, 6, dtype=torch.bool),
        },
    }
    model = FixedPointModel(point_prediction, metric_prediction)

    _, metrics = model(
        batch,
        weights={
            "point_regression": 0.0,
            "visibility": 0.0,
            "metric_depth": 1.0,
        },
    )

    assert not any("history" in name or "object" in name for name in metrics)


def test_gripper_openness_and_action_use_full_frame_l1() -> None:
    openness_prediction = torch.zeros(1, 2, 1)
    action_prediction = torch.zeros(1, 2, 1)
    batch = {
        "point_feats": torch.empty(1, 0),
        "actor_feats": torch.empty(1, 0),
        "actor_query_frame_id": torch.tensor([[1, 2]]),
        "target": {
            "gripper_openness": torch.tensor([[[3.0], [4.0]]]),
            "gripper_action": torch.tensor([[[-1.0], [1.0]]]),
        },
    }
    model = FixedObjectModel(openness_prediction, action_prediction)

    loss, metrics = model(
        batch,
        weights={
            "future_weight": 2.0,
            "gripper_openness": 2.0,
            "gripper_action": 0.5,
        },
    )
    loss.backward()

    torch.testing.assert_close(metrics["loss_future_gripper_openness"], torch.tensor(14.0))
    torch.testing.assert_close(metrics["loss_gripper_openness"], torch.tensor(14.0))
    torch.testing.assert_close(metrics["loss_future_gripper_action"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["loss_gripper_action"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["loss_total"], torch.tensor(15.0))
    assert not any("history" in name for name in metrics)
    assert torch.count_nonzero(model.openness_prediction.grad) == 2
    assert torch.count_nonzero(model.action_prediction.grad) == 2


def test_completion_classifies_current_observation_only() -> None:
    prediction = torch.zeros(1, 1, 1)
    model = FixedFrameModel(prediction)
    batch = {
        "point_feats": torch.empty(1, 0),
        "actor_feats": torch.empty(1, 0),
        "frame_query_frame_id": torch.tensor([[0]]),
        "target": {"is_complete": torch.tensor([[1.0]])},
    }

    loss, metrics = model(
        batch,
        weights={"is_complete": 2.0},
    )
    loss.backward()

    expected = torch.tensor(2.0 * math.log(2.0))
    torch.testing.assert_close(metrics["loss_current_is_complete"], expected)
    assert "loss_history_is_complete" not in metrics
    assert "loss_future_is_complete" not in metrics
    torch.testing.assert_close(metrics["loss_is_complete"], expected)
    torch.testing.assert_close(loss, expected)
    assert torch.count_nonzero(model.prediction.grad) == 1


def test_model_outputs_shared_points_and_separate_metric_depth() -> None:
    config = ModelConfig()
    assert config.point_dim == 8
    assert config.output_dims["point"] == 4
    assert config.output_dims["metric_depth"] == 1
    assert config.output_dims["gripper_openness"] == 1
    assert config.output_dims["gripper_action"] == 1
    assert "history_weight" not in config.weights
    assert "history_actor" not in config.weights
    assert "history_object" not in config.weights

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
