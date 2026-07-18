from __future__ import annotations

from inspect import signature

import torch

from examples.libero.config.model_config import ModelConfig
from src.model.model import PointQueryModel


def test_future_point_residual_is_disabled_by_default() -> None:
    default = signature(PointQueryModel).parameters["use_future_point_residual"].default
    assert default is False
    assert ModelConfig().use_future_point_residual is False
    assert "residual_point_dims" not in signature(PointQueryModel).parameters
    assert not hasattr(ModelConfig(), "residual_point_dims")

    model = PointQueryModel.__new__(PointQueryModel)
    torch.nn.Module.__init__(model)
    model.use_future_point_residual = False
    raw_point = torch.randn(1, 2, 6)

    output = model._apply_future_point_residual(
        raw_point,
        point_feats=torch.empty(0),
        actor_feats=torch.empty(0),
        object_id=torch.empty(0, dtype=torch.long),
        point_id=torch.empty(0, dtype=torch.long),
        frame_id=torch.empty(0, dtype=torch.long),
    )

    assert output is raw_point


def test_future_point_residual_is_applied_when_enabled() -> None:
    model = PointQueryModel.__new__(PointQueryModel)
    torch.nn.Module.__init__(model)
    model.use_future_point_residual = True
    raw_point = torch.ones(1, 2, 4)
    point_feats = torch.zeros(1, 1, 1, 1, 6)
    actor_feats = torch.tensor([[[[[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]]]])

    output = model._apply_future_point_residual(
        raw_point,
        point_feats=point_feats,
        actor_feats=actor_feats,
        object_id=torch.zeros(1, 2, dtype=torch.long),
        point_id=torch.zeros(1, 2, dtype=torch.long),
        frame_id=torch.tensor([[0, 1]]),
    )

    expected = raw_point.clone()
    expected[:, 1, (0, 1, 2)] += actor_feats[0, -1, 0, 0, (0, 1, 2)]
    torch.testing.assert_close(output, expected)


def test_future_metric_depth_residual_uses_anchor_d_mask_for_every_node() -> None:
    model = PointQueryModel.__new__(PointQueryModel)
    torch.nn.Module.__init__(model)
    model.use_future_point_residual = True
    raw_metric_depth = torch.ones(1, 4, 1)
    point_feats = torch.tensor(
        [[[[[0.0, 0.0, 0.0, 1.0, 20.0, 1.0], [0.0, 0.0, 0.0, 1.0, 30.0, 0.0]]]]]
    )
    actor_feats = torch.tensor([[[[[0.0, 0.0, 0.0, 1.0, 10.0, 1.0]]]]])

    output = model._apply_future_metric_depth_residual(
        raw_metric_depth,
        point_feats=point_feats,
        actor_feats=actor_feats,
        object_id=torch.tensor([[0, 0, 1, 1]]),
        point_id=torch.tensor([[0, 0, 0, 1]]),
        frame_id=torch.tensor([[0, 1, 1, 1]]),
    )

    expected = torch.tensor([[[1.0], [11.0], [21.0], [1.0]]])
    torch.testing.assert_close(output, expected)
