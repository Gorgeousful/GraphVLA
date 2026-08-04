from __future__ import annotations

import pytest
import torch

from src.model.encoder import EntityEncoder


def _make_encoder(
    global_layer_types: tuple[int, ...] | None,
    *,
    num_layers: int = 2,
) -> EntityEncoder:
    return EntityEncoder(
        hidden_dim=32,
        actor_num_points=4,
        num_layers=num_layers,
        num_heads=4,
        mlp_ratio=2.0,
        condition_dim=8,
        max_history=2,
        cls_token_num=2,
        dropout=0.0,
        global_layer_types=global_layer_types,
    )


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.randn(2, 2, 3, 4, 3)
    point_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    point_mask[0, :, 2, 2:] = False
    scene_condition = torch.randn(2, 2, 8)
    return points, point_mask, scene_condition


def test_all_register_layers_match_legacy_default() -> None:
    torch.manual_seed(0)
    legacy = _make_encoder(None).eval()
    explicit = _make_encoder((0, 0)).eval()
    explicit.load_state_dict(legacy.state_dict())
    points, point_mask, scene_condition = _make_inputs()

    legacy_output = legacy(points, point_mask, scene_condition)
    explicit_output = explicit(points, point_mask, scene_condition)

    for actual, expected in zip(explicit_output, legacy_output, strict=True):
        torch.testing.assert_close(actual, expected)


def test_dense_global_ignores_masked_point_values_and_backpropagates() -> None:
    torch.manual_seed(0)
    encoder = _make_encoder((0, 1)).eval()
    points, point_mask, scene_condition = _make_inputs()
    changed = points.clone()
    changed[~point_mask] = 1_000.0

    expected = encoder(points, point_mask, scene_condition)
    actual = encoder(changed, point_mask, scene_condition)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-5, rtol=1e-5)

    encoder.train()
    loss = sum(output.square().mean() for output in encoder(
        points, point_mask, scene_condition,
    ))
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


@pytest.mark.parametrize("global_layer_types", [(0,), (0, 2)])
def test_global_layer_types_are_validated(global_layer_types: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="global_layer_types"):
        _make_encoder(global_layer_types)
