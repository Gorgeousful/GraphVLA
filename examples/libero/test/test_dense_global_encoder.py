from __future__ import annotations

import pytest
import torch

from src.policy.graphpoint.encoder import EntityEncoder


def _make_encoder(
    global_layer_types: tuple[int, ...] | None,
    *,
    num_layers: int = 2,
    node_attention_mode: str = "full",
    encoder_output_type: str = "current",
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
        node_attention_mode=node_attention_mode,
        encoder_output_type=encoder_output_type,
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


def test_node_attention_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="node_attention_mode"):
        _make_encoder((0, 0), node_attention_mode="invalid")


def test_encoder_output_type_controls_history_cls_memory() -> None:
    torch.manual_seed(0)
    current = _make_encoder((0, 0), encoder_output_type="current").eval()
    all_history = _make_encoder((0, 0), encoder_output_type="all").eval()
    all_history.load_state_dict(current.state_dict())
    points, point_mask, scene_condition = _make_inputs()

    current_memory, current_relation, current_semantics = current(
        points, point_mask, scene_condition,
    )
    all_memory, all_relation, all_semantics = all_history(
        points, point_mask, scene_condition,
    )

    assert current_memory.shape == (2, 3 * 2, 32)
    assert all_memory.shape == (2, 2 * 3 * 2, 32)
    torch.testing.assert_close(current_memory, all_memory[:, -6:])
    torch.testing.assert_close(current_relation, all_relation)
    assert current_semantics.shape == (2, 2, 32)
    torch.testing.assert_close(current_semantics, all_semantics)


def test_encoder_output_type_is_validated() -> None:
    with pytest.raises(ValueError, match="encoder_output_type"):
        _make_encoder((0, 0), encoder_output_type="invalid")


@pytest.mark.parametrize("global_layer_type", [0, 1])
def test_role_chain_blocks_direct_actor_target_node_attention(
    global_layer_type: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _make_encoder(
        (global_layer_type,), num_layers=1, node_attention_mode="role_chain"
    ).eval()
    captured: list[torch.Tensor] = []
    attention = encoder.global_blocks[0].attention
    original_forward = attention.forward

    def capture_attention(*args, **kwargs):
        captured.append(kwargs["attention_mask"].detach().clone())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(attention, "forward", capture_attention)
    points, point_mask, scene_condition = _make_inputs()
    encoder(points, point_mask, scene_condition)

    mask = captured[0]
    tokens_per_entity = encoder.cls_token_num + (points.shape[3] if global_layer_type else 0)
    actor_cls = 0
    patient_cls = tokens_per_entity
    target_cls = 2 * tokens_per_entity
    target_point = target_cls + encoder.cls_token_num
    scene_token = 2 * 3 * tokens_per_entity
    assert mask[actor_cls, patient_cls]
    assert mask[patient_cls, target_cls]
    assert not mask[actor_cls, target_cls]
    assert not mask[target_cls, actor_cls]
    if global_layer_type:
        assert not mask[actor_cls, target_point]
        actor_point = encoder.cls_token_num
        assert not mask[actor_point, target_cls]
        assert not mask[target_point, actor_point]
    assert mask[actor_cls, scene_token]
