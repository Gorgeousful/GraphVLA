from __future__ import annotations

import pytest
import torch

from src.model.encoder import EntityEncoder


def test_role_film_conditions_all_register_local_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = EntityEncoder(
        hidden_dim=32,
        actor_num_points=4,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        condition_dim=8,
        max_history=2,
        cls_token_num=2,
        dropout=0.0,
        global_layer_types=(0, 0),
    ).eval()
    points = torch.randn(2, 2, 3, 4, 3)
    point_mask = torch.ones(2, 2, 3, 4, dtype=torch.bool)
    scene_condition = torch.randn(2, 2, 8)
    local_lengths: list[int] = []

    for block in encoder.local_blocks:
        original_forward = block.forward

        def capture_local(*args, _forward=original_forward, **kwargs):
            local_lengths.append(args[0].shape[1])
            return _forward(*args, **kwargs)

        monkeypatch.setattr(block, "forward", capture_local)

    encoder(points, point_mask, scene_condition)

    expected_length = encoder.cls_token_num + points.shape[3]
    assert local_lengths == [expected_length] * len(encoder.local_blocks)


def test_encoder_film_is_identity_initialized_and_receives_gradients() -> None:
    encoder = EntityEncoder(
        hidden_dim=32,
        actor_num_points=4,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        condition_dim=8,
        max_history=2,
        cls_token_num=2,
        dropout=0.0,
        global_layer_types=(0,),
    )
    for norm in (
        encoder.local_blocks[0].attention_norm,
        encoder.local_blocks[0].ffn_norm,
        encoder.global_blocks[0].attention_norm,
        encoder.global_blocks[0].ffn_norm,
    ):
        assert torch.count_nonzero(norm.modulation[-1].weight) == 0
        assert torch.count_nonzero(norm.modulation[-1].bias) == 0

    outputs = encoder(
        torch.randn(2, 2, 3, 4, 3),
        torch.ones(2, 2, 3, 4, dtype=torch.bool),
        torch.randn(2, 2, 8),
    )
    sum(output.sum() for output in outputs).backward()

    assert encoder.local_blocks[0].attention_norm.modulation[-1].weight.grad is not None
    assert encoder.global_blocks[0].attention_norm.modulation[-1].weight.grad is not None
