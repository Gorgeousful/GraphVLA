from __future__ import annotations

import pytest
import torch

from src.model.encoder import EntityEncoder


def test_role_token_participates_in_all_register_local_attention(
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

    expected_length = encoder.cls_token_num + 1 + points.shape[3]
    assert local_lengths == [expected_length] * len(encoder.local_blocks)
