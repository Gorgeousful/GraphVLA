from __future__ import annotations

import torch

from src.model.encoder import SetEncoderViT


def test_set_encoder_is_equivariant_to_point_permutation() -> None:
    torch.manual_seed(0)
    encoder = SetEncoderViT(
        point_dim=3,
        num_points=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        condition_dim=None,
    ).eval()
    points = torch.randn(1, 2, 1, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    output = encoder(points)
    permuted_output = encoder(points[..., permutation, :])

    torch.testing.assert_close(
        permuted_output,
        output[..., permutation, :],
        rtol=1e-5,
        atol=1e-6,
    )
