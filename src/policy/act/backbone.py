"""ResNet18 image backbone adapted from the official ACT repository."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from src.policy.checkpointing import checkpoint_module
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_features: int, temperature: int = 10_000) -> None:
        super().__init__()
        self.num_features = num_features
        self.temperature = temperature

    def forward(self, value: Tensor) -> Tensor:
        mask = torch.ones_like(value[:1, :1])
        y_embed = mask.cumsum(2, dtype=torch.float32)
        x_embed = mask.cumsum(3, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * (2 * math.pi)
        x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * (2 * math.pi)
        frequencies = torch.arange(
            self.num_features, dtype=torch.float32, device=value.device,
        )
        frequencies = self.temperature ** (
            2 * torch.div(frequencies, 2, rounding_mode="floor") / self.num_features
        )
        pos_x = x_embed[..., None] / frequencies
        pos_y = y_embed[..., None] / frequencies
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1,
        ).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1,
        ).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1).squeeze(1).permute(0, 3, 1, 2)


class ACTBackbone(nn.Module):
    """Official ACT ResNet18 with frozen batch normalization, shared by cameras."""

    num_channels = 512

    def __init__(self, hidden_dim: int, pretrained: bool = True,
                 weights_path: str | Path | None = None) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained and weights_path is None else None
        network = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        if pretrained and weights_path is not None:
            network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        # Checkpoint boundaries retain activations consumed by the next layer.
        for module in network.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features = image
        for layer in self.features:
            features = checkpoint_module(layer, features)
        return features, self.position_embedding(features).to(features.dtype)
