"""ResNet image backbone and sine position embedding for ACT."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_messages,
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_messages,
        )

    def forward(self, value: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        scale = weight * (self.running_var.reshape(1, -1, 1, 1) + 1e-5).rsqrt()
        offset = bias - self.running_mean.reshape(1, -1, 1, 1) * scale
        return value * scale + offset


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
    num_channels = 512

    def __init__(self, hidden_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(network, return_layers={"layer4": "features"})
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features = self.body(image)["features"]
        return features, self.position_embedding(features).to(features.dtype)
