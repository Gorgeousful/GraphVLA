"""Conditional one-dimensional U-Net blocks used by Diffusion Policy."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import Tensor, nn
from src.policy.checkpointing import checkpoint_module


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.dim // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale
        )
        embedding = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        if self.dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        *,
        kernel_size: int,
        groups: int,
        predict_scale: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            (
                Conv1dBlock(in_channels, out_channels, kernel_size, groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, groups),
            )
        )
        self.out_channels = out_channels
        self.predict_scale = predict_scale
        self.condition = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dim, out_channels * (2 if predict_scale else 1)),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        hidden = self.blocks[0](value)
        modulation = self.condition(condition)
        if self.predict_scale:
            scale, bias = modulation.reshape(-1, 2, self.out_channels, 1).unbind(1)
            hidden = scale * hidden + bias
        else:
            hidden = hidden + modulation[..., None]
        return self.blocks[1](hidden) + self.residual(value)


class ConditionalUnet1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_condition_dim: int,
        *,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        groups: int = 8,
        predict_scale: bool = True,
    ) -> None:
        super().__init__()
        condition_dim = diffusion_step_embed_dim + global_condition_dim
        self.timestep_encoder = nn.Sequential(
            SinusoidalPositionEmbedding(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        dimensions = (input_dim, *down_dims)
        pairs = tuple(pairwise(dimensions))

        def residual(in_dim: int, out_dim: int) -> ConditionalResidualBlock1d:
            return ConditionalResidualBlock1d(
                in_dim,
                out_dim,
                condition_dim,
                kernel_size=kernel_size,
                groups=groups,
                predict_scale=predict_scale,
            )

        self.down_modules = nn.ModuleList()
        for index, (in_dim, out_dim) in enumerate(pairs):
            downsample = (
                nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1)
                if index < len(pairs) - 1
                else nn.Identity()
            )
            self.down_modules.append(nn.ModuleList((residual(in_dim, out_dim), residual(out_dim, out_dim), downsample)))

        final_dim = dimensions[-1]
        self.middle_modules = nn.ModuleList((residual(final_dim, final_dim), residual(final_dim, final_dim)))
        self.up_modules = nn.ModuleList()
        for in_dim, out_dim in reversed(pairs[1:]):
            upsample = nn.ConvTranspose1d(in_dim, in_dim, 4, stride=2, padding=1)
            self.up_modules.append(
                nn.ModuleList((residual(out_dim * 2, in_dim), residual(in_dim, in_dim), upsample))
            )
        self.output = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample: Tensor, timestep: Tensor | int, global_condition: Tensor) -> Tensor:
        value = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=value.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(value.device)
        timestep = timestep.expand(value.shape[0])
        condition = torch.cat((self.timestep_encoder(timestep), global_condition), dim=-1)

        skips = []
        for first, second, downsample in self.down_modules:
            value = checkpoint_module(second, checkpoint_module(first, value, condition), condition)
            skips.append(value)
            value = downsample(value)
        for middle in self.middle_modules:
            value = checkpoint_module(middle, value, condition)
        for first, second, upsample in self.up_modules:
            value = torch.cat((value, skips.pop()), dim=1)
            value = upsample(checkpoint_module(second, checkpoint_module(first, value, condition), condition))
        return self.output(value).transpose(1, 2)
