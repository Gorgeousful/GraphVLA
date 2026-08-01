"""RoPE attention and time-conditioned normalization shared by GraphVLA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Apply one-axis RoPE to ``x`` shaped [B, L, heads, head_dim]."""
    if x.shape[-1] % 2:
        raise ValueError(f"RoPE dimension must be even, got {x.shape[-1]}")
    exponents = (2.0 / x.shape[-1]) * torch.arange(
        x.shape[-1] // 2, device=x.device, dtype=torch.float32,
    )
    radians = positions.to(device=x.device, dtype=torch.float32)[..., None] / (
        max_wavelength ** exponents
    )
    if positions.ndim == 1:
        radians = radians[None, :, None]
    elif positions.ndim == 2:
        radians = radians[:, :, None]
    else:
        raise ValueError(f"Expected positions [L] or [B,L], got {positions.shape}")
    x1, x2 = x.float().chunk(2, dim=-1)
    sin, cos = radians.sin(), radians.cos()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def apply_multi_axis_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    axis_dims: tuple[int, ...],
) -> torch.Tensor:
    """Apply independent RoPE segments for positions shaped [L,A] or [B,L,A]."""
    if positions.shape[-1] != len(axis_dims) or sum(axis_dims) != x.shape[-1]:
        raise ValueError(
            f"RoPE axes {positions.shape}/{axis_dims} do not match head dim {x.shape[-1]}"
        )
    chunks = x.split(axis_dims, dim=-1)
    return torch.cat(
        [apply_rope(chunk, positions[..., axis]) for axis, chunk in enumerate(chunks)],
        dim=-1,
    )


class RotaryAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        axis_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if axis_dims is not None and (sum(axis_dims) != self.head_dim or any(dim % 2 for dim in axis_dims)):
            raise ValueError(f"Invalid multi-axis RoPE split {axis_dims} for head dim {self.head_dim}")
        self.axis_dims = axis_dims
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def _rope(self, value: torch.Tensor, positions: torch.Tensor | None) -> torch.Tensor:
        if positions is None:
            return value
        if self.axis_dims is None:
            return apply_rope(value, positions)
        return apply_multi_axis_rope(value, positions, self.axis_dims)

    def project(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_positions: torch.Tensor | None,
        key_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_length, _ = query.shape
        key_length = key_value.shape[1]
        q = self.q_projection(query).view(batch, query_length, self.num_heads, self.head_dim)
        k = self.k_projection(key_value).view(batch, key_length, self.num_heads, self.head_dim)
        v = self.v_projection(key_value).view(batch, key_length, self.num_heads, self.head_dim)
        q = self._rope(q, query_positions).transpose(1, 2)
        k = self._rope(k, key_positions).transpose(1, 2)
        return q, k, v.transpose(1, 2)

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask.to(device=scores.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            elif mask.ndim != 4:
                raise ValueError(f"Expected attention mask [Q,K], [B,Q,K], or [B,H,Q,K], got {mask.shape}")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(scores.softmax(dim=-1))
        hidden = torch.matmul(weights, value).transpose(1, 2).contiguous()
        hidden = hidden.view(hidden.shape[0], hidden.shape[1], -1)
        return self.output_projection(hidden)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self.project(query, key_value, query_positions, key_positions)
        return self.attend(q, k, v, attention_mask)


class RotaryEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        axis_dims: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = RotaryAttention(hidden_dim, num_heads, dropout, axis_dims)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        token: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(token)
        token = token + self.residual_dropout(
            self.attention(normalized, normalized, positions, positions, attention_mask)
        )
        return token + self.residual_dropout(self.ffn(self.ffn_norm(token)))


class AdaRMSNorm(nn.Module):
    """RMS normalization with zero-initialized scale, shift, and residual gate."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, token: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        variance = token.float().square().mean(dim=-1, keepdim=True)
        normalized = token * torch.rsqrt(variance + 1e-6).to(token.dtype)
        modulation = self.modulation(condition)
        if modulation.ndim == 2:
            modulation = modulation[:, None]
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return normalized * (1.0 + scale) + shift, gate


class RotaryFlowBlock(nn.Module):
    """One modality expert block used in paired Point/Action joint attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        self_axis_dims: tuple[int, ...],
        history_axis_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.self_norm = AdaRMSNorm(hidden_dim)
        self.self_attention = RotaryAttention(hidden_dim, num_heads, dropout, self_axis_dims)
        self.history_norm = AdaRMSNorm(hidden_dim)
        self.history_attention = RotaryAttention(hidden_dim, num_heads, dropout, history_axis_dims)
        self.semantic_norm = AdaRMSNorm(hidden_dim)
        self.semantic_attention = RotaryAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = AdaRMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def prepare_self_attention(
        self,
        token: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized, gate = self.self_norm(token, condition)
        q, k, v = self.self_attention.project(normalized, normalized, positions, positions)
        return q, k, v, gate

    def finish(
        self,
        token: torch.Tensor,
        self_update: torch.Tensor,
        self_gate: torch.Tensor,
        history_memory: torch.Tensor,
        scene_memory: torch.Tensor,
        condition: torch.Tensor,
        query_positions: torch.Tensor,
        history_positions: torch.Tensor,
    ) -> torch.Tensor:
        token = token + self.residual_dropout(self_update) * self_gate
        normalized, gate = self.history_norm(token, condition)
        update = self.history_attention(
            normalized, history_memory, query_positions, history_positions,
        )
        token = token + self.residual_dropout(update) * gate
        normalized, gate = self.semantic_norm(token, condition)
        update = self.semantic_attention(normalized, scene_memory)
        token = token + self.residual_dropout(update) * gate
        normalized, gate = self.ffn_norm(token, condition)
        return token + self.residual_dropout(self.ffn(normalized)) * gate
