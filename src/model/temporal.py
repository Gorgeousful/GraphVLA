"""RoPE attention and time-conditioned normalization shared by GraphVLA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to ``x`` shaped [B, L, heads, head_dim]."""
    if x.shape[-1] % 2:
        raise ValueError(f"RoPE head dimension must be even, got {x.shape[-1]}")
    frequency_exponents = (2.0 / x.shape[-1]) * torch.arange(
        x.shape[-1] // 2, device=x.device, dtype=torch.float32,
    )
    timescale = max_wavelength ** frequency_exponents
    radians = positions.to(device=x.device, dtype=torch.float32)[..., None] / timescale
    if positions.ndim == 1:
        radians = radians[None, :, None, :]
    elif positions.ndim == 2:
        radians = radians[:, :, None, :]
    else:
        raise ValueError(f"Expected positions [L] or [B,L], got {positions.shape}")
    sin, cos = radians.sin(), radians.cos()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class RotaryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        if (hidden_dim // num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        use_sdpa: bool = False,
    ) -> torch.Tensor:
        batch, query_length, hidden_dim = query.shape
        key_length = key_value.shape[1]
        q = self.q_projection(query).view(batch, query_length, self.num_heads, self.head_dim)
        k = self.k_projection(key_value).view(batch, key_length, self.num_heads, self.head_dim)
        v = self.v_projection(key_value).view(batch, key_length, self.num_heads, self.head_dim)
        q = apply_rope(q, query_positions)
        k = apply_rope(k, key_positions)
        if use_sdpa:
            if attention_mask is not None:
                raise ValueError("SDPA path does not support attention_mask and key_mask together")
            sdpa_mask = None
            if key_mask is not None:
                if key_mask.shape != (batch, key_length):
                    raise ValueError(
                        f"Expected key mask [B,K]=[{batch},{key_length}], got {key_mask.shape}"
                    )
                sdpa_mask = key_mask.to(device=q.device, dtype=torch.bool)[:, None, None, :]
            output = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=sdpa_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
            )
            output = output.transpose(1, 2).reshape(batch, query_length, hidden_dim)
            return self.output_projection(output)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask.to(device=scores.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            else:
                raise ValueError(f"Expected attention mask [Q,K] or [B,Q,K], got {mask.shape}")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        if key_mask is not None:
            if key_mask.shape != (batch, key_length):
                raise ValueError(
                    f"Expected key mask [B,K]=[{batch},{key_length}], got {key_mask.shape}"
                )
            scores = scores.masked_fill(
                ~key_mask.to(device=scores.device, dtype=torch.bool)[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        weights = self.attention_dropout(scores.softmax(dim=-1))
        output = torch.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch, query_length, hidden_dim)
        return self.output_projection(output)


class RotaryEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = RotaryAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        token: torch.Tensor,
        positions: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        use_sdpa: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(token)
        token = token + self.residual_dropout(self.attention(
            normalized,
            normalized,
            positions,
            positions,
            attention_mask=attention_mask,
            key_mask=key_mask,
            use_sdpa=use_sdpa,
        ))
        return token + self.residual_dropout(self.ffn(self.ffn_norm(token)))


class AdaRMSNorm(nn.Module):
    """OpenPI-style RMS normalization with zero-initialized scale, shift and residual gate."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, token: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        variance = token.float().square().mean(dim=-1, keepdim=True)
        normalized = token * torch.rsqrt(variance + 1e-6).to(token.dtype)
        scale, shift, gate = self.modulation(condition).unsqueeze(1).chunk(3, dim=-1)
        return normalized * (1.0 + scale) + shift, gate


class RotaryFlowBlock(nn.Module):
    """Decoder block with separate geometry, semantics, and flow-time conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.self_norm = AdaRMSNorm(hidden_dim)
        self.self_attention = RotaryAttention(hidden_dim, num_heads, dropout)
        self.cross_norm = AdaRMSNorm(hidden_dim)
        self.cross_attention = RotaryAttention(hidden_dim, num_heads, dropout)
        self.semantic_norm = AdaRMSNorm(hidden_dim)
        self.semantic_attention = RotaryAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = AdaRMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        semantic_memory: torch.Tensor,
        condition: torch.Tensor,
        token_positions: torch.Tensor,
        memory_positions: torch.Tensor,
        semantic_positions: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized, gate = self.self_norm(token, condition)
        update = self.self_attention(
            normalized, normalized, token_positions, token_positions, self_attention_mask
        )
        token = token + self.residual_dropout(update) * gate
        normalized, gate = self.cross_norm(token, condition)
        update = self.cross_attention(normalized, memory, token_positions, memory_positions)
        token = token + self.residual_dropout(update) * gate
        normalized, gate = self.semantic_norm(token, condition)
        update = self.semantic_attention(
            normalized, semantic_memory, token_positions, semantic_positions,
        )
        token = token + self.residual_dropout(update) * gate
        normalized, gate = self.ffn_norm(token, condition)
        return token + self.residual_dropout(self.ffn(normalized)) * gate
