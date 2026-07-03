"""D4RT-style token memory encoder for EEF/object embeddings."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from .embedding import RelativeTokenPositionEmbedding


def _normalize_attention_pattern(pattern: str | None) -> str:
    raw = (pattern or "global").strip().lower()
    aliases = {
        "interleaved_local_framewise_and_global": "interleaved_local_global",
        "interleaved_local_framewise_global": "interleaved_local_global",
        "interleaved_local_and_global": "interleaved_local_global",
        "global": "global",
        "full_global": "global",
    }
    return aliases.get(raw, raw)


class SetEncoder(nn.Module):
    """Encode per-object tracked point sets into object tokens.

    Args:
        F = [u_0,v_0,d_rel_0, u,v,d_rel,vis, d_metric]
        if lacking, fill in with -1
        point_feats: ``[B, T, N, P, F]`` where ``P`` is the number of
            tracked points for each object. Returns ``[B, T, N, C]``.
    """

    def __init__(
        self,
        point_dim: int,
        hidden_dim: int = 1024,
        num_points: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ff_dim = int(math.ceil(hidden_dim * mlp_ratio))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_attn": nn.LayerNorm(hidden_dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            batch_first=True,
                        ),
                        "norm_ff": nn.LayerNorm(hidden_dim),
                        "ff": nn.Sequential(
                            nn.Linear(hidden_dim, ff_dim),
                            nn.GELU(),
                            nn.Linear(ff_dim, hidden_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.norm_pool_q = nn.LayerNorm(hidden_dim)
        self.norm_pool_kv = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        if point_feats.ndim != 5:
            raise ValueError(f"Expected point_feats [B, T, N, P, F], got {point_feats.shape}")
        bsz, steps, num_objects, num_points, point_dim = point_feats.shape
        if point_dim != self.point_dim:
            raise ValueError(f"Expected point dim {self.point_dim}, got {point_dim}")
        if num_points > self.num_points:
            raise ValueError(f"Expected at most {self.num_points} points, got {num_points}")

        x = self.point_mlp(point_feats)
        x = x.reshape(bsz * steps * num_objects, num_points, self.hidden_dim)
        for block in self.blocks:
            q = block["norm_attn"](x)
            attn_out, _ = block["attn"](
                q,
                q,
                q,
                need_weights=False,
            )
            x = x + attn_out
            x = x + block["ff"](block["norm_ff"](x))

        query = self.pool_query.expand(x.shape[0], -1, -1)
        query = self.norm_pool_q(query)
        kv = self.norm_pool_kv(x)
        pooled, _ = self.pool_attn(
            query,
            kv,
            kv,
            need_weights=False,
        )
        pooled = self.out_norm(pooled.squeeze(1))
        return pooled.reshape(bsz, steps, num_objects, self.hidden_dim)


class PointEncoder(nn.Module):
    """DP3-style PointNet encoder for per-object point features.

    Supports ``[B, T, N, P, F]`` inputs and returns ``[B, T, N, C]``.
    The core implementation follows DP3's point cloud encoder: per-point MLP,
    max pooling over points, then a final projection.
    """

    def __init__(
        self,
        point_dim: int,
        hidden_dim: int = 1024,
        block_channels: tuple[int, ...] = (64, 128, 256, 512),
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if not block_channels:
            raise ValueError("block_channels must contain at least one channel")
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim

        layers: list[nn.Module] = []
        in_dim = point_dim
        for idx, out_dim in enumerate(block_channels):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            if idx < len(block_channels) - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channels[-1], hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channels[-1], hidden_dim)
        else:
            raise ValueError(f"Unsupported final_norm: {final_norm}")

    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        if point_feats.ndim != 5:
            raise ValueError(f"Expected point_feats [B, T, N, P, F], got {point_feats.shape}")
        bsz, steps, num_objects, num_points, point_dim = point_feats.shape
        if point_dim != self.point_dim:
            raise ValueError(f"Expected point dim {self.point_dim}, got {point_dim}")

        x = point_feats.reshape(bsz * steps * num_objects, num_points, point_dim)
        x = self.mlp(x)
        x = torch.max(x, dim=1).values
        x = self.final_projection(x)
        return x.reshape(bsz, steps, num_objects, self.hidden_dim)


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention + MLP.

    This is adapted from Open-d4rt's encoder block.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1) -> None:
        super().__init__()
        ff_dim = int(math.ceil(hidden_dim * mlp_ratio))
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_attn(tokens)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class TokenMemoryEncoder(nn.Module):
    """Encode embedded EEF/object tokens into flat D4RT-style memory.

    Args:
        tokens: ``[B, T, P, C]`` where ``P = 1 + N``. Token id 0 is expected
            to be the robot EEF state token, and ids 1..N are object tokens.

    Returns:
        Flat memory tokens ``[B, T * P (+ extra), C]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_pattern: str | None = "interleaved_local_global",
        dropout: float = 0.1,
        position_embedding: RelativeTokenPositionEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_pattern = _normalize_attention_pattern(attention_pattern)
        self.position_embedding = position_embedding or RelativeTokenPositionEmbedding(hidden_dim=hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.block_modes = self._build_block_modes(num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _build_block_modes(self, num_layers: int) -> list[Literal["local", "global"]]:
        if self.attention_pattern == "interleaved_local_global":
            return ["local" if (i % 2 == 0) else "global" for i in range(num_layers)]
        if self.attention_pattern == "global":
            return ["global"] * num_layers
        raise ValueError(f"Unsupported attention_pattern: {self.attention_pattern}")

    def forward(self, tokens: torch.Tensor, extra_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens [B, T, P, C], got {tokens.shape}")
        bsz, steps, tokens_per_step, hidden = tokens.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {hidden}")

        x = tokens + self.position_embedding.encode_grid(steps, tokens_per_step, tokens.device).to(dtype=tokens.dtype)
        flat_tokens = x.reshape(bsz, steps * tokens_per_step, hidden)
        token_count = flat_tokens.shape[1]

        if extra_tokens is not None:
            if extra_tokens.ndim != 3:
                raise ValueError(f"Expected extra_tokens [B, N_extra, C], got {extra_tokens.shape}")
            if extra_tokens.shape[0] != bsz or extra_tokens.shape[2] != hidden:
                raise ValueError(
                    f"extra_tokens must match batch and hidden dim: expected [B={bsz}, *, C={hidden}], "
                    f"got {extra_tokens.shape}"
                )

        for mode, block in zip(self.block_modes, self.blocks):
            if mode == "local":
                local = flat_tokens.reshape(bsz, steps, tokens_per_step, hidden).reshape(
                    bsz * steps, tokens_per_step, hidden
                )
                local = block(local)
                flat_tokens = local.reshape(bsz, steps, tokens_per_step, hidden).reshape(
                    bsz, steps * tokens_per_step, hidden
                )
                continue

            if extra_tokens is None:
                flat_tokens = block(flat_tokens)
                continue

            merged = torch.cat([flat_tokens, extra_tokens], dim=1)
            merged = block(merged)
            flat_tokens = merged[:, :token_count]
            extra_tokens = merged[:, token_count:]

        if extra_tokens is not None:
            flat_tokens = torch.cat([flat_tokens, extra_tokens], dim=1)
        return self.final_norm(flat_tokens)
