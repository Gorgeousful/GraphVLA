"""D4RT-style token memory encoder for EEF/object embeddings."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import RelativeTokenPositionEmbedding
from .embedding import sinusoidal_scalar_embedding


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


#: ===========================================
class _DropPath(nn.Module):
    """DINO/timm-style stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class _LayerScale(nn.Module):
    """DINO-style residual branch scaling."""

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _DinoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {dim} and {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(bsz, num_tokens, dim)
        return self.proj(x)


class _DinoBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path: float = 0.0,
        init_values: float | None = 1e-5,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _DinoAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
        )
        self.ls1 = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=ffn_bias),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=ffn_bias),
        )
        self.ls2 = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SetEncoderViT(nn.Module):
    """DINO-style ViT encoder for per-object tracked point tokens.

    The point MLP replaces DINO conv patch embedding. Each object at each time
    step is encoded independently, with a cls token aggregating the object
    feature after self-attention over cls/register/point tokens.
    """

    def __init__(
        self,
        point_dim: int,
        num_points: int = 128,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: float | None = 0.01,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_register_tokens < 0:
            raise ValueError(f"num_register_tokens must be non-negative, got {num_register_tokens}")
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        self.num_register_tokens = num_register_tokens
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_points + 1, hidden_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, hidden_dim)) if num_register_tokens else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * num_layers
        else:
            dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                _DinoBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    init_values=init_values,
                    qk_norm=qk_norm,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def interpolate_pos_encoding(self, x: torch.Tensor, num_points: int) -> torch.Tensor:
        if num_points == self.num_points:
            return self.pos_embed.to(dtype=x.dtype)
        cls_pos = self.pos_embed[:, :1].float()
        point_pos = self.pos_embed[:, 1:].float().transpose(1, 2)
        point_pos = F.interpolate(point_pos, size=num_points, mode="linear", align_corners=False)
        point_pos = point_pos.transpose(1, 2)
        return torch.cat([cls_pos, point_pos], dim=1).to(dtype=x.dtype)

    def prepare_tokens(self, point_feats: torch.Tensor) -> torch.Tensor:
        bsz, steps, num_objects, num_points, _ = point_feats.shape
        x = self.point_mlp(point_feats)
        x = x.reshape(bsz * steps * num_objects, num_points, self.hidden_dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.interpolate_pos_encoding(x, num_points)
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([x[:, :1], registers, x[:, 1:]], dim=1)
        return x

    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        if point_feats.ndim != 5:
            raise ValueError(f"Expected point_feats [B, T, N, P, F], got {point_feats.shape}")
        bsz, steps, num_objects, num_points, point_dim = point_feats.shape
        if point_dim != self.point_dim:
            raise ValueError(f"Expected point dim {self.point_dim}, got {point_dim}")

        x = self.prepare_tokens(point_feats)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        return cls.reshape(bsz, steps, num_objects, self.hidden_dim)


#: ===========================================
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


class SetEncoderQuery(nn.Module):
    """Encode per-object tracked point sets into object tokens.

    Args:
        F = [u,v,d_rel,vis, d_metric, d_metric_mask]
        if lacking, fill in with -1; others are normalized to 0~1
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
        point_id = torch.arange(num_points, device=point_feats.device)
        point_pos = sinusoidal_scalar_embedding(point_id, self.hidden_dim).to(dtype=x.dtype)
        x = x + point_pos[None, None, None, :, :]
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



#: ===========================================
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
        position_embedding: nn.Module | None = None,
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
