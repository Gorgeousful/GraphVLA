"""D4RT-style token memory encoder for EEF/object embeddings."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embedding import LearnableFrameObjectPointEmbedding

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
    step is encoded independently. By default the encoder keeps point tokens as
    the main output path; a cls token can be enabled for object-level pooling.
    """

    def __init__(
        self,
        point_dim: int,
        num_points: int = 128,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        use_cls_token: bool = False,
        num_register_tokens: int = 0,
        condition_dim: int | None = 384,
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
        self.use_cls_token = use_cls_token
        self.condition_dim = condition_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_proj = None
        self.condition_film = None
        if condition_dim is not None:
            self.condition_proj = nn.Identity() if condition_dim == hidden_dim else nn.Linear(condition_dim, hidden_dim)
            self.condition_film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_points + int(use_cls_token), hidden_dim))
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
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(self._init_weights)
        if self.condition_film is not None:
            nn.init.zeros_(self.condition_film.weight)
            nn.init.zeros_(self.condition_film.bias)

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
        if not self.use_cls_token:
            point_pos = self.pos_embed.float().transpose(1, 2)
            point_pos = F.interpolate(point_pos, size=num_points, mode="linear", align_corners=False)
            return point_pos.transpose(1, 2).to(dtype=x.dtype)
        cls_pos = self.pos_embed[:, :1].float()
        point_pos = self.pos_embed[:, 1:].float().transpose(1, 2)
        point_pos = F.interpolate(point_pos, size=num_points, mode="linear", align_corners=False)
        point_pos = point_pos.transpose(1, 2)
        return torch.cat([cls_pos, point_pos], dim=1).to(dtype=x.dtype)

    def prepare_tokens(self, point_feats: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        bsz, steps, num_objects, num_points, _ = point_feats.shape
        x = self.point_mlp(point_feats)
        if condition is not None:
            if self.condition_proj is None or self.condition_film is None:
                raise ValueError("condition_dim must be provided when passing condition to SetEncoderViT")
            if condition.ndim != 3:
                raise ValueError(f"Expected condition [B, N, C], got {condition.shape}")
            if condition.shape[0] != bsz or condition.shape[1] != num_objects:
                raise ValueError(
                    "condition must match batch/object dims: "
                    f"condition={condition.shape}, point_feats={point_feats.shape}"
                )
            cond = self.condition_proj(condition)
            gamma, beta = self.condition_film(cond).chunk(2, dim=-1)
            gamma = gamma[:, None, :, None, :].to(dtype=x.dtype)
            beta = beta[:, None, :, None, :].to(dtype=x.dtype)
            x = x * (1.0 + gamma) + beta
        x = x.reshape(bsz * steps * num_objects, num_points, self.hidden_dim)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.interpolate_pos_encoding(x, num_points)
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(x.shape[0], -1, -1)
            if self.use_cls_token:
                x = torch.cat([x[:, :1], registers, x[:, 1:]], dim=1)
            else:
                x = torch.cat([registers, x], dim=1)
        return x

    def forward(
        self,
        point_feats: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if point_feats.ndim != 5:
            raise ValueError(f"Expected point_feats [B, T, N, P, F], got {point_feats.shape}")
        bsz, steps, num_objects, num_points, point_dim = point_feats.shape
        if point_dim != self.point_dim:
            raise ValueError(f"Expected point dim {self.point_dim}, got {point_dim}")

        x = self.prepare_tokens(point_feats, condition=condition)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        point_start = self.num_register_tokens + int(self.use_cls_token)
        point_tokens = x[:, point_start : point_start + num_points].reshape(
            bsz, steps, num_objects, num_points, self.hidden_dim
        )
        if not self.use_cls_token:
            return point_tokens
        cls = x[:, 0].reshape(bsz, steps, num_objects, self.hidden_dim)
        return point_tokens, cls


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


class PointMemoryEncoder(nn.Module):
    """Encode point tokens into flat D4RT-style memory.

    Actor object_id is 0. Object ids are 1..N. Local attention is applied
    within each time step over all actor/object point tokens; global attention
    is applied over the flattened time-token sequence.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_pattern: str | None = "interleaved_local_global",
        dropout: float = 0.1,
        position_embedding: LearnableFrameObjectPointEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_pattern = _normalize_attention_pattern(attention_pattern)
        self.position_embedding = position_embedding
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

    def _with_position(
        self,
        tokens: torch.Tensor,
        object_offset: int,
    ) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError(f"Expected point tokens [B, T, N, P, C], got {tokens.shape}")
        if self.position_embedding is None:
            raise ValueError("position_embedding must be provided for PointMemoryEncoder")
        bsz, steps, num_tokens, num_points, hidden = tokens.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {hidden}")
        pos = self.position_embedding.encode_grid(
            num_frames=steps,
            num_objects=num_tokens,
            num_points=num_points,
            device=tokens.device,
            object_offset=object_offset,
        ).to(dtype=tokens.dtype)
        return tokens + pos.unsqueeze(0)

    def forward(
        self,
        object_tokens: torch.Tensor,
        actor_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        object_tokens = self._with_position(object_tokens, object_offset=1)
        bsz, steps, num_objects, num_points, hidden = object_tokens.shape
        per_step = [object_tokens.reshape(bsz, steps, num_objects * num_points, hidden)]

        if actor_tokens is not None:
            actor_tokens = self._with_position(actor_tokens, object_offset=0)
            if actor_tokens.shape[0] != bsz or actor_tokens.shape[1] != steps or actor_tokens.shape[-1] != hidden:
                raise ValueError(
                    "actor/object tokens must match batch/time/hidden dims: "
                    f"actor={actor_tokens.shape}, object={object_tokens.shape}"
                )
            per_step.insert(0, actor_tokens.reshape(bsz, steps, -1, hidden))

        step_tokens = torch.cat(per_step, dim=2)
        tokens_per_step = step_tokens.shape[2]
        flat_tokens = step_tokens.reshape(bsz, steps * tokens_per_step, hidden)

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

            flat_tokens = block(flat_tokens)
        return self.final_norm(flat_tokens)


