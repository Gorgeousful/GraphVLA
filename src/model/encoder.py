"""Shared entity encoder with CLS-mediated cross-entity communication."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTOR_NUM_POINTS, NUM_ENTITIES, POINT_FEATURE_DIM


def _encoder_block(hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> nn.Module:
    return nn.TransformerEncoderLayer(
        hidden_dim,
        num_heads,
        int(hidden_dim * mlp_ratio),
        dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class EntityEncoder(nn.Module):
    """Encode point sets locally and exchange information globally through entity CLS tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        condition_dim: int,
        max_history: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.point_stem = nn.Sequential(
            nn.Linear(POINT_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.role_embedding = nn.Embedding(NUM_ENTITIES, hidden_dim)
        self.time_embedding = nn.Embedding(max_history, hidden_dim)
        self.actor_keypoint_embedding = nn.Embedding(ACTOR_NUM_POINTS, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.local_blocks = nn.ModuleList([
            _encoder_block(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.global_blocks = nn.ModuleList([
            _encoder_block(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gradient_checkpointing = False
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 5 or points.shape[2] != NUM_ENTITIES:
            raise ValueError(f"Expected entity_points [B,T,{NUM_ENTITIES},P,3], got {points.shape}")
        batch, steps, entities, num_points, _ = points.shape
        if point_mask.shape != points.shape[:-1]:
            raise ValueError(f"point mask {point_mask.shape} does not match points {points.shape}")
        if condition.shape[:2] != (batch, entities):
            raise ValueError(f"Expected entity_condition [B,{entities},C], got {condition.shape}")

        roles = torch.arange(entities, device=points.device)
        times = torch.arange(steps, device=points.device)
        role_emb = self.role_embedding(roles)[None, None, :, None]
        time_emb = self.time_embedding(times)[None, :, None, None]
        tokens = self.point_stem(points) + role_emb + time_emb
        actor_ids = torch.arange(ACTOR_NUM_POINTS, device=points.device)
        tokens[:, :, 0, :ACTOR_NUM_POINTS] += self.actor_keypoint_embedding(actor_ids)[None, None]

        cls = self.cls_token.expand(batch, steps, entities, -1)
        cls = cls + role_emb.squeeze(3) + time_emb.squeeze(3)
        cls = cls + self.condition_projection(condition)[:, None]
        relation_local: torch.Tensor | None = None

        for local_block, global_block in zip(self.local_blocks, self.global_blocks, strict=True):
            local = torch.cat([cls.unsqueeze(3), tokens], dim=3)
            local = local.reshape(batch * steps * entities, num_points + 1, -1)
            padding = torch.cat([
                torch.zeros(batch, steps, entities, 1, dtype=torch.bool, device=points.device),
                ~point_mask.bool(),
            ], dim=3).reshape(batch * steps * entities, num_points + 1)
            if self.gradient_checkpointing and self.training:
                local = checkpoint(
                    local_block, local, src_key_padding_mask=padding,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                local = local_block(local, src_key_padding_mask=padding)
            local = local.view(batch, steps, entities, num_points + 1, -1)
            cls, tokens = local[:, :, :, 0], local[:, :, :, 1:]
            if relation_local is None:
                relation_local = cls[:, -1, 1:3].clone()
            global_cls = cls.reshape(batch, steps * entities, -1)
            if self.gradient_checkpointing and self.training:
                global_cls = checkpoint(
                    global_block, global_cls, use_reentrant=False, preserve_rng_state=False,
                )
            else:
                global_cls = global_block(global_cls)
            cls = global_cls.view(batch, steps, entities, -1)

        assert relation_local is not None
        return self.norm(cls.reshape(batch, steps * entities, -1)), self.norm(relation_local)
