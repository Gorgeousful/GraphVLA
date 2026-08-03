"""Shared entity encoder with CLS-mediated cross-entity communication."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTOR_POINT_INDICES, NUM_ENTITIES, POINT_FEATURE_DIM
from src.model.temporal import RotaryEncoderBlock


class EntityEncoder(nn.Module):
    """Encode point sets locally and exchange information globally through entity CLS tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        max_history: int,
        cls_token_num: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if cls_token_num < 1:
            raise ValueError(f"cls_token_num must be at least 1, got {cls_token_num}")
        self.cls_token_num = cls_token_num
        self.point_stem = nn.Sequential(
            nn.Linear(POINT_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.max_history = max_history
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, cls_token_num, hidden_dim))
        self.local_blocks = nn.ModuleList([
            RotaryEncoderBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        head_dim = hidden_dim // num_heads
        global_axis_dims = (head_dim // 2, head_dim - head_dim // 2)
        self.global_blocks = nn.ModuleList([
            RotaryEncoderBlock(hidden_dim, num_heads, mlp_ratio, dropout, global_axis_dims)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gradient_checkpointing = False
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.ndim != 5 or points.shape[2] != NUM_ENTITIES:
            raise ValueError(f"Expected entity_points [B,T,{NUM_ENTITIES},P,3], got {points.shape}")
        batch, steps, entities, num_points, _ = points.shape
        if steps > self.max_history:
            raise ValueError(f"Expected at most {self.max_history} history steps, got {steps}")
        if point_mask.shape != points.shape[:-1]:
            raise ValueError(f"point mask {point_mask.shape} does not match points {points.shape}")
        tokens = self.point_stem(points)
        cls = self.cls_token.expand(batch, steps, entities, -1, -1)

        cls_positions = torch.arange(-self.cls_token_num, 0, device=points.device)
        point_positions = torch.zeros(
            entities, num_points, device=points.device, dtype=cls_positions.dtype,
        )
        point_positions[0, :len(ACTOR_POINT_INDICES)] = torch.arange(
            len(ACTOR_POINT_INDICES), device=points.device,
        )
        local_positions = torch.cat([
            cls_positions.expand(entities, -1), point_positions,
        ], dim=1)[None, None].expand(batch, steps, -1, -1).reshape(
            batch * steps * entities, self.cls_token_num + num_points,
        )

        history_positions = torch.arange(1 - steps, 1, device=points.device)
        global_entity_tokens = steps * entities * self.cls_token_num
        entity_slots = torch.arange(entities * self.cls_token_num, device=points.device)
        global_positions = torch.stack([
            history_positions.repeat_interleave(entities * self.cls_token_num),
            entity_slots.repeat(steps),
        ], dim=-1)

        for local_block, global_block in zip(self.local_blocks, self.global_blocks, strict=True):
            local = torch.cat([cls, tokens], dim=3)
            local = local.reshape(
                batch * steps * entities, num_points + self.cls_token_num, -1
            )
            padding = torch.cat([
                torch.zeros(
                    batch, steps, entities, self.cls_token_num,
                    dtype=torch.bool, device=points.device,
                ),
                ~point_mask.bool(),
            ], dim=3).reshape(
                batch * steps * entities, num_points + self.cls_token_num
            )
            local_attention_mask = (~padding)[:, None, :].expand(
                -1, num_points + self.cls_token_num, -1,
            )
            if self.gradient_checkpointing and self.training:
                local = checkpoint(
                    local_block, local, local_positions, local_attention_mask,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                local = local_block(local, local_positions, local_attention_mask)
            local = local.view(
                batch, steps, entities, num_points + self.cls_token_num, -1
            )
            cls = local[:, :, :, :self.cls_token_num]
            tokens = local[:, :, :, self.cls_token_num:]
            global_cls = cls.reshape(batch, global_entity_tokens, -1)
            if self.gradient_checkpointing and self.training:
                global_cls = checkpoint(
                    global_block, global_cls, global_positions,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                global_cls = global_block(global_cls, global_positions)
            cls = global_cls.view(
                batch, steps, entities, self.cls_token_num, -1
            )

        current_cls = cls[:, -1]
        relation_local = current_cls[:, 1:3].flatten(1, 2)
        history_memory = cls.flatten(1, 3)
        return (
            self.norm(history_memory),
            global_positions,
            self.norm(relation_local),
        )
