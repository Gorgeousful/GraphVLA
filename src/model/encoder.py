"""Shared entity encoder with CLS-mediated cross-entity communication."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTOR_NUM_POINTS, NUM_ENTITIES, POINT_FEATURE_DIM
from src.model.temporal import RotaryEncoderBlock


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
        self.actor_keypoint_embedding = nn.Embedding(ACTOR_NUM_POINTS, hidden_dim)
        self.role_type_embedding = nn.Embedding(NUM_ENTITIES, hidden_dim)
        self.action_projection = nn.Linear(condition_dim, hidden_dim)
        self.degree_projection = nn.Linear(condition_dim, hidden_dim)
        self.scene_type_embedding = nn.Embedding(2, hidden_dim)
        self.null_degree_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, cls_token_num, hidden_dim))
        self.local_blocks = nn.ModuleList([
            _encoder_block(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.global_blocks = nn.ModuleList([
            RotaryEncoderBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gradient_checkpointing = False
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: torch.Tensor,
        scene_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 5 or points.shape[2] != NUM_ENTITIES:
            raise ValueError(f"Expected entity_points [B,T,{NUM_ENTITIES},P,3], got {points.shape}")
        batch, steps, entities, num_points, _ = points.shape
        if steps > self.max_history:
            raise ValueError(f"Expected at most {self.max_history} history steps, got {steps}")
        if point_mask.shape != points.shape[:-1]:
            raise ValueError(f"point mask {point_mask.shape} does not match points {points.shape}")
        if scene_condition.shape[:2] != (batch, 2):
            raise ValueError(f"Expected scene_condition [B,2,C], got {scene_condition.shape}")

        tokens = self.point_stem(points)
        actor_ids = torch.arange(ACTOR_NUM_POINTS, device=points.device)
        tokens[:, :, 0, :ACTOR_NUM_POINTS] += self.actor_keypoint_embedding(actor_ids)[None, None]

        cls = self.cls_token.expand(batch, steps, entities, -1, -1)
        role_types = self.role_type_embedding(
            torch.arange(entities, device=points.device)
        )[None, None, :, None]
        scene_types = self.scene_type_embedding(torch.arange(2, device=points.device))[None]
        action_token = self.action_projection(scene_condition[:, 0])
        projected_degree = self.degree_projection(scene_condition[:, 1])
        has_degree = scene_condition[:, 1].abs().sum(dim=-1, keepdim=True) > 0
        degree_token = torch.where(
            has_degree,
            projected_degree,
            self.null_degree_token.expand(batch, -1),
        )
        scene_tokens = torch.stack([action_token, degree_token], dim=1) + scene_types
        history_positions = torch.arange(1 - steps, 1, device=points.device)
        global_positions = torch.cat([
            history_positions.repeat_interleave(entities * self.cls_token_num),
            torch.zeros(2, device=points.device, dtype=history_positions.dtype),
        ])
        global_entity_tokens = steps * entities * self.cls_token_num

        for layer_index, (local_block, global_block) in enumerate(
            zip(self.local_blocks, self.global_blocks, strict=True)
        ):
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
            if self.gradient_checkpointing and self.training:
                local = checkpoint(
                    local_block, local, src_key_padding_mask=padding,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                local = local_block(local, src_key_padding_mask=padding)
            local = local.view(
                batch, steps, entities, num_points + self.cls_token_num, -1
            )
            cls = local[:, :, :, :self.cls_token_num]
            tokens = local[:, :, :, self.cls_token_num:]
            if layer_index == 0:
                cls = cls + role_types
            global_cls = torch.cat(
                [cls.reshape(batch, global_entity_tokens, -1), scene_tokens], dim=1
            )
            if self.gradient_checkpointing and self.training:
                global_cls = checkpoint(
                    global_block, global_cls, global_positions,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                global_cls = global_block(global_cls, global_positions)
            cls = global_cls[:, :global_entity_tokens].view(
                batch, steps, entities, self.cls_token_num, -1
            )
            scene_tokens = global_cls[:, global_entity_tokens:]

        current_cls = cls[:, -1]
        relation_local = current_cls[:, 1:3].flatten(1, 2)
        memory = torch.cat([current_cls.flatten(1, 2), scene_tokens], dim=1)
        return self.norm(memory), self.norm(relation_local)
