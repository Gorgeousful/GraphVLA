"""Shared entity encoder with CLS-mediated cross-entity communication."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.common.schema import NUM_ENTITIES, POINT_FEATURE_DIM
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
    """Alternate local CLS/role/point attention with register-only or dense global attention."""

    def __init__(
        self,
        hidden_dim: int,
        actor_num_points: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        condition_dim: int,
        max_history: int,
        cls_token_num: int = 1,
        dropout: float = 0.0,
        global_layer_types: tuple[int, ...] | list[int] | None = None,
        node_attention_mode: str = "full",
        encoder_output_type: str = "current",
        include_scene_condition: bool = True,
    ) -> None:
        super().__init__()
        if actor_num_points < 1:
            raise ValueError(f"actor_num_points must be at least 1, got {actor_num_points}")
        if cls_token_num < 1:
            raise ValueError(f"cls_token_num must be at least 1, got {cls_token_num}")
        if global_layer_types is None:
            global_layer_types = (0,) * num_layers
        self.global_layer_types = tuple(global_layer_types)
        if len(self.global_layer_types) != num_layers:
            raise ValueError(
                f"global_layer_types must contain {num_layers} entries, "
                f"got {len(self.global_layer_types)}"
            )
        if any(layer_type not in (0, 1) for layer_type in self.global_layer_types):
            raise ValueError(
                f"global_layer_types entries must be 0 (register) or 1 (dense), "
                f"got {self.global_layer_types}"
            )
        if node_attention_mode not in ("full", "role_chain"):
            raise ValueError(
                "node_attention_mode must be 'full' or 'role_chain', "
                f"got {node_attention_mode!r}"
            )
        if encoder_output_type not in ("current", "all"):
            raise ValueError(
                "encoder_output_type must be 'current' or 'all', "
                f"got {encoder_output_type!r}"
            )
        self.actor_num_points = actor_num_points
        self.cls_token_num = cls_token_num
        self.node_attention_mode = node_attention_mode
        self.encoder_output_type = encoder_output_type
        self.include_scene_condition = include_scene_condition
        self.scene_token_count = 2 if include_scene_condition else 0
        self.point_stem = nn.Sequential(
            nn.Linear(POINT_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.max_history = max_history
        self.actor_keypoint_embedding = nn.Embedding(actor_num_points, hidden_dim)
        self.role_type_embedding = nn.Embedding(NUM_ENTITIES, hidden_dim)
        self.action_projection = (
            nn.Linear(condition_dim, hidden_dim) if include_scene_condition else None
        )
        self.degree_projection = (
            nn.Linear(condition_dim, hidden_dim) if include_scene_condition else None
        )
        self.scene_type_embedding = (
            nn.Embedding(2, hidden_dim) if include_scene_condition else None
        )
        self.null_degree_token = (
            nn.Parameter(torch.zeros(1, hidden_dim)) if include_scene_condition else None
        )
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
        scene_condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 5 or points.shape[2] != NUM_ENTITIES:
            raise ValueError(f"Expected entity_points [B,T,{NUM_ENTITIES},P,3], got {points.shape}")
        batch, steps, entities, num_points, _ = points.shape
        if steps > self.max_history:
            raise ValueError(f"Expected at most {self.max_history} history steps, got {steps}")
        if point_mask.shape != points.shape[:-1]:
            raise ValueError(f"point mask {point_mask.shape} does not match points {points.shape}")
        if self.include_scene_condition:
            if scene_condition is None or scene_condition.shape[:2] != (batch, 2):
                shape = None if scene_condition is None else tuple(scene_condition.shape)
                raise ValueError(f"Expected scene_condition [B,2,C], got {shape}")
        elif scene_condition is not None:
            raise ValueError("scene_condition must be None when encoder scene tokens are disabled")

        tokens = self.point_stem(points)
        actor_ids = torch.arange(self.actor_num_points, device=points.device)
        tokens[:, :, 0, :self.actor_num_points] += self.actor_keypoint_embedding(actor_ids)[None, None]

        cls = self.cls_token.expand(batch, steps, entities, -1, -1)
        role_tokens = self.role_type_embedding(
            torch.arange(entities, device=points.device)
        )[None, None, :, None].expand(batch, steps, -1, -1, -1)
        if self.include_scene_condition:
            assert scene_condition is not None
            assert self.scene_type_embedding is not None
            assert self.action_projection is not None
            assert self.degree_projection is not None
            assert self.null_degree_token is not None
            scene_types = self.scene_type_embedding(
                torch.arange(self.scene_token_count, device=points.device)
            )[None]
            action_token = self.action_projection(scene_condition[:, 0])
            projected_degree = self.degree_projection(scene_condition[:, 1])
            has_degree = scene_condition[:, 1].abs().sum(dim=-1, keepdim=True) > 0
            degree_token = torch.where(
                has_degree,
                projected_degree,
                self.null_degree_token.expand(batch, -1),
            )
            scene_tokens = torch.stack([action_token, degree_token], dim=1) + scene_types
        else:
            scene_tokens = tokens.new_empty(batch, 0, tokens.shape[-1])
        history_positions = torch.arange(1 - steps, 1, device=points.device)
        global_positions = torch.cat([
            history_positions.repeat_interleave(entities * self.cls_token_num),
            torch.zeros(
                self.scene_token_count,
                device=points.device,
                dtype=history_positions.dtype,
            ),
        ])
        global_entity_tokens = steps * entities * self.cls_token_num
        dense_positions = torch.cat([
            history_positions.repeat_interleave(
                entities * (self.cls_token_num + num_points)
            ),
            torch.zeros(
                self.scene_token_count,
                device=points.device,
                dtype=history_positions.dtype,
            ),
        ])
        dense_entity_tokens = steps * entities * (self.cls_token_num + num_points)
        dense_key_mask = torch.cat([
            torch.cat([
                torch.ones(
                    batch, steps, entities, self.cls_token_num,
                    dtype=torch.bool, device=points.device,
                ),
                point_mask.bool(),
            ], dim=3).reshape(batch, dense_entity_tokens),
            torch.ones(
                batch, self.scene_token_count,
                dtype=torch.bool, device=points.device,
            ),
        ], dim=1)
        register_attention_mask = None
        dense_attention_mask = None
        if self.node_attention_mode == "role_chain":
            def make_attention_mask(tokens_per_entity: int) -> torch.Tensor:
                role_ids = torch.arange(entities, device=points.device).repeat_interleave(
                    tokens_per_entity
                ).repeat(steps)
                role_ids = torch.cat([
                    role_ids,
                    role_ids.new_full((self.scene_token_count,), -1),
                ])
                is_entity_query = role_ids >= 0
                is_entity_key = role_ids >= 0
                blocks_actor_target = (role_ids[:, None] - role_ids[None, :]).abs() > 1
                return ~(
                    is_entity_query[:, None] & is_entity_key[None, :] & blocks_actor_target
                )

            register_attention_mask = make_attention_mask(self.cls_token_num)
            dense_attention_mask = make_attention_mask(self.cls_token_num + num_points)
        for layer_index, (local_block, global_block) in enumerate(
            zip(self.local_blocks, self.global_blocks, strict=True)
        ):
            local = torch.cat([cls, role_tokens, tokens], dim=3)
            local = local.reshape(
                batch * steps * entities, num_points + self.cls_token_num + 1, -1
            )
            padding = torch.cat([
                torch.zeros(
                    batch, steps, entities, self.cls_token_num + 1,
                    dtype=torch.bool, device=points.device,
                ),
                ~point_mask.bool(),
            ], dim=3).reshape(
                batch * steps * entities, num_points + self.cls_token_num + 1
            )
            if self.gradient_checkpointing and self.training:
                local = checkpoint(
                    local_block, local, src_key_padding_mask=padding,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                local = local_block(local, src_key_padding_mask=padding)
            local = local.view(
                batch, steps, entities, num_points + self.cls_token_num + 1, -1
            )
            cls = local[:, :, :, :self.cls_token_num]
            role_tokens = local[:, :, :, self.cls_token_num:self.cls_token_num + 1]
            tokens = local[:, :, :, self.cls_token_num + 1:]
            if self.global_layer_types[layer_index] == 0:
                global_cls = torch.cat(
                    [cls.reshape(batch, global_entity_tokens, -1), scene_tokens], dim=1
                )
                if self.gradient_checkpointing and self.training:
                    global_cls = checkpoint(
                        global_block, global_cls, global_positions, None, False,
                        register_attention_mask,
                        use_reentrant=False, preserve_rng_state=True,
                    )
                else:
                    global_cls = global_block(
                        global_cls, global_positions,
                        attention_mask=register_attention_mask,
                    )
                cls = global_cls[:, :global_entity_tokens].view(
                    batch, steps, entities, self.cls_token_num, -1
                )
                scene_tokens = global_cls[:, global_entity_tokens:]
                continue

            dense = torch.cat([
                torch.cat([cls, tokens], dim=3).reshape(batch, dense_entity_tokens, -1),
                scene_tokens,
            ], dim=1)
            if self.gradient_checkpointing and self.training:
                dense = checkpoint(
                    global_block, dense, dense_positions, dense_key_mask,
                    dense_attention_mask is None, dense_attention_mask,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                dense = global_block(
                    dense,
                    dense_positions,
                    key_mask=dense_key_mask,
                    use_sdpa=dense_attention_mask is None,
                    attention_mask=dense_attention_mask,
                )
            dense_entities = dense[:, :dense_entity_tokens].view(
                batch, steps, entities, self.cls_token_num + num_points, -1
            )
            cls = dense_entities[:, :, :, :self.cls_token_num]
            tokens = dense_entities[:, :, :, self.cls_token_num:]
            tokens = tokens.masked_fill(~point_mask.bool().unsqueeze(-1), 0.0)
            scene_tokens = dense[:, dense_entity_tokens:]

        current_cls = cls[:, -1]
        relation_local = current_cls[:, 1:3].flatten(1, 2)
        entity_memory = (
            current_cls.flatten(1, 2)
            if self.encoder_output_type == "current"
            else cls.flatten(1, 3)
        )
        memory = torch.cat([entity_memory, scene_tokens], dim=1)
        return self.norm(memory), self.norm(relation_local)
