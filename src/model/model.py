"""D4RT-style token query model for GraphVLA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.decoder import IndependentQueryDecoder
from src.model.encoder import SetEncoderViT
from src.model.heads import PredictionHeads

from src.model.embedding import FrameQueryEmbedder
from src.model.embedding import LearnableFrameObjectPointEmbedding
from src.model.embedding import PointQueryEmbedder
from src.model.encoder import PointMemoryEncoder


class PointQueryModel(nn.Module):
    """D4RT-style frame-object-point query model for tracked object points.

    SetEncoderViT keeps per-point tokens. The memory contains actor point
    tokens plus object point tokens over the history-to-now window. Queries ask
    for object_id/point_id at a relative frame_id and produce query-level
    head outputs, e.g. [B, Q, 6] for output_dims={"point": 6}.
    """

    def __init__(
        self,
        point_dim: int = 6,
        num_points: int = 32,
        actor_num_points: int = 3,
        set_hidden_dim: int = 384,
        set_layers: int = 12,
        set_heads: int = 6,
        set_mlp_ratio: float = 4.0,
        set_register_tokens: int = 0,
        condition_dim: int | None = 384*3,

        encoder_hidden_dim: int = 1024,
        encoder_layers: int = 24,
        encoder_heads: int = 16,
        encoder_mlp_ratio: float = 4.0,

        decoder_hidden_dim: int = 1024,
        decoder_layers: int = 8,
        decoder_heads: int = 16,
        decoder_mlp_ratio: float = 4.0,

        output_dims: dict[str, int] | None = None,

        num_query_types: int = 0,
        num_frame_query_types: int = 0,
        max_objects: int = 3, # 1+2
        min_frame: int = -15,
        max_frame: int = 15,
        attention_pattern: str | None = "interleaved_local_global",
        dropout: float = 0.1,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        #: pre-encoder
        self.num_points = num_points
        self.actor_num_points = actor_num_points
        self.max_objects = max_objects
        self.weights = {} if weights is None else dict(weights)
        self.object_encoder = SetEncoderViT(
            point_dim=point_dim,
            hidden_dim=set_hidden_dim,
            num_points=num_points,
            num_heads=set_heads,
            num_layers=set_layers,
            mlp_ratio=set_mlp_ratio,
            num_register_tokens=set_register_tokens,
            condition_dim=condition_dim,
            use_cls_token=False,
        )
        self.actor_encoder = SetEncoderViT(
            point_dim=point_dim,
            hidden_dim=set_hidden_dim,
            num_points=actor_num_points,
            num_heads=set_heads,
            num_layers=set_layers,
            mlp_ratio=set_mlp_ratio,
            num_register_tokens=set_register_tokens,
            condition_dim=condition_dim,
            use_cls_token=False,
        )
        self.set_proj = (
            nn.Identity()
            if set_hidden_dim == encoder_hidden_dim
            else nn.Linear(set_hidden_dim, encoder_hidden_dim)
        )
        #: encoder
        max_points = max(num_points, actor_num_points)
        self.encoder_position_embedding = LearnableFrameObjectPointEmbedding(
            hidden_dim=encoder_hidden_dim,
            max_objects=max_objects,
            max_points=max_points,
            min_frame=min_frame,
            max_frame=max_frame,
        )
        self.encoder = PointMemoryEncoder(
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            attention_pattern=attention_pattern,
            dropout=dropout,
            position_embedding=self.encoder_position_embedding,
        )
        self.memory_proj = (
            nn.Identity()
            if decoder_hidden_dim == encoder_hidden_dim
            else nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        )
        #: decoder
        self.query_position_embedding = (
            self.encoder_position_embedding
            if decoder_hidden_dim == encoder_hidden_dim
            else LearnableFrameObjectPointEmbedding(
                hidden_dim=decoder_hidden_dim,
                max_objects=max_objects,
                max_points=max_points,
                min_frame=min_frame,
                max_frame=max_frame,
            )
        )
        self.query_embedder = PointQueryEmbedder(
            hidden_dim=decoder_hidden_dim,
            num_query_types=num_query_types,
            position_embedding=self.query_position_embedding,
        )
        self.frame_query_embedder = FrameQueryEmbedder(
            hidden_dim=decoder_hidden_dim,
            num_query_types=num_frame_query_types,
            min_frame=min_frame,
            max_frame=max_frame,
        )
        self.decoder = IndependentQueryDecoder(
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            dropout=dropout,
        )
        #: heads
        if output_dims is None:
            raise ValueError("output_dims must be provided")
        self.heads = PredictionHeads(hidden_dim=decoder_hidden_dim, output_dims=output_dims)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.object_encoder.set_gradient_checkpointing(enabled)
        self.actor_encoder.set_gradient_checkpointing(enabled)
        self.encoder.set_gradient_checkpointing(enabled)
        self.decoder.set_gradient_checkpointing(enabled)

    def encode_sets(
        self,
        point_feats: torch.Tensor,
        actor_feats: torch.Tensor,
        object_condition: torch.Tensor | None = None,
        actor_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        object_tokens = self.object_encoder(point_feats, condition=object_condition)
        actor_tokens = self.actor_encoder(actor_feats, condition=actor_condition)
        object_tokens = self.set_proj(object_tokens)
        actor_tokens = self.set_proj(actor_tokens)
        if actor_tokens.shape[:2] != object_tokens.shape[:2] or actor_tokens.shape[-1] != object_tokens.shape[-1]:
            raise ValueError(
                "actor/object point tokens must match batch/time/hidden dims: "
                f"actor={actor_tokens.shape}, object={object_tokens.shape}"
            )
        return object_tokens, actor_tokens

    def encode(
        self,
        point_feats: torch.Tensor,
        actor_feats: torch.Tensor,
        object_condition: torch.Tensor | None = None,
        actor_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        object_tokens, actor_tokens = self.encode_sets(
            point_feats=point_feats,
            actor_feats=actor_feats,
            object_condition=object_condition,
            actor_condition=actor_condition,
        )
        memory = self.encoder(object_tokens=object_tokens, actor_tokens=actor_tokens)
        return self.memory_proj(memory)

    def decode(
        self,
        memory: torch.Tensor,
        object_id: torch.Tensor,
        point_id: torch.Tensor,
        frame_id: torch.Tensor,
        query_type: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        query_tokens = self.query_embedder(
            object_id=object_id,
            point_id=point_id,
            frame_id=frame_id,
            query_type=query_type,
        )
        decoded = self.decoder(query_tokens=query_tokens, memory_tokens=memory)
        return self.heads(decoded, head_names=head_names)

    def decode_frame(
        self,
        memory: torch.Tensor,
        frame_id: torch.Tensor,
        query_type: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        query_tokens = self.frame_query_embedder(frame_id=frame_id, query_type=query_type)
        decoded = self.decoder(query_tokens=query_tokens, memory_tokens=memory)
        return self.heads(decoded, head_names=head_names)

    def infer(
        self,
        point_feats: torch.Tensor,
        actor_feats: torch.Tensor,
        object_id: torch.Tensor,
        point_id: torch.Tensor,
        frame_id: torch.Tensor,
        object_condition: torch.Tensor | None = None,
        actor_condition: torch.Tensor | None = None,
        query_type: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,

        frame_query_frame_id: torch.Tensor | None = None,
        frame_query_type: torch.Tensor | None = None,
        frame_head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode(
            point_feats=point_feats,
            actor_feats=actor_feats,
            object_condition=object_condition,
            actor_condition=actor_condition,
        )
        outputs = self.decode(
            memory=memory,
            object_id=object_id,
            point_id=point_id,
            frame_id=frame_id,
            query_type=query_type,
            head_names=head_names,
        )
        if frame_query_frame_id is not None:
            outputs.update(
                self.decode_frame(
                    memory=memory,
                    frame_id=frame_query_frame_id,
                    query_type=frame_query_type,
                    head_names=frame_head_names,
                )
            )
        return outputs

    def forward(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = batch["target"]
        if not isinstance(target, dict):
            raise ValueError("batch[target] must be a dict")

        weights = self.weights if weights is None else weights
        metrics: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None

        memory = self.encode(
            point_feats=batch["point_feats"],
            actor_feats=batch["actor_feats"],
            object_condition=batch.get("object_condition"),
            actor_condition=batch.get("actor_condition"),
        )

        if "point" in target:
            point_outputs = self.decode(
                memory=memory,
                object_id=batch["object_id"],
                point_id=batch["point_id"],
                frame_id=batch["frame_id"],
                head_names="point",
            )
            point_err = (point_outputs["point"] - target["point"]).abs().mean(dim=-1)
            point_mask = target.get("point_mask")
            if point_mask is None:
                point_mask = torch.ones_like(point_err, dtype=torch.bool)
            else:
                point_mask = point_mask.to(device=point_err.device, dtype=torch.bool)

            actor_err = point_err[(batch["object_id"] == 0) & point_mask]
            actor_loss = actor_err.mean() if actor_err.numel() else point_err.new_zeros(())
            actor_loss = actor_loss * float(weights.get("actor", weights.get("actor_point", weights.get("point", 1.0))))
            metrics["loss_actor"] = actor_loss
            total = actor_loss if total is None else total + actor_loss

            object_err = point_err[(batch["object_id"] > 0) & point_mask]
            object_loss = object_err.mean() if object_err.numel() else point_err.new_zeros(())
            object_loss = object_loss * float(weights.get("object", weights.get("object_point", weights.get("point", 1.0))))
            metrics["loss_object"] = object_loss
            total = object_loss if total is None else total + object_loss

        if "is_complete" in target:
            frame_outputs = self.decode_frame(
                memory=memory,
                frame_id=batch.get("frame_query_frame_id", batch["frame_id"]),
                head_names="is_complete",
            )
            pred = frame_outputs["is_complete"]
            gt = target["is_complete"].to(dtype=pred.dtype)
            if pred.shape[-1:] == (1,) and gt.shape == pred.shape[:-1]:
                pred = pred.squeeze(-1)
            if gt.shape[-1:] == (1,) and pred.shape == gt.shape[:-1]:
                gt = gt.squeeze(-1)

            complete_loss = F.binary_cross_entropy_with_logits(pred, gt)
            complete_loss = complete_loss * float(weights.get("is_complete", 1.0))
            metrics["loss_is_complete"] = complete_loss
            total = complete_loss if total is None else total + complete_loss

        if total is None:
            raise ValueError("No supported loss targets found in batch[target]")
        metrics["loss_total"] = total
        return total, metrics
