"""D4RT-style token query model for GraphVLA."""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import IndependentQueryDecoder
from .encoder import SetEncoderViT
from .encoder import TokenMemoryEncoder
from .heads import PredictionHeads
from .embedding import LearnableTokenTimeEmbedding
from .embedding import TokenQueryEmbedder


class TokenQueryModel(nn.Module):
    """Encode EEF/object token histories and decode token/time queries."""

    def __init__(
        self,
        encoder_hidden_dim: int = 1024, # encoder
        encoder_layers: int = 32,
        encoder_heads: int = 16,
        encoder_mlp_ratio: float = 4.0,
        decoder_hidden_dim: int = 1024, # decoder
        decoder_layers: int = 8,
        decoder_heads: int = 16,
        decoder_mlp_ratio: float = 4.0,
        output_dims: dict[str, int] | None = None,
        num_query_types: int = 0, # extra 2
        max_tokens: int = 64,
        min_time: int = -63,
        max_time: int = 0,
        attention_pattern: str | None = "interleaved_local_global",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_position_embedding = LearnableTokenTimeEmbedding(
            hidden_dim=encoder_hidden_dim,
            max_tokens=max_tokens,
            min_time=min_time,
            max_time=max_time,
        )
        self.query_position_embedding = (
            self.encoder_position_embedding
            if decoder_hidden_dim == encoder_hidden_dim
            else LearnableTokenTimeEmbedding(
                hidden_dim=decoder_hidden_dim,
                max_tokens=max_tokens,
                min_time=min_time,
                max_time=max_time,
            )
        )
        self.encoder = TokenMemoryEncoder(
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
        self.query_embedder = TokenQueryEmbedder(
            hidden_dim=decoder_hidden_dim,
            num_query_types=num_query_types,
            position_embedding=self.query_position_embedding,
        )
        self.decoder = IndependentQueryDecoder(
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            mlp_ratio=decoder_mlp_ratio,
            dropout=dropout,
        )
        if output_dims is None:
            raise ValueError("output_dims must be provided")
        self.heads = PredictionHeads(
            hidden_dim=decoder_hidden_dim,
            output_dims=output_dims,
        )

    def encode(self, tokens: torch.Tensor, extra_tokens: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encoder(tokens=tokens, extra_tokens=extra_tokens)
        return self.memory_proj(memory)

    def embed_queries(
        self,
        token_id: torch.Tensor,
        time: torch.Tensor,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.query_embedder(
            token_id=token_id,
            time=time,
            query_type=query_type,
            extra_query=extra_query,
        )

    def decode(
        self,
        memory: torch.Tensor,
        token_id: torch.Tensor,
        time: torch.Tensor,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        query_tokens = self.embed_queries(
            token_id=token_id,
            time=time,
            query_type=query_type,
            extra_query=extra_query,
        )
        decoded = self.decoder(query_tokens=query_tokens, memory_tokens=memory)
        return self.heads(decoded, head_names=head_names)

    def forward(
        self,
        tokens: torch.Tensor,
        token_id: torch.Tensor,
        time: torch.Tensor,
        extra_tokens: torch.Tensor | None = None,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode(tokens=tokens, extra_tokens=extra_tokens)
        return self.decode(
            memory=memory,
            token_id=token_id,
            time=time,
            query_type=query_type,
            extra_query=extra_query,
            head_names=head_names,
        )


class SetQueryModel(nn.Module):
    """Encode actor/object point sets, then run token/time query decoding.

    ``SetEncoderViT`` converts object points ``[B, T, N, P, F]`` into object
    tokens ``[B, T, N, C]``. A second ``SetEncoderViT`` encodes
    ``actor_feats`` into the token-id-0 actor token before passing the full
    token grid to ``TokenQueryModel``.
    """

    def __init__(
        self,
        point_dim: int = 6,
        num_points: int = 128,
        actor_num_points: int = 3,
        set_hidden_dim: int = 384,
        set_layers: int = 12,
        set_heads: int = 6,
        set_mlp_ratio: float = 4.0,
        set_register_tokens: int = 0,

        encoder_hidden_dim: int = 1024,
        encoder_layers: int = 24,
        encoder_heads: int = 16,
        encoder_mlp_ratio: float = 4.0,

        decoder_hidden_dim: int = 1024,
        decoder_layers: int = 8,
        decoder_heads: int = 16,
        decoder_mlp_ratio: float = 4.0,

        output_dims: dict[str, int] | None = None,

        num_query_types: int = 2,
        max_tokens: int = 3,
        min_time: int = -15,
        max_time: int = 15,
        attention_pattern: str | None = "interleaved_local_global",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        actor_num_points = num_points if actor_num_points is None else actor_num_points
        self.object_encoder = SetEncoderViT(
            point_dim=point_dim,
            hidden_dim=set_hidden_dim,
            num_points=num_points,
            num_heads=set_heads,
            num_layers=set_layers,
            mlp_ratio=set_mlp_ratio,
            num_register_tokens=set_register_tokens,
        )
        self.actor_encoder = SetEncoderViT(
            point_dim=point_dim,
            hidden_dim=set_hidden_dim,
            num_points=actor_num_points,
            num_heads=set_heads,
            num_layers=set_layers,
            mlp_ratio=set_mlp_ratio,
            num_register_tokens=set_register_tokens,
        )
        self.set_proj = (
            nn.Identity()
            if set_hidden_dim == encoder_hidden_dim
            else nn.Linear(set_hidden_dim, encoder_hidden_dim)
        )
        self.token_query_model = TokenQueryModel(
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            encoder_mlp_ratio=encoder_mlp_ratio,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            decoder_mlp_ratio=decoder_mlp_ratio,
            output_dims=output_dims,
            num_query_types=num_query_types,
            max_tokens=max_tokens,
            min_time=min_time,
            max_time=max_time,
            attention_pattern=attention_pattern,
            dropout=dropout,
        )

    def encode_sets(self, point_feats: torch.Tensor, actor_feats: torch.Tensor) -> torch.Tensor:
        object_tokens = self.set_proj(self.object_encoder(point_feats))
        actor_tokens = self.set_proj(self.actor_encoder(actor_feats))
        if actor_tokens.shape[:2] != object_tokens.shape[:2] or actor_tokens.shape[3] != object_tokens.shape[3]:
            raise ValueError(
                "actor/object tokens must match batch/time/hidden dims: "
                f"actor={actor_tokens.shape}, object={object_tokens.shape}"
            )
        return torch.cat([actor_tokens, object_tokens], dim=2)

    def encode(
        self,
        point_feats: torch.Tensor,
        actor_feats: torch.Tensor,
        extra_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.encode_sets(point_feats=point_feats, actor_feats=actor_feats)
        return self.token_query_model.encode(tokens=tokens, extra_tokens=extra_tokens)

    def decode(
        self,
        memory: torch.Tensor,
        token_id: torch.Tensor,
        time: torch.Tensor,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.token_query_model.decode(
            memory=memory,
            token_id=token_id,
            time=time,
            query_type=query_type,
            extra_query=extra_query,
            head_names=head_names,
        )

    def forward(
        self,
        point_feats: torch.Tensor,
        actor_feats: torch.Tensor,
        token_id: torch.Tensor,
        time: torch.Tensor,
        extra_tokens: torch.Tensor | None = None,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
        head_names: str | list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.encode_sets(point_feats=point_feats, actor_feats=actor_feats)
        return self.token_query_model(
            tokens=tokens,
            token_id=token_id,
            time=time,
            extra_tokens=extra_tokens,
            query_type=query_type,
            extra_query=extra_query,
            head_names=head_names,
        )
