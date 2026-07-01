"""D4RT-style token query model for GraphVLA."""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import IndependentQueryDecoder
from .encoder import TokenMemoryEncoder
from .heads import PredictionHeads
from .embedding import RelativeTokenPositionEmbedding
from .embedding import TokenQueryEmbedder


class GraphVLATokenQueryModel(nn.Module):
    """Encode EEF/object token histories and decode token/time queries."""

    def __init__(
        self,
        encoder_hidden_dim: int = 1024,
        decoder_hidden_dim: int = 1024,
        encoder_layers: int = 8,
        encoder_heads: int = 8,
        encoder_mlp_ratio: float = 4.0,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        output_dims: dict[str, int] | None = None,
        num_query_types: int = 0,
        attention_pattern: str | None = "interleaved_local_global",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_position_embedding = RelativeTokenPositionEmbedding(hidden_dim=encoder_hidden_dim)
        self.query_position_embedding = RelativeTokenPositionEmbedding(hidden_dim=decoder_hidden_dim)
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
        self.heads = PredictionHeads(
            hidden_dim=decoder_hidden_dim,
            output_dims=output_dims or {"prediction": decoder_hidden_dim},
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
    ) -> dict[str, torch.Tensor]:
        query_tokens = self.embed_queries(
            token_id=token_id,
            time=time,
            query_type=query_type,
            extra_query=extra_query,
        )
        decoded = self.decoder(query_tokens=query_tokens, memory_tokens=memory)
        return self.heads(decoded)

    def forward(
        self,
        tokens: torch.Tensor,
        token_id: torch.Tensor,
        time: torch.Tensor,
        extra_tokens: torch.Tensor | None = None,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode(tokens=tokens, extra_tokens=extra_tokens)
        return self.decode(
            memory=memory,
            token_id=token_id,
            time=time,
            query_type=query_type,
            extra_query=extra_query,
        )
