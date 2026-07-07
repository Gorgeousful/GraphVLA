"""Embeddings for encoder token grids and token/time queries."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_scalar_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return fixed sin/cos embeddings for arbitrary scalar values."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal_scalar_embedding requires even dim")
    values = values.to(dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=values.device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    angles = values.unsqueeze(-1) * frequencies
    emb = torch.empty(*values.shape, dim, device=values.device, dtype=torch.float32)
    emb[..., 0::2] = torch.sin(angles)
    emb[..., 1::2] = torch.cos(angles)
    return emb



class LearnableTokenTimeEmbedding(nn.Module):
    """Learnable embeddings indexed by token id and relative time.

    Relative times use the external convention ``..., -2, -1, 0, 1, 2, ...``.
    Internally they are shifted to non-negative embedding indices by
    ``time_index = time - min_time``. Encoder memory is assumed to be a
    contiguous history-to-now window ``[-T + 1, ..., -1, 0]``.
    """

    def __init__(self, hidden_dim: int, max_tokens: int, min_time: int, max_time: int) -> None:
        super().__init__()
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if min_time > max_time:
            raise ValueError(f"min_time must be <= max_time, got {min_time} > {max_time}")
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.min_time = int(min_time)
        self.max_time = int(max_time)
        self.num_time_embeddings = self.max_time - self.min_time + 1
        self.token_embed = nn.Embedding(max_tokens, hidden_dim)
        self.time_embed = nn.Embedding(self.num_time_embeddings, hidden_dim)

    def _check_token_id(self, token_id: torch.Tensor) -> torch.Tensor:
        token_id = token_id.long()
        if token_id.numel() > 0:
            min_id = int(token_id.min().item())
            max_id = int(token_id.max().item())
            if min_id < 0 or max_id >= self.max_tokens:
                raise ValueError(
                    f"token id must be in [0, {self.max_tokens - 1}], got range [{min_id}, {max_id}]"
                )
        return token_id

    def _time_to_index(self, time: torch.Tensor) -> torch.Tensor:
        if time.is_floating_point() and not torch.all(time == time.round()):
            raise ValueError("learnable time embedding expects integer relative times")
        time = time.long()
        if time.numel() > 0:
            min_t = int(time.min().item())
            max_t = int(time.max().item())
            if min_t < self.min_time or max_t > self.max_time:
                raise ValueError(
                    f"relative time must be in [{self.min_time}, {self.max_time}], got range [{min_t}, {max_t}]"
                )
        return time - self.min_time

    def encode_grid(self, num_steps: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        if num_tokens > self.max_tokens:
            raise ValueError(f"Expected at most {self.max_tokens} tokens, got {num_tokens}")
        rel_time = torch.arange(-num_steps + 1, 1, device=device)
        time_id = self._time_to_index(rel_time)
        token_id = torch.arange(num_tokens, device=device)
        return self.time_embed(time_id)[:, None, :] + self.token_embed(token_id)[None, :, :]

    def encode_query(self, token_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if token_id.shape != time.shape:
            raise ValueError(f"token_id and time must have the same shape, got {token_id.shape} vs {time.shape}")
        return self.token_embed(self._check_token_id(token_id)) + self.time_embed(self._time_to_index(time))


class RelativeTokenPositionEmbedding(nn.Module):
    """Fixed sinusoidal embeddings indexed by relative time and token id.

    The encoder time order is semantic, not window-dependent: ``0, -1, -2, ...``.
    Query times use the same real relative-time values, so ``t=0`` receives the
    same encoding no matter how many history frames are provided.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def _check_node_id(self, token_id: torch.Tensor) -> torch.Tensor:
        if token_id.numel() > 0:
            min_id = int(token_id.min().item())
            if min_id < 0:
                raise ValueError(f"token id must be non-negative, got minimum {min_id}")
        return token_id.long()

    def encode_grid(self, num_steps: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Return grid embeddings with shape ``[T, P, C]``.

        Encoder time order is assumed to be ``0, -1, -2, ...``.
        """
        rel_time = -torch.arange(num_steps, device=device)
        token_id = torch.arange(num_tokens, device=device)
        time_emb = sinusoidal_scalar_embedding(rel_time, self.hidden_dim)
        node_emb = sinusoidal_scalar_embedding(self._check_node_id(token_id), self.hidden_dim)
        return time_emb[:, None, :] + node_emb[None, :, :]

    def encode_query(self, token_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Return query embeddings with shape ``[B, Q, C]``."""
        if token_id.shape != time.shape:
            raise ValueError(f"token_id and time must have the same shape, got {token_id.shape} vs {time.shape}")
        node_emb = sinusoidal_scalar_embedding(self._check_node_id(token_id), self.hidden_dim)
        time_emb = sinusoidal_scalar_embedding(time, self.hidden_dim)
        return node_emb + time_emb


class TokenQueryEmbedder(nn.Module):
    """Build query tokens from token id and relative time.

    The query asks for a token slot ``token_id`` at relative time ``time``. The
    same ``RelativeTokenPositionEmbedding`` module can be shared with the encoder
    so memory and query positions use identical node/time encoding rules.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_query_types: int = 0,
        position_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_embedding = position_embedding or RelativeTokenPositionEmbedding(hidden_dim=hidden_dim)
        self.query_type_embed = nn.Embedding(num_query_types, hidden_dim) if num_query_types > 0 else None
        self.extra_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        token_id: torch.Tensor,
        time: torch.Tensor,
        query_type: torch.Tensor | None = None,
        extra_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_id.ndim != 2 or time.ndim != 2:
            raise ValueError(f"Expected token_id/time [B, Q], got {token_id.shape} and {time.shape}")
        token = self.position_embedding.encode_query(token_id=token_id, time=time)

        if query_type is not None:
            if self.query_type_embed is None:
                raise ValueError("query_type was provided but num_query_types is 0")
            if query_type.shape != token_id.shape:
                raise ValueError(f"query_type must match token_id shape, got {query_type.shape} vs {token_id.shape}")
            token = token + self.query_type_embed(query_type.long())

        if extra_query is not None:
            if extra_query.shape != token.shape:
                raise ValueError(f"extra_query must have shape {token.shape}, got {extra_query.shape}")
            token = token + self.extra_proj(extra_query)

        return self.out_norm(token)
