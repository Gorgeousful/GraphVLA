"""Embeddings for encoder token grids and token/time queries."""

from __future__ import annotations

import math
from typing import Literal

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


#: =====================================================
class LearnableFrameObjectPointEmbedding(nn.Module):
    """Learnable embeddings indexed by frame id, object id, and point id."""

    def __init__(
        self,
        hidden_dim: int,
        max_objects: int,
        max_actor_points: int,
        max_object_points: int,
        min_frame: int,
        max_frame: int,
    ) -> None:
        super().__init__()
        if max_objects <= 0:
            raise ValueError(f"max_objects must be positive, got {max_objects}")
        if max_actor_points <= 0 or max_object_points <= 0:
            raise ValueError(
                "max_actor_points and max_object_points must be positive, "
                f"got {max_actor_points} and {max_object_points}"
            )
        if min_frame > max_frame:
            raise ValueError(f"min_frame must be <= max_frame, got {min_frame} > {max_frame}")
        self.hidden_dim = hidden_dim
        self.max_objects = int(max_objects)
        self.max_actor_points = int(max_actor_points)
        self.max_object_points = int(max_object_points)
        self.min_frame = int(min_frame)
        self.max_frame = int(max_frame)
        self.num_frame_embeddings = self.max_frame - self.min_frame + 1
        self.frame_embed = nn.Embedding(self.num_frame_embeddings, hidden_dim)
        self.object_embed = nn.Embedding(self.max_objects, hidden_dim)
        self.actor_point_embed = nn.Embedding(self.max_actor_points, hidden_dim)
        self.object_point_embed = nn.Embedding(self.max_object_points, hidden_dim)

    def _point_embedding(self, point_type: Literal["actor", "object"]) -> tuple[nn.Embedding, int]:
        if point_type == "actor":
            return self.actor_point_embed, self.max_actor_points
        if point_type == "object":
            return self.object_point_embed, self.max_object_points
        raise ValueError(f"Unsupported point_type: {point_type!r}")

    def _check_object_id(self, object_id: torch.Tensor) -> torch.Tensor:
        object_id = object_id.long()
        if object_id.numel() > 0:
            min_id = int(object_id.min().item())
            max_id = int(object_id.max().item())
            if min_id < 0 or max_id >= self.max_objects:
                raise ValueError(
                    f"object_id must be in [0, {self.max_objects - 1}], got range [{min_id}, {max_id}]"
                )
        return object_id

    def _check_point_id(
        self,
        point_id: torch.Tensor,
        point_type: Literal["actor", "object"],
    ) -> torch.Tensor:
        point_id = point_id.long()
        if point_id.numel() > 0:
            min_id = int(point_id.min().item())
            max_id = int(point_id.max().item())
            _, max_points = self._point_embedding(point_type)
            if min_id < 0 or max_id >= max_points:
                raise ValueError(
                    f"{point_type} point_id must be in [0, {max_points - 1}], "
                    f"got range [{min_id}, {max_id}]"
                )
        return point_id

    def _frame_to_index(self, frame_id: torch.Tensor) -> torch.Tensor:
        if frame_id.is_floating_point() and not torch.all(frame_id == frame_id.round()):
            raise ValueError("learnable frame embedding expects integer relative frame ids")
        frame_id = frame_id.long()
        if frame_id.numel() > 0:
            min_f = int(frame_id.min().item())
            max_f = int(frame_id.max().item())
            if min_f < self.min_frame or max_f > self.max_frame:
                raise ValueError(
                    f"frame_id must be in [{self.min_frame}, {self.max_frame}], got range [{min_f}, {max_f}]"
                )
        return frame_id - self.min_frame

    def encode_grid(
        self,
        num_frames: int,
        num_objects: int,
        num_points: int,
        device: torch.device,
        *,
        object_offset: int,
        point_type: Literal["actor", "object"],
    ) -> torch.Tensor:
        frame_id = torch.arange(-num_frames + 1, 1, device=device)
        object_id = torch.arange(object_offset, object_offset + num_objects, device=device)
        point_id = torch.arange(num_points, device=device)
        point_embed, _ = self._point_embedding(point_type)
        if point_type == "actor" and (object_offset != 0 or num_objects != 1):
            raise ValueError("actor point grid requires exactly object_id 0")
        if point_type == "object" and object_offset <= 0:
            raise ValueError("object point grid requires positive object ids")
        return (
            self.frame_embed(self._frame_to_index(frame_id))[:, None, None, :]
            + self.object_embed(self._check_object_id(object_id))[None, :, None, :]
            + point_embed(self._check_point_id(point_id, point_type))[None, None, :, :]
        )

    def encode_query(self, object_id: torch.Tensor, point_id: torch.Tensor, frame_id: torch.Tensor) -> torch.Tensor:
        if object_id.shape != point_id.shape or object_id.shape != frame_id.shape:
            raise ValueError(
                "object_id, point_id, and frame_id must have the same shape, "
                f"got {object_id.shape}, {point_id.shape}, {frame_id.shape}"
            )
        object_id = self._check_object_id(object_id)
        if object_id.numel() > 0 and torch.any(object_id != 0).item():
            raise ValueError("point query only supports actor object_id 0")
        frame_id = frame_id.long()
        if frame_id.numel() > 0 and torch.any(frame_id <= 0).item():
            raise ValueError("point query only supports future frames")
        point_id = self._check_point_id(point_id, "actor")
        return (
            self.object_embed(object_id)
            + self.actor_point_embed(point_id)
            + self.frame_embed(self._frame_to_index(frame_id))
        )

    def encode_frame_query(self, frame_id: torch.Tensor) -> torch.Tensor:
        return self.frame_embed(self._frame_to_index(frame_id))

    def encode_object_query(self, object_id: torch.Tensor, frame_id: torch.Tensor) -> torch.Tensor:
        if object_id.shape != frame_id.shape:
            raise ValueError(
                "object_id and frame_id must have the same shape, "
                f"got {object_id.shape} and {frame_id.shape}"
            )
        return (
            self.object_embed(self._check_object_id(object_id))
            + self.frame_embed(self._frame_to_index(frame_id))
        )


class FrameQueryEmbedder(nn.Module):
    """Build query tokens from learnable relative frame embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        position_embedding: LearnableFrameObjectPointEmbedding,
        num_query_types: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_embedding = position_embedding
        self.query_type_embed = nn.Embedding(num_query_types, hidden_dim) if num_query_types > 0 else None
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        frame_id: torch.Tensor,
        query_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_id.ndim != 2:
            raise ValueError(f"Expected frame_id [B, Q], got {frame_id.shape}")
        token = self.position_embedding.encode_frame_query(frame_id)

        if query_type is not None:
            if self.query_type_embed is None:
                raise ValueError("query_type was provided but num_query_types is 0")
            if query_type.shape != frame_id.shape:
                raise ValueError(f"query_type must match frame_id shape, got {query_type.shape} vs {frame_id.shape}")
            token = token + self.query_type_embed(query_type.long())

        return self.out_norm(token)


class ObjectQueryEmbedder(nn.Module):
    """Build query tokens from object id and relative frame id."""

    def __init__(
        self,
        hidden_dim: int,
        position_embedding: LearnableFrameObjectPointEmbedding,
        num_query_types: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_embedding = position_embedding
        self.query_type_embed = nn.Embedding(num_query_types, hidden_dim) if num_query_types > 0 else None
        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(
        self,
        object_id: torch.Tensor,
        frame_id: torch.Tensor,
        query_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if object_id.ndim != 2 or frame_id.ndim != 2:
            raise ValueError(f"Expected object_id/frame_id [B, Q], got {object_id.shape} and {frame_id.shape}")
        token = self.position_embedding.encode_object_query(object_id=object_id, frame_id=frame_id)
        if query_type is not None:
            if self.query_type_embed is None:
                raise ValueError("query_type was provided but num_query_types is 0")
            if query_type.shape != frame_id.shape:
                raise ValueError(f"query_type must match frame_id shape, got {query_type.shape} vs {frame_id.shape}")
            token = token + self.query_type_embed(query_type.long())
        return self.out_norm(token)


class PointQueryEmbedder(nn.Module):
    """Build query tokens from object id, point id, and frame id."""

    def __init__(
        self,
        hidden_dim: int,
        position_embedding: LearnableFrameObjectPointEmbedding,
        num_query_types: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_embedding = position_embedding
        self.query_type_embed = nn.Embedding(num_query_types, hidden_dim) if num_query_types > 0 else None
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        object_id: torch.Tensor,
        point_id: torch.Tensor,
        frame_id: torch.Tensor,
        query_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if object_id.ndim != 2 or point_id.ndim != 2 or frame_id.ndim != 2:
            raise ValueError(
                f"Expected object_id/point_id/frame_id [B, Q], got {object_id.shape}, {point_id.shape}, {frame_id.shape}"
            )
        token = self.position_embedding.encode_query(object_id=object_id, point_id=point_id, frame_id=frame_id)

        if query_type is not None:
            if self.query_type_embed is None:
                raise ValueError("query_type was provided but num_query_types is 0")
            if query_type.shape != object_id.shape:
                raise ValueError(f"query_type must match object_id shape, got {query_type.shape} vs {object_id.shape}")
            token = token + self.query_type_embed(query_type.long())

        return self.out_norm(token)