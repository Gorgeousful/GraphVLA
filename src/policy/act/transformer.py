"""DETR transformer used by ACT.

Adapted from the official ACT implementation and DETR.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from src.policy.checkpointing import checkpoint_module


def _with_pos(tensor: Tensor, position: Tensor | None) -> Tensor:
    return tensor if position is None else tensor + position


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        source: Tensor,
        *,
        padding_mask: Tensor | None = None,
        position: Tensor | None = None,
    ) -> Tensor:
        if self.pre_norm:
            normalized = self.norm1(source)
            query = key = _with_pos(normalized, position)
            update = self.attention(
                query, key, normalized, key_padding_mask=padding_mask,
            )[0]
            source = source + self.dropout1(update)
            normalized = self.norm2(source)
            update = self.linear2(self.dropout(F.relu(self.linear1(normalized))))
            return source + self.dropout2(update)

        query = key = _with_pos(source, position)
        update = self.attention(query, key, source, key_padding_mask=padding_mask)[0]
        source = self.norm1(source + self.dropout1(update))
        update = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout2(update))


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, num_layers: int, final_norm: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = nn.LayerNorm(layer.linear2.out_features) if final_norm else None

    def forward(
        self,
        source: Tensor,
        *,
        padding_mask: Tensor | None = None,
        position: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            source = checkpoint_module(layer, source, padding_mask=padding_mask, position=position)
        return self.norm(source) if self.norm is not None else source


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        memory_padding_mask: Tensor | None = None,
        position: Tensor | None = None,
        query_position: Tensor | None = None,
    ) -> Tensor:
        if self.pre_norm:
            normalized = self.norm1(target)
            query = key = _with_pos(normalized, query_position)
            update = self.self_attention(query, key, normalized)[0]
            target = target + self.dropout1(update)
            normalized = self.norm2(target)
            update = self.cross_attention(
                _with_pos(normalized, query_position),
                _with_pos(memory, position),
                memory,
                key_padding_mask=memory_padding_mask,
            )[0]
            target = target + self.dropout2(update)
            normalized = self.norm3(target)
            update = self.linear2(self.dropout(F.relu(self.linear1(normalized))))
            return target + self.dropout3(update)

        query = key = _with_pos(target, query_position)
        update = self.self_attention(query, key, target)[0]
        target = self.norm1(target + self.dropout1(update))
        update = self.cross_attention(
            _with_pos(target, query_position),
            _with_pos(memory, position),
            memory,
            key_padding_mask=memory_padding_mask,
        )[0]
        target = self.norm2(target + self.dropout2(update))
        update = self.linear2(self.dropout(F.relu(self.linear1(target))))
        return self.norm3(target + self.dropout3(update))


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = nn.LayerNorm(layer.linear2.out_features)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        memory_padding_mask: Tensor | None = None,
        position: Tensor | None = None,
        query_position: Tensor | None = None,
    ) -> Tensor:
        outputs = []
        for layer in self.layers:
            target = checkpoint_module(
                layer, target,
                memory,
                memory_padding_mask=memory_padding_mask,
                position=position,
                query_position=query_position,
            )
            outputs.append(self.norm(target))
        return torch.stack(outputs)


class ACTTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        encoder_layer = EncoderLayer(
            hidden_dim, num_heads, feedforward_dim, dropout, pre_norm,
        )
        decoder_layer = DecoderLayer(
            hidden_dim, num_heads, feedforward_dim, dropout, pre_norm,
        )
        self.encoder = Encoder(encoder_layer, encoder_layers, final_norm=pre_norm)
        self.decoder = Decoder(decoder_layer, decoder_layers)
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        source: Tensor,
        position: Tensor,
        query_embed: Tensor,
        latent: Tensor,
        proprio: Tensor,
        language: Tensor,
        additional_position: Tensor,
    ) -> Tensor:
        batch_size = source.shape[0]
        source = source.flatten(2).permute(2, 0, 1)
        position = position.flatten(2).permute(2, 0, 1).repeat(1, batch_size, 1)
        extra_position = additional_position[:, None].repeat(1, batch_size, 1)
        position = torch.cat((extra_position, position), dim=0)
        source = torch.cat((torch.stack((latent, proprio, language)), source), dim=0)
        query_position = query_embed[:, None].repeat(1, batch_size, 1)
        target = torch.zeros_like(query_position)
        memory = self.encoder(source, position=position)
        decoded = self.decoder(
            target,
            memory,
            position=position,
            query_position=query_position,
        )
        return decoded[0].transpose(0, 1)  # Official DETRVAE consumes hs[0].


def make_latent_encoder(
    hidden_dim: int,
    num_heads: int,
    feedforward_dim: int,
    num_layers: int,
    dropout: float,
    pre_norm: bool,
) -> Encoder:
    layer = EncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout, pre_norm)
    return Encoder(layer, num_layers, final_norm=pre_norm)
