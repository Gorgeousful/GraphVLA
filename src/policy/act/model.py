"""Action Chunking with Transformers policy for LIBERO."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.policy.act.backbone import ACTBackbone
from src.policy.act.transformer import ACTTransformer, make_latent_encoder


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    dimensions = torch.arange(width, dtype=torch.float32)[None]
    table = positions / (10_000 ** (2 * torch.div(dimensions, 2, rounding_mode="floor") / width))
    table[:, 0::2] = table[:, 0::2].sin()
    table[:, 1::2] = table[:, 1::2].cos()
    return table[:, None]


def _kl_divergence(mean: Tensor, log_variance: Tensor) -> Tensor:
    return (-0.5 * (1 + log_variance - mean.square() - log_variance.exp())).sum(1).mean()


class ACTPolicy(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        chunk_size: int = 10,
        hidden_dim: int = 512,
        feedforward_dim: int = 3200,
        encoder_layers: int = 4,
        decoder_layers: int = 7,
        num_heads: int = 8,
        dropout: float = 0.1,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        num_cameras: int = 2,
        pretrained_backbone: bool = True,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError(f"hidden_dim must be even, got {hidden_dim}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.kl_weight = kl_weight
        self.num_cameras = num_cameras
        self.latent_dim = latent_dim

        self.backbone = ACTBackbone(hidden_dim, pretrained=pretrained_backbone)
        self.input_projection = nn.Conv2d(self.backbone.num_channels, hidden_dim, 1)
        self.proprio_projection = nn.Linear(state_dim, hidden_dim)
        self.transformer = ACTTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
            pre_norm=pre_norm,
        )
        self.query_embedding = nn.Embedding(chunk_size, hidden_dim)
        self.additional_position = nn.Embedding(2, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.padding_head = nn.Linear(hidden_dim, 1)

        self.latent_encoder = make_latent_encoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            pre_norm=pre_norm,
        )
        self.encoder_cls = nn.Embedding(1, hidden_dim)
        self.encoder_action_projection = nn.Linear(action_dim, hidden_dim)
        self.encoder_state_projection = nn.Linear(state_dim, hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_output_projection = nn.Linear(latent_dim, hidden_dim)
        self.register_buffer(
            "encoder_position",
            _sinusoidal_positions(chunk_size + 2, hidden_dim),
            persistent=True,
        )

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        state, images = self._inputs(batch)
        actions = batch["actions"]
        is_pad = batch["is_pad"].bool()
        if actions.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.chunk_size},{self.action_dim}], got {actions.shape}"
            )
        if is_pad.shape != actions.shape[:2]:
            raise ValueError(f"Expected is_pad {actions.shape[:2]}, got {is_pad.shape}")

        predicted, _, mean, log_variance = self._predict(
            state, images, actions=actions, is_pad=is_pad,
        )
        valid = (~is_pad).unsqueeze(-1)
        loss_l1 = (F.l1_loss(predicted, actions, reduction="none") * valid).mean()
        loss_kl = _kl_divergence(mean, log_variance)
        loss = loss_l1 + self.kl_weight * loss_kl
        return loss, {"loss_l1": loss_l1.detach(), "loss_kl": loss_kl.detach()}

    @torch.inference_mode()
    def predict_action(self, batch: dict[str, Any]) -> Tensor:
        state, images = self._inputs(batch)
        predicted, _, _, _ = self._predict(state, images)
        return predicted

    def _inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        state = batch["state"]
        images = batch["images"]
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape}")
        if images.ndim != 5 or images.shape[1] != self.num_cameras:
            raise ValueError(
                f"Expected images [B,{self.num_cameras},C,H,W], got {images.shape}"
            )
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, None, :, None, None]
        return state, (images - mean) / std

    def _predict(
        self,
        state: Tensor,
        images: Tensor,
        *,
        actions: Tensor | None = None,
        is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        batch_size = state.shape[0]
        if actions is None:
            mean = log_variance = None
            latent = state.new_zeros((batch_size, self.latent_dim))
        else:
            cls = self.encoder_cls.weight[None].repeat(batch_size, 1, 1)
            encoder_input = torch.cat(
                (
                    cls,
                    self.encoder_state_projection(state)[:, None],
                    self.encoder_action_projection(actions),
                ),
                dim=1,
            ).permute(1, 0, 2)
            prefix_pad = torch.zeros((batch_size, 2), dtype=torch.bool, device=state.device)
            padding_mask = torch.cat((prefix_pad, is_pad), dim=1)
            position = self.encoder_position.repeat(1, batch_size, 1)
            encoded = self.latent_encoder(
                encoder_input, padding_mask=padding_mask, position=position,
            )[0]
            latent_parameters = self.latent_projection(encoded)
            mean, log_variance = latent_parameters.chunk(2, dim=-1)
            latent = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)

        camera_features = []
        camera_positions = []
        for camera_index in range(self.num_cameras):
            features, position = self.backbone(images[:, camera_index])
            camera_features.append(self.input_projection(features))
            camera_positions.append(position)
        source = torch.cat(camera_features, dim=3)
        position = torch.cat(camera_positions, dim=3)
        hidden = self.transformer(
            source,
            position,
            self.query_embedding.weight,
            self.latent_output_projection(latent),
            self.proprio_projection(state),
            self.additional_position.weight,
        )
        return self.action_head(hidden), self.padding_head(hidden), mean, log_variance
