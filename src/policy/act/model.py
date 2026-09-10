"""Action Chunking with Transformers policy for LIBERO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from src.policy.checkpointing import GradientCheckpointingMixin

from src.policy.act.backbone import ACTBackbone
from src.policy.language import FrozenBgeClsEncoder
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


class ACTPolicy(GradientCheckpointingMixin, nn.Module):
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
        img_size: int = 224,
        pretrained_backbone: bool = True,
        pretrained_backbone_path: str | Path | None = None,
        language_model_path: str | Path | None = None,
        language_dim: int = 384,
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
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.language_dim = language_dim

        self.backbones = nn.ModuleList(
            ACTBackbone(
                hidden_dim,
                pretrained=pretrained_backbone,
                weights_path=pretrained_backbone_path,
            )
            for _ in range(1)
        )
        object.__setattr__(
            self,
            "_language_encoder",
            FrozenBgeClsEncoder(language_model_path)
            if language_model_path is not None else None,
        )
        self.input_projection = nn.Conv2d(self.backbones[0].num_channels, hidden_dim, 1)
        self.proprio_projection = nn.Linear(state_dim, hidden_dim)
        self.language_projection = nn.Linear(language_dim, hidden_dim)
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
        self.additional_position = nn.Embedding(3, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

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
        state, images, language = self._inputs(batch)
        actions = batch["actions"]
        is_pad = batch["is_pad"].bool()
        if actions.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.chunk_size},{self.action_dim}], got {actions.shape}"
            )
        if is_pad.shape != actions.shape[:2]:
            raise ValueError(f"Expected is_pad {actions.shape[:2]}, got {is_pad.shape}")

        predicted, mean, log_variance = self._predict(
            state, images, language, actions=actions, is_pad=is_pad,
        )
        valid = (~is_pad).unsqueeze(-1)
        loss_l1 = (F.l1_loss(predicted, actions, reduction="none") * valid).mean()
        loss_kl = _kl_divergence(mean, log_variance)
        loss = loss_l1 + self.kl_weight * loss_kl
        return loss, {"loss_l1": loss_l1.detach(), "loss_kl": loss_kl.detach()}

    @torch.inference_mode()
    def predict_action(self, batch: dict[str, Any]) -> Tensor:
        state, images, language = self._inputs(batch)
        predicted, _, _ = self._predict(state, images, language)
        return predicted

    def _inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        state = batch["state"]
        images = batch["images"]
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {state.shape}")
        if images.ndim != 5 or images.shape[1] != self.num_cameras:
            raise ValueError(
                f"Expected images [B,{self.num_cameras},C,H,W], got {images.shape}"
            )
        if images.shape[-2:] != (self.img_size, self.img_size):
            batch_size, cameras = images.shape[:2]
            images = F.interpolate(
                images.flatten(0, 1),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, cameras, 3, self.img_size, self.img_size)
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, None, :, None, None]
        language = self._language_features(batch, state.device)
        if language.shape[0] != state.shape[0]:
            raise ValueError(
                f"Language batch size {language.shape[0]} does not match {state.shape[0]}"
            )
        return state, (images - mean) / std, language

    def _language_features(self, batch: dict[str, Any], device: torch.device) -> Tensor:
        if "language_embedding" in batch:
            features = torch.as_tensor(batch["language_embedding"], device=device).float()
        else:
            texts = batch.get("language")
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, Sequence):
                raise TypeError("ACT requires language strings or language_embedding")
            encoder = self._language_encoder
            if encoder is None:
                raise RuntimeError("language_model_path is required to encode language strings")
            features = encoder.encode(texts, device=device)
        if features.ndim != 2 or features.shape[-1] != self.language_dim:
            raise ValueError(
                f"Expected language embedding [B,{self.language_dim}], got {features.shape}"
            )
        return features

    def _predict(
        self,
        state: Tensor,
        images: Tensor,
        language: Tensor,
        *,
        actions: Tensor | None = None,
        is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
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
            features, position = self.backbones[0](images[:, camera_index])
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
            self.language_projection(language),
            self.additional_position.weight,
        )
        return self.action_head(hidden), mean, log_variance
