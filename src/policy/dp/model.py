"""Official image U-Net Diffusion Policy adapted to LIBERO with BGE CLS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from src.policy.checkpointing import GradientCheckpointingMixin, checkpoint_module
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

from src.policy.dp.blocks import ConditionalUnet1d
from src.policy.language import FrozenBgeClsEncoder


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _replace_batch_norm(child)


class CameraEncoder(nn.Module):
    def __init__(self, crop_size: int, feature_dim: int, weights_path: str | Path | None) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        if weights_path is not None:
            backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        _replace_batch_norm(backbone)
        backbone.fc = nn.Identity() if feature_dim == 512 else nn.Linear(512, feature_dim)
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        num_cameras: int,
        img_size: int,
        crop_size: int,
        feature_dim: int,
        weights_path: str | Path | None,
    ) -> None:
        super().__init__()
        if not 0 < crop_size <= img_size:
            raise ValueError(f"crop_size must be in [1, img_size], got {crop_size} and {img_size}")
        self.img_size = img_size
        self.crop_size = crop_size
        self.encoders = nn.ModuleList(
            CameraEncoder(crop_size, feature_dim, weights_path) for _ in range(num_cameras)
        )

    def _prepare(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False,
        )
        if self.crop_size < self.img_size:
            if self.training:
                crops = []
                limit = self.img_size - self.crop_size + 1
                for image in images:
                    top = int(torch.randint(limit, (), device=image.device))
                    left = int(torch.randint(limit, (), device=image.device))
                    crops.append(image[:, top : top + self.crop_size, left : left + self.crop_size])
                images = torch.stack(crops)
            else:
                images = TF.center_crop(images, [self.crop_size, self.crop_size])
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (images - mean) / std

    def forward(self, images: Tensor) -> Tensor:
        features = [
            checkpoint_module(encoder, self._prepare(images[:, camera_index]))
            for camera_index, encoder in enumerate(self.encoders)
        ]
        return torch.cat(features, dim=-1)


class DiffusionPolicy(GradientCheckpointingMixin, nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        num_cameras: int = 2,
        img_size: int = 224,
        crop_size: int = 224,
        visual_feature_dim: int = 512,
        pretrained_backbone_path: str | Path | None = None,
        language_model_path: str | Path | None = None,
        language_dim: int = 384,
        obs_steps: int = 2,
        horizon: int = 16,
        action_steps: int = 10,
        diffusion_steps: int = 100,
        inference_steps: int = 100,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        num_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        if action_steps + obs_steps - 1 > horizon:
            raise ValueError("action_steps + obs_steps - 1 must not exceed horizon")
        if horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must be divisible by the U-Net downsampling factor")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_cameras = num_cameras
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.action_steps = action_steps
        self.inference_steps = inference_steps
        self.language_dim = language_dim
        self.vision_encoder = VisualEncoder(
            num_cameras,
            img_size,
            crop_size,
            visual_feature_dim,
            pretrained_backbone_path,
        )
        object.__setattr__(
            self,
            "_language_encoder",
            FrozenBgeClsEncoder(language_model_path) if language_model_path is not None else None,
        )
        observation_dim = num_cameras * visual_feature_dim + state_dim
        self.noise_predictor = ConditionalUnet1d(
            action_dim,
            observation_dim * obs_steps + language_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            groups=num_groups,
            predict_scale=cond_predict_scale,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            variance_type="fixed_small",
            prediction_type="epsilon",
        )

    def _language_features(self, batch: dict[str, Any], device: torch.device) -> Tensor:
        if "language_embedding" in batch:
            features = torch.as_tensor(batch["language_embedding"], device=device).float()
        else:
            texts = batch.get("language")
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, Sequence):
                raise TypeError("Diffusion Policy requires language strings or language_embedding")
            encoder = self._language_encoder
            if encoder is None:
                raise RuntimeError("language_model_path is required to encode language strings")
            features = encoder.encode(texts, device=device)
        if features.ndim != 2 or features.shape[-1] != self.language_dim:
            raise ValueError(f"Expected language embedding [B,{self.language_dim}], got {features.shape}")
        return features

    def _condition(self, batch: dict[str, Any]) -> Tensor:
        state = batch["state"]
        images = batch["images"]
        expected_images = (self.obs_steps, self.num_cameras)
        if state.ndim != 3 or state.shape[1:] != (self.obs_steps, self.state_dim):
            raise ValueError(f"Expected state [B,{self.obs_steps},{self.state_dim}], got {state.shape}")
        if images.ndim != 6 or images.shape[1:3] != expected_images:
            raise ValueError(f"Expected images [B,{self.obs_steps},{self.num_cameras},C,H,W], got {images.shape}")
        batch_size = state.shape[0]
        image_features = self.vision_encoder(
            images.flatten(0, 1).reshape(batch_size * self.obs_steps, self.num_cameras, *images.shape[3:])
        ).reshape(batch_size, self.obs_steps, -1)
        language = self._language_features(batch, state.device)
        if language.shape[0] != batch_size:
            raise ValueError(f"Language batch size {language.shape[0]} does not match {batch_size}")
        return torch.cat((torch.cat((image_features, state), dim=-1).flatten(1), language), dim=-1)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        actions = batch["actions"]
        is_pad = batch["is_pad"].bool()
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"Expected actions [B,{self.horizon},{self.action_dim}], got {actions.shape}")
        if is_pad.shape != actions.shape[:2]:
            raise ValueError(f"Expected is_pad {actions.shape[:2]}, got {is_pad.shape}")
        condition = self._condition(batch)
        batch_size = actions.shape[0]
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=actions.device,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.noise_predictor(noisy_actions, timesteps, condition)
        valid = (~is_pad).unsqueeze(-1)
        squared_error = (prediction - noise).square() * valid
        loss = squared_error.sum() / (valid.sum().clamp_min(1) * self.action_dim)
        return loss, {"loss_mse": loss.detach()}

    @torch.inference_mode()
    def predict_action(self, batch: dict[str, Any]) -> Tensor:
        condition = self._condition(batch)
        sample = torch.randn(
            (condition.shape[0], self.horizon, self.action_dim),
            device=condition.device,
            dtype=condition.dtype,
        )
        self.noise_scheduler.set_timesteps(self.inference_steps, device=sample.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.noise_predictor(sample, timestep, condition)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        start = self.obs_steps - 1
        return sample[:, start : start + self.action_steps]
