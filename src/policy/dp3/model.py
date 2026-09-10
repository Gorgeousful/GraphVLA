"""DP3's XYZ PointNet and x0 diffusion objective, with language conditioning.

Architecture reference: YanjieZe/3D-Diffusion-Policy (MIT), config/dp3.yaml.
The shared U-Net implements its global FiLM path with conditioning at all levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import Tensor, nn
from src.policy.checkpointing import GradientCheckpointingMixin, checkpoint_module

from src.policy.dp.blocks import ConditionalUnet1d
from src.policy.language import FrozenBgeClsEncoder


class DP3Policy(GradientCheckpointingMixin, nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        num_points: int = 512,
        obs_steps: int = 2,
        horizon: int = 16,
        action_steps: int = 10,
        encoder_output_dim: int = 64,
        language_model_path: str | Path | None = None,
        language_dim: int = 384,
        language_projection_dim: int = 64,
        diffusion_steps: int = 100,
        inference_steps: int = 10,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if obs_steps < 1:
            raise ValueError("obs_steps must be positive")
        if not down_dims or horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must be divisible by the U-Net downsampling factor")
        if not 1 <= action_steps <= horizon - obs_steps + 1:
            raise ValueError("action_steps must be in [1, horizon]")
        if not 1 <= inference_steps <= diffusion_steps:
            raise ValueError("inference_steps must be in [1, diffusion_steps]")
        self.obs_steps = obs_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_points = num_points
        self.horizon = horizon
        self.action_steps = action_steps
        self.language_dim = language_dim
        self.inference_steps = inference_steps
        # Official XYZ encoder: 3 -> 64 -> 128 -> 256, max pooling, projection + LN.
        point_layers = []
        for in_dim, out_dim in ((3, 64), (64, 128), (128, 256)):
            point_layers.extend((nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()))
        self.nets = nn.ModuleDict({
            "point_mlp": nn.Sequential(*point_layers),
            "point_projection": nn.Sequential(nn.Linear(256, encoder_output_dim), nn.LayerNorm(encoder_output_dim)),
            "state_encoder": nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64)),
            "language_projection": nn.Linear(language_dim, language_projection_dim),
            "unet": ConditionalUnet1d(
                action_dim,
                (encoder_output_dim + 64) * obs_steps + language_projection_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                groups=num_groups,
                predict_scale=True,
            ),
        })
        object.__setattr__(self, "_language_encoder", (
            FrozenBgeClsEncoder(language_model_path) if language_model_path is not None else None
        ))
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample",
        )

    def _condition(self, batch: dict[str, Any], nets: nn.ModuleDict) -> Tensor:
        points, state = batch["point_cloud"], batch["state"]
        if points.ndim != 4 or points.shape[1:] != (self.obs_steps, self.num_points, 3):
            raise ValueError(f"Expected point_cloud [B,{self.obs_steps},{self.num_points},3], got {points.shape}")
        if state.shape != (points.shape[0], self.obs_steps, self.state_dim):
            raise ValueError(f"Expected state [B,{self.obs_steps},{self.state_dim}], got {state.shape}")
        if "language_embedding" in batch:
            language = torch.as_tensor(batch["language_embedding"], device=state.device).float()
        else:
            texts = batch.get("language")
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
                raise TypeError("DP3 requires language strings or language_embedding")
            if self._language_encoder is None:
                raise RuntimeError("language_model_path is required for language strings")
            # The shared frozen encoder returns inference tensors. Clone before a trainable Linear.
            language = self._language_encoder.encode(texts, device=state.device).clone()
        if language.shape != (state.shape[0], self.language_dim):
            raise ValueError(f"Expected language [B,{self.language_dim}], got {language.shape}")
        point_features = nets["point_projection"](checkpoint_module(nets["point_mlp"], points).amax(dim=2))
        observation = torch.cat((point_features, nets["state_encoder"](state)), -1).flatten(1)
        return torch.cat((observation, nets["language_projection"](language)), -1)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        condition = self._condition(batch, self.nets)
        actions, is_pad = batch["actions"], batch["is_pad"].bool()
        if actions.shape != (condition.shape[0], self.horizon, self.action_dim):
            raise ValueError(f"Expected actions [B,{self.horizon},{self.action_dim}], got {actions.shape}")
        if is_pad.shape != actions.shape[:2]:
            raise ValueError("is_pad must have shape [B,horizon]")
        timesteps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=actions.device)
        noisy_actions = self.noise_scheduler.add_noise(actions, torch.randn_like(actions), timesteps)
        prediction = self.nets["unet"](noisy_actions, timesteps, condition)
        valid = (~is_pad)[..., None]
        loss = ((prediction - actions).square() * valid).sum() / (valid.sum().clamp_min(1) * self.action_dim)
        return loss, {"loss_mse": loss.detach()}

    @torch.inference_mode()
    def predict_action(self, batch: dict[str, Any]) -> Tensor:
        condition = self._condition(batch, self.nets)
        sample = torch.randn((condition.shape[0], self.horizon, self.action_dim), device=condition.device, dtype=condition.dtype)
        self.noise_scheduler.set_timesteps(self.inference_steps, device=sample.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.nets["unet"](sample, timestep, condition)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        # The predicted horizon starts at the oldest observation, as in official DP3.
        start = self.obs_steps - 1
        return sample[:, start:start + self.action_steps]
