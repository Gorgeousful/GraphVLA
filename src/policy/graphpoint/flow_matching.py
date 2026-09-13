"""Rectified-flow utilities for GraphPoint."""

from __future__ import annotations

import torch


def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    concentration1 = torch.full((batch_size,), 1.5, device=device)
    concentration0 = torch.ones(batch_size, device=device)
    time = torch.distributions.Beta(concentration1, concentration0).sample()
    return time * 0.999 + 0.001


def training_path(
    target: torch.Tensor,
    time: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(target)
    expanded = time.view(time.shape[0], *([1] * (target.ndim - 1)))
    return expanded * noise + (1.0 - expanded) * target, noise - target, noise


def make_scheduler(num_steps: int, device: torch.device):
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(num_steps, device=device)
    return scheduler
