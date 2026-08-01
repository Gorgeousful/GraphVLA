"""OpenWAM-style discrete rectified-flow schedules."""

from __future__ import annotations

import torch


class FlowMatchScheduler:
    """Shifted sigma grid shared by training and Euler inference."""

    def __init__(self, num_train_timesteps: int, shift: float) -> None:
        if num_train_timesteps <= 0 or shift <= 0:
            raise ValueError("Flow scheduler timesteps and shift must be positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.training_sigmas = self._shift(
            torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        )
        x = self.training_sigmas * self.num_train_timesteps
        weight = torch.exp(-2.0 * ((x - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2)
        weight = weight - weight.min()
        self.training_weights = weight * (self.num_train_timesteps / weight.sum())
        self.sigmas = torch.empty(0)

    def _shift(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)

    def sample_training(
        self,
        target: torch.Tensor,
        *,
        horizon_dim: int = 1,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, horizon = target.shape[0], target.shape[horizon_dim]
        timestep_ids = torch.randint(
            self.num_train_timesteps, (batch, horizon), device=target.device,
        )
        sigma = self.training_sigmas.to(target.device)[timestep_ids].to(target.dtype)
        if noise is None:
            noise = torch.randn_like(target)
        shape = [batch] + [1] * (target.ndim - 1)
        shape[horizon_dim] = horizon
        expanded_sigma = sigma.view(shape)
        state = (1.0 - expanded_sigma) * target + expanded_sigma * noise
        velocity = noise - target
        weight = self.training_weights.to(target.device)[timestep_ids].to(target.dtype)
        return state, velocity, sigma, weight

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        base = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        self.sigmas = self._shift(base)

    @property
    def timesteps(self) -> torch.Tensor:
        if self.sigmas.numel() == 0:
            raise RuntimeError("Call set_timesteps before inference")
        return self.sigmas[:-1] * self.num_train_timesteps

    def step(self, velocity: torch.Tensor, step_index: int, sample: torch.Tensor) -> torch.Tensor:
        sigma = self.sigmas[step_index].to(sample.dtype)
        sigma_next = self.sigmas[step_index + 1].to(sample.dtype)
        return sample + velocity * (sigma_next - sigma)


def make_scheduler(
    num_steps: int,
    num_train_timesteps: int,
    shift: float,
    device: torch.device,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(num_train_timesteps, shift)
    scheduler.set_timesteps(num_steps, device)
    return scheduler
