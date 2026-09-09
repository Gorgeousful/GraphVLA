"""Optimizer and learning-rate management."""

from __future__ import annotations

import math
from typing import Any

import torch


class TrainingOptimizer:
    def __init__(self, model: torch.nn.Module, config: Any) -> None:
        self.config = config
        backbone_lr = getattr(config, "backbone_lr", None)
        parameters: Any = model.parameters()
        if backbone_lr is not None:
            regular = []
            backbone = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                (backbone if "backbone" in name else regular).append(parameter)
            parameters = [
                {"params": regular, "lr_scale": 1.0},
                {
                    "params": backbone,
                    "lr": backbone_lr,
                    "lr_scale": backbone_lr / config.peak_lr,
                },
            ]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.peak_lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

    def set_step_lr(self, step: int) -> float:
        lr = self.lr_at_step(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        return lr

    def lr_at_step(self, step: int) -> float:
        warmup_steps = self.config.warmup_steps
        peak_lr = self.config.peak_lr
        decay_lr = self.config.decay_lr
        decay_steps = max(1, self.config.decay_steps)

        if warmup_steps > 0 and step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps

        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return decay_lr + (peak_lr - decay_lr) * cosine

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)
