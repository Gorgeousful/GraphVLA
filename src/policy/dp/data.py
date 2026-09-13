"""Dataset formatting for visual Diffusion Policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DiffusionPolicyBatchTransform:
    image_keys: tuple[str, ...]
    state_key: str = "observation.state"
    action_key: str = "action"
    language_key: str = "language"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = [self._images(data[key]) for key in self.image_keys]
        state = torch.as_tensor(data[self.state_key], dtype=torch.float32)
        actions = torch.as_tensor(data[self.action_key], dtype=torch.float32)
        is_pad = torch.as_tensor(data[f"{self.action_key}_is_pad"], dtype=torch.bool)
        if state.ndim != 2:
            raise ValueError(f"Expected state [T,D], got {state.shape}")
        if actions.ndim != 2 or is_pad.shape != actions.shape[:1]:
            raise ValueError(f"Expected actions [H,D] and is_pad [H], got {actions.shape}, {is_pad.shape}")
        return {
            "images": torch.stack(images, dim=1),
            "state": state,
            "actions": actions,
            "is_pad": is_pad,
            "language": str(data[self.language_key]),
        }

    @staticmethod
    def _images(value: Any) -> torch.Tensor:
        images = torch.as_tensor(value)
        if images.ndim != 4:
            raise ValueError(f"Expected images [T,C,H,W] or [T,H,W,C], got {images.shape}")
        if images.shape[-1] == 3:
            images = images.movedim(-1, -3)
        if images.shape[-3] != 3:
            raise ValueError(f"Expected RGB images, got {images.shape}")
        images = images.float()
        if images.max() > 1.0:
            images *= 1.0 / 255.0
        return images
