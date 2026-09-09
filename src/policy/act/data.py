"""Dataset formatting for ACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ACTBatchTransform:
    image_keys: tuple[str, ...]
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = [self._image_tensor(data[key]) for key in self.image_keys]
        state = torch.as_tensor(data[self.state_key], dtype=torch.float32)
        actions = torch.as_tensor(data[self.action_key], dtype=torch.float32)
        is_pad = torch.as_tensor(data[f"{self.action_key}_is_pad"], dtype=torch.bool)
        if state.ndim != 1:
            raise ValueError(f"Expected state [D], got {state.shape}")
        if actions.ndim != 2:
            raise ValueError(f"Expected actions [H,D], got {actions.shape}")
        if is_pad.shape != actions.shape[:1]:
            raise ValueError(f"Expected is_pad {actions.shape[:1]}, got {is_pad.shape}")
        return {
            "images": torch.stack(images),
            "state": state,
            "actions": actions,
            "is_pad": is_pad,
        }

    @staticmethod
    def _image_tensor(value: Any) -> torch.Tensor:
        image = torch.as_tensor(value)
        if image.ndim != 3:
            raise ValueError(f"Expected image [C,H,W] or [H,W,C], got {image.shape}")
        if image.shape[-1] == 3:
            image = image.movedim(-1, 0)
        if image.shape[0] != 3:
            raise ValueError(f"Expected RGB image, got {image.shape}")
        image = image.float()
        if image.max() > 1.0:
            image *= 1.0 / 255.0
        return image
