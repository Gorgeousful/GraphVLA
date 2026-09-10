"""Model configuration for the LIBERO official image U-Net Diffusion Policy baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    policy_name: str = "dp"
    state_dim: int = 8
    action_dim: int = 7
    camera_keys: tuple[str, ...] = (
        "observation.images.image",
        "observation.images.wrist_image",
    )
    img_size: int = 224
    crop_size: int = 224
    visual_feature_dim: int = 512
    pretrained_backbone_path: str | Path | None = None
    language_model_path: str | Path | None = Path(
        "/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5"
    )
    language_dim: int = 384
    obs_steps: int = 2
    horizon: int = 16
    action_steps: int = 10
    diffusion_steps: int = 100
    inference_steps: int = 100
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    num_groups: int = 8
    cond_predict_scale: bool = True
    action_delta: bool = True

    def __post_init__(self) -> None:
        if self.policy_name != "dp":
            raise ValueError(
                f"Expected policy_name='dp', got {self.policy_name!r}"
            )
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Diffusion Policy camera_keys must be non-empty and unique")
        if self.img_size <= 0 or not 0 < self.crop_size <= self.img_size:
            raise ValueError("img_size must be positive and crop_size must not exceed it")
        if self.visual_feature_dim <= 0 or self.language_dim <= 0:
            raise ValueError("visual_feature_dim and language_dim must be positive")

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("policy_name")
        kwargs.pop("action_delta")
        kwargs["num_cameras"] = len(kwargs.pop("camera_keys"))
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
