"""Model configuration for the LIBERO ACT baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    policy_name: str = "act"
    state_dim: int = 8
    action_dim: int = 7
    chunk_size: int = 10
    hidden_dim: int = 512
    feedforward_dim: int = 3200
    encoder_layers: int = 4
    decoder_layers: int = 7
    num_heads: int = 8
    dropout: float = 0.1
    latent_dim: int = 32
    kl_weight: float = 10.0
    camera_keys: tuple[str, ...] = (
        "observation.images.image",
        "observation.images.wrist_image",
    )
    img_size: int = 224
    pretrained_backbone: bool = True
    pretrained_backbone_path: str | Path | None = Path(
        "/data0/luokang/dataset/luokang/ckpts/resnet18-imagenet1k-v1/resnet18-f37072fd.pth"
    )
    language_model_path: str | Path | None = Path(
        "/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5"
    )
    language_dim: int = 384
    pre_norm: bool = False
    action_delta: bool = True

    def __post_init__(self) -> None:
        if self.policy_name != "act":
            raise ValueError(f"Expected policy_name='act', got {self.policy_name!r}")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("ACT camera_keys must be non-empty and unique")
        if self.img_size <= 0:
            raise ValueError(f"ACT img_size must be positive, got {self.img_size}")

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("policy_name")
        kwargs.pop("action_delta")
        kwargs["num_cameras"] = len(kwargs.pop("camera_keys"))
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
