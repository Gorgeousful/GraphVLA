"""Official DP3 architecture with two observation frames and language."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    policy_name: str = "dp3"
    state_dim: int = 8
    action_dim: int = 7
    num_points: int = 512
    obs_steps: int = 2
    horizon: int = 16
    action_steps: int = 10
    encoder_output_dim: int = 64
    language_model_path: str | Path | None = Path("/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5")
    language_dim: int = 384
    language_projection_dim: int = 64
    diffusion_steps: int = 100
    inference_steps: int = 10
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    num_groups: int = 8
    action_delta: bool = True

    def __post_init__(self):
        if self.policy_name != "dp3" or self.obs_steps < 1:
            raise ValueError("Expected dp3 with positive obs_steps")
        if not self.action_delta:
            raise ValueError("LIBERO DP3 uses delta actions")

    def to_kwargs(self):
        kwargs = asdict(self)
        kwargs.pop("policy_name")
        kwargs.pop("action_delta")
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
