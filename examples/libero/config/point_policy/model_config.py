"""Point-Policy deterministic actor adapted to LIBERO tracked points and language."""

from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.schema import validate_actor_point_indices


@dataclass
class ModelConfig:
    policy_name: str = "point_policy"
    num_points: int = 32
    max_objects: int = 10
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    history_frames: tuple[int, ...] = (-9, -8, -7, -6, -5, -4, -3, -2, -1)
    future_horizon: int = 10
    repr_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 2
    dropout: float = 0.1
    stddev: float = 0.1
    language_dim: int = 384
    language_model_path: str | Path | None = Path("/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5")
    action_delta: bool = False
    point_coordinate_frame: str = "camera"

    def __post_init__(self):
        if self.policy_name != "point_policy" or self.action_delta or self.point_coordinate_frame != "camera":
            raise ValueError("Point-Policy predicts camera-frame points and executes absolute poses")
        self.actor_point_indices = validate_actor_point_indices(self.actor_point_indices)
        self.history_frames = tuple(self.history_frames)
        if self.history_frames != tuple(range(-len(self.history_frames), 0)):
            raise ValueError("Point-Policy requires consecutive history frames ending at -1")

    @property
    def history_horizon(self):
        return len(self.history_frames)

    def to_kwargs(self):
        kwargs = asdict(self)
        for key in ("policy_name", "history_frames", "action_delta", "point_coordinate_frame"):
            kwargs.pop(key)
        kwargs["history_horizon"] = self.history_horizon
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
