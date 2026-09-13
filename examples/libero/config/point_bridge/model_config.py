"""Point Bridge with shared LIBERO perception and native pose/points action heads."""

from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.schema import validate_actor_point_indices


@dataclass
class ModelConfig:
    policy_name: str = "point_bridge"
    action_mode: str = "points"  # "pose" or "points"
    num_points: int = 32
    max_objects: int = 10
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    history_frames: tuple[int, ...] = ()  # Official default: current observation only.
    future_horizon: int = 10
    repr_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    stddev: float = 0.1
    language_dim: int = 384
    language_model_path: str | Path | None = Path("/data0/luokang/dataset/luokang/ckpts/all-MiniLM-L6-v2")
    action_delta: bool = False
    point_coordinate_frame: str = "camera"

    def __post_init__(self):
        if self.policy_name != "point_bridge" or self.action_mode not in ("pose", "points"):
            raise ValueError("Expected point_bridge with action_mode='pose' or 'points'")
        if self.action_delta or self.point_coordinate_frame != "camera":
            raise ValueError("Point Bridge uses camera observations and absolute LIBERO execution")
        self.actor_point_indices = validate_actor_point_indices(self.actor_point_indices)
        self.history_frames = tuple(self.history_frames)
        if self.history_frames != tuple(range(-len(self.history_frames), 0)):
            raise ValueError("Expected consecutive history frames ending at -1")

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
