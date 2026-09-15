"""Released CbF architecture adapted only at LIBERO input/output boundaries."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    policy_name: str = "cbf_code"
    state_dim: int = 8
    action_dim: int = 7
    # LIBERO stores 32 segmented XYZ points per object; the release used 100.
    num_points: int = 32
    obs_steps: int = 2
    horizon: int = 16
    action_steps: int = 8
    clip_model_path: str | Path | None = Path("/data0/luokang/dataset/luokang/ckpts/ViT-B-32.pt")
    clip_bpe_path: str | Path | None = Path(
        "/data0/luokang/dataset/luokang/ckpts/bpe_simple_vocab_16e6.txt.gz"
    )
    diffusion_steps: int = 100
    inference_steps: int = 100
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    num_groups: int = 8
    # The released code passes edge_attr to GATConv without edge_dim, so no
    # relationship semantics enter attention. False is intentionally code-faithful;
    # it is not the paper's graph-structure ablation because the fixed graph remains.
    semantic_edge_injection: bool = False
    action_delta: bool = True
    # The fields below describe the common LIBERO server boundary only.
    history_frames: tuple[int, ...] = (-1,)
    history_horizon: int = 1
    future_horizon: int = 8
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    point_coordinate_frame: str = "camera"

    def __post_init__(self) -> None:
        if self.policy_name != "cbf_code":
            raise ValueError("Expected policy_name='cbf_code'")
        if self.num_points != 32:
            raise ValueError("This LIBERO CbF-Code adaptation uses 32 segmented points")
        if self.semantic_edge_injection:
            raise ValueError("CbF-Code must preserve the release's absent edge semantics")
        if not self.action_delta or self.action_dim != 7:
            raise ValueError("CbF-Code outputs 7D LIBERO delta actions")
        if self.history_horizon != 1 or self.history_frames != (-1,) or self.future_horizon != self.action_steps:
            raise ValueError("Server horizons must match the CbF observation/action contract")

    def to_kwargs(self):
        kwargs = asdict(self)
        for key in (
            "policy_name", "action_delta", "history_frames", "history_horizon",
            "future_horizon", "actor_point_indices", "point_coordinate_frame",
        ):
            kwargs.pop(key)
        return kwargs


LIBERO_MODEL_CONFIG = ModelConfig()
