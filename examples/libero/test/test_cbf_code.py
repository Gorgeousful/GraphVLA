from __future__ import annotations

import threading

import torch
import numpy as np
from torch import nn

from script.server import CbFCodePreprocessor, InferenceSession
from src.policy.cbf_code.clip_text import FrozenClipTextEncoder
from src.policy.cbf_code.data import CbFCodeBatchTransform
from src.policy.cbf_code.model import CbFCodePolicy


def _model() -> CbFCodePolicy:
    return CbFCodePolicy(
        clip_model_path=None,
        clip_bpe_path=None,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        num_groups=8,
        diffusion_steps=4,
        inference_steps=2,
    )


def test_cbf_code_transform_uses_32_point_segmented_nodes() -> None:
    transform = CbFCodeBatchTransform()
    points = torch.arange(16 * 4 * 32 * 3, dtype=torch.float32).reshape(16, 4, 32, 3)
    valid = torch.ones(16, 4, dtype=torch.bool)
    active = torch.zeros(16, 4, dtype=torch.bool)
    active[:, (1, 3)] = True
    result = transform({
        "node_points_xyz": points,
        "valid_node_mask": valid,
        "subtask_node_mask": active,
        "gripper_points_xyz": torch.randn(16, 6, 3),
        "state": torch.randn(16, 8),
        "action": torch.randn(16, 7),
        "history_horizon": 1,
        "subtaskstructure": {
            "subtask": "put the bowl right of the plate",
            "nodes": [
                {"role": "actor"}, {"role": "patient"}, {"role": "target"},
            ],
        },
    })
    assert result["node_point_clouds"].shape == (2, 3, 32, 3)
    torch.testing.assert_close(result["node_point_clouds"][:, 1], points[:2, 1])
    torch.testing.assert_close(result["node_point_clouds"][:, 2], points[:2, 3])
    assert result["language"] == "put the bowl right of the plate"


def test_cbf_code_predicts_epsilon_and_returns_eight_delta_actions() -> None:
    model = _model()
    batch = {
        "node_point_clouds": torch.randn(2, 2, 3, 32, 3),
        "state": torch.randn(2, 2, 8),
        "actions": torch.randn(2, 16, 7),
        "language_embedding": torch.randn(2, 512),
    }
    loss, metrics = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and metrics["loss_mse"].shape == ()
    actions = model.predict_action(batch)
    assert actions.shape == (2, 8, 7)
    assert not any(name.startswith("ema") for name, _ in model.named_parameters())


def test_cbf_code_has_no_semantic_edge_input() -> None:
    model = _model()
    assert "edge" not in " ".join(dict(model.named_parameters())).lower()


def test_cbf_clip_embeddings_are_cached_per_unique_text() -> None:
    class FakeTokenizer:
        def tokenize(self, texts):
            return torch.tensor([[len(text)] for text in texts])

    class FakeTextTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def encode_text(self, tokens):
            self.calls += 1
            return torch.nn.functional.one_hot(tokens[:, 0] % 512, 512).float()

    encoder = FrozenClipTextEncoder.__new__(FrozenClipTextEncoder)
    encoder.model = FakeTextTower()
    encoder.tokenizer = FakeTokenizer()
    encoder.device = torch.device("cpu")
    encoder.cache = {}
    encoder.lock = threading.Lock()
    first = encoder.encode(["same", "same"], device=torch.device("cpu"))
    second = encoder.encode(["same", "different"], device=torch.device("cpu"))
    assert encoder.model.calls == 2
    torch.testing.assert_close(first[0], first[1])
    torch.testing.assert_close(first[0], second[0])
    assert set(encoder.cache) == {"same", "different"}


def test_cbf_online_input_resamples_segmentation_without_point_tracker() -> None:
    preprocessor = CbFCodePreprocessor(
        history_horizon=1,
        history_frames=(-1,),
        future_horizon=8,
        num_points=32,
        actor_point_indices=(0, 1, 2, 3, 4, 5),
        robot_cls=object,
        dataset_dir="/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0904_20hz",
    )
    subtask = {
        "subtask": "put the bowl right of the plate",
        "nodes": [{"role": "actor"}, {"role": "patient"}, {"role": "target"}],
    }
    session = InferenceSession("test", "libero", "task")
    session.taskstructure = {"subtasks": [subtask]}
    session.object_nodes = [{"role": "patient"}, {"role": "target"}]
    tracks = np.ones((2, 32, 3), dtype=np.float32)
    frames = [{
        "tracks": tracks,
        "metric_depth": np.ones((4, 4), dtype=np.float32),
        "gripper_points_xyz": np.ones((6, 3), dtype=np.float32),
        "intrinsic": np.eye(3, dtype=np.float32),
        "state": np.zeros(8, dtype=np.float32),
    }] * 2
    result = preprocessor._build_model_input(session, frames, subtask)
    assert preprocessor.sam_only and session.point_tracker is None
    assert np.asarray(result["node_point_clouds"]).shape == (1, 2, 3, 32, 3)
