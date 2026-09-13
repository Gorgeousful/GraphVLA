from __future__ import annotations

import torch

from src.policy.dp.data import DiffusionPolicyBatchTransform
from src.policy.dp.model import DiffusionPolicy


def make_policy() -> DiffusionPolicy:
    return DiffusionPolicy(
        state_dim=8,
        action_dim=7,
        num_cameras=2,
        img_size=32,
        crop_size=28,
        visual_feature_dim=16,
        language_model_path=None,
        language_dim=8,
        obs_steps=2,
        horizon=8,
        action_steps=4,
        diffusion_steps=4,
        inference_steps=2,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        kernel_size=3,
        num_groups=8,
    )


def test_batch_transform_formats_camera_history() -> None:
    transform = DiffusionPolicyBatchTransform(("agent", "wrist"))
    batch = transform(
        {
            "agent": torch.zeros(2, 32, 32, 3),
            "wrist": torch.zeros(2, 32, 32, 3),
            "observation.state": torch.zeros(2, 8),
            "action": torch.zeros(8, 7),
            "action_is_pad": torch.zeros(8, dtype=torch.bool),
            "language": "pick up the bowl",
        }
    )
    assert batch["images"].shape == (2, 2, 3, 32, 32)
    assert batch["language"] == "pick up the bowl"


def test_policy_train_and_inference_shapes() -> None:
    policy = make_policy()
    batch = {
        "images": torch.rand(2, 2, 2, 3, 32, 32),
        "state": torch.rand(2, 2, 8),
        "actions": torch.rand(2, 8, 7) * 2 - 1,
        "is_pad": torch.zeros(2, 8, dtype=torch.bool),
        "language_embedding": torch.rand(2, 8),
    }
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(metrics) == {"loss_mse"}
    actions = policy.eval().predict_action(batch)
    assert actions.shape == (2, 4, 7)
    assert torch.isfinite(actions).all()
