from __future__ import annotations

import torch

from src.policy.act.data import ACTBatchTransform
from src.policy.act.model import ACTPolicy


def make_policy() -> ACTPolicy:
    return ACTPolicy(
        state_dim=8,
        action_dim=7,
        chunk_size=3,
        hidden_dim=32,
        feedforward_dim=64,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        dropout=0.0,
        latent_dim=4,
        kl_weight=1.0,
        num_cameras=1,
        pretrained_backbone=False,
    )


def test_act_forward_and_predict_action() -> None:
    policy = make_policy()
    batch = {
        "state": torch.randn(2, 8),
        "images": torch.rand(2, 1, 3, 64, 64),
        "actions": torch.randn(2, 3, 7),
        "is_pad": torch.tensor([[False, False, False], [False, True, True]]),
    }

    loss, metrics = policy(batch)
    loss.backward()
    actions = policy.eval().predict_action(batch)

    assert loss.ndim == 0
    assert set(metrics) == {"loss_l1", "loss_kl"}
    assert actions.shape == (2, 3, 7)


def test_act_batch_transform() -> None:
    transform = ACTBatchTransform(image_keys=("agent",))
    result = transform(
        {
            "agent": torch.zeros(3, 8, 8),
            "observation.state": torch.zeros(8),
            "action": torch.zeros(3, 7),
            "action_is_pad": torch.tensor([False, False, True]),
        }
    )

    assert result["images"].shape == (1, 3, 8, 8)
    assert result["images"].dtype == torch.float32
    assert result["images"].max() == 0
    assert result["is_pad"].tolist() == [False, False, True]
