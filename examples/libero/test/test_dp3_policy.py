from copy import deepcopy

import pytest
import torch

from examples.libero.config.dp3.model_config import ModelConfig
from src.policy.registry import build_policy


def make_policy():
    return build_policy(ModelConfig(
        num_points=8, horizon=4, action_steps=2, encoder_output_dim=8,
        language_model_path=None, language_dim=8, language_projection_dim=8,
        down_dims=(16, 32), diffusion_step_embed_dim=16,
        diffusion_steps=4, inference_steps=2,
    ))


def test_dp3_language_gradient_padding_and_checkpoint():
    torch.set_num_threads(1)
    policy = make_policy()
    batch = {
        "point_cloud": torch.randn(2, 2, 8, 3),
        "state": torch.randn(2, 2, 8),
        "language_embedding": torch.randn(2, 8),
        "actions": torch.rand(2, 4, 7) * 2 - 1,
        "is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    loss, _ = policy(batch)
    loss.backward()
    assert policy.nets["language_projection"].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not any("ema" in key for key in policy.state_dict())
    restored = make_policy()
    restored.load_state_dict(deepcopy(policy.state_dict()))
    policy.eval()
    restored.eval()
    torch.manual_seed(5)
    expected = policy.predict_action(batch)
    torch.manual_seed(5)
    torch.testing.assert_close(restored.predict_action(batch), expected)
    assert expected.shape == (2, 2, 7)
    batch["is_pad"].fill_(True)
    assert policy(batch)[0].item() == 0
    with pytest.raises(ValueError, match="obs_steps"):
        ModelConfig(obs_steps=0)


def test_history_server_and_data_transform(tmp_path):
    import numpy as np
    from script.server import DP3InferenceServer
    from src.dataset.transform import Normalize, Unnormalize
    from src.policy.dp3.data import DP3BatchTransform, depth_to_point_cloud
    from types import SimpleNamespace

    policy = make_policy()
    config = ModelConfig(num_points=8, horizon=4, action_steps=2, encoder_output_dim=8,
                         language_model_path=None, language_dim=8, language_projection_dim=8,
                         down_dims=(16, 32), diffusion_step_embed_dim=16, diffusion_steps=4, inference_steps=2)
    checkpoint = tmp_path / "step_1.pt"
    torch.save({"model": policy.state_dict()}, checkpoint)
    stats = {"state": {"mean": [0.] * 8, "std": [1.] * 8},
             "action": {"mean": [0.] * 7, "std": [1.] * 7}}
    data_config = SimpleNamespace(transforms=(Normalize(stats, field_map={"state": "state"}, use_quantiles=False),),
                                 out_transforms=(Unnormalize(stats, field_map={"action": "action"}, use_quantiles=False),))
    server = DP3InferenceServer(host="localhost", port=8000, execute_chunk_len=2,
                               model_config=config, data_config=data_config,
                               ckpt_path=checkpoint, device=torch.device("cpu"))
    captured = {}
    def predict(batch):
        captured.update(batch)
        return torch.zeros(1, 2, 7)
    server.model.predict_action = predict
    depth = np.ones((4, 4), dtype=np.float32)
    intrinsic = np.array([[4., 0, 2], [0, 4., 2], [0, 0, 1]])
    request = {"benchmark": "libero", "session_id": "test", "language": "put the bowl on the stove",
               "observation.depth.metric": np.stack((depth * 2, depth)),
               "camera.intrinsics": np.stack((intrinsic, intrinsic)),
               "observation.state": np.stack((np.zeros(8), np.ones(8)))}
    assert len(server.infer_from_observation(request)["action"]) == 2
    expected = torch.stack([depth_to_point_cloud(frame, intrinsic, num_points=8)
                            for frame in (depth * 2, depth)])
    torch.testing.assert_close(captured["point_cloud"][0], expected)
    assert captured["state"].shape == (1, 2, 8)
    assert captured["language"] == [request["language"]]
    batch = DP3BatchTransform(num_points=8)({"observation.point_cloud": expected,
          "observation.state": np.stack((np.zeros(8), np.ones(8))), "action": np.zeros((4, 7)),
          "action_is_pad": [False, False, True, True], "language": request["language"]})
    torch.testing.assert_close(batch["point_cloud"], captured["point_cloud"][0])
    request["observation.depth.metric"] = depth
    request["observation.state"] = np.ones(8)
    server.infer_from_observation(request)
    torch.testing.assert_close(captured["point_cloud"][0, 0], captured["point_cloud"][0, 1])
    assert captured["state"].shape == (1, 2, 8)
    with pytest.raises(ValueError, match="valid depth"):
        depth_to_point_cloud(np.zeros((4, 4)), intrinsic, num_points=8)
