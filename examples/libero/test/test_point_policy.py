from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from examples.libero.config.point_policy.model_config import ModelConfig
from src.policy.registry import build_policy
from src.policy.point_policy.data import PointPolicyTransform, PointPolicyOutputTransform


def make_policy():
    torch.set_num_threads(1)
    return build_policy(ModelConfig(num_points=2, max_objects=2, actor_point_indices=(0, 1),
                                    history_frames=(-1,), future_horizon=3, repr_dim=16,
                                    hidden_dim=16, num_layers=2, num_heads=2, dropout=0.2,
                                    language_dim=8, language_model_path=None))


def make_batch():
    return {"point_tracks": torch.rand(2, 2, 6, 3), "point_mask": torch.ones(2, 2, 6, dtype=torch.bool),
            "gripper_history": torch.rand(2, 2, 1), "language_embedding": torch.rand(2, 8),
            "target_points": torch.rand(2, 3, 2, 3), "target_gripper": torch.rand(2, 3, 1),
            "target_mask": torch.ones(2, 3, dtype=torch.bool)}


def test_language_checkpointing_mask_and_weight_roundtrip():
    policy = make_policy()
    batch = make_batch()
    initial = deepcopy(policy.state_dict())
    runs = []
    for enabled in (False, True):
        policy.load_state_dict(initial)
        policy.zero_grad(set_to_none=True)
        policy.set_gradient_checkpointing(enabled)
        torch.manual_seed(19)
        loss, _ = policy(batch)
        loss.backward()
        runs.append((loss.detach(), {k: p.grad.clone() for k, p in policy.named_parameters()}))
    torch.testing.assert_close(runs[0][0], runs[1][0])
    for name in runs[0][1]:
        torch.testing.assert_close(runs[0][1][name], runs[1][1][name])
    assert runs[1][1]["language_projection.weight"].abs().sum() > 0
    policy.eval()
    batch["point_mask"][:, :, -2:] = False
    before = policy.sample(batch)["point_plan"]
    batch["point_tracks"][:, :, -2:] = float("nan")
    torch.testing.assert_close(policy.sample(batch)["point_plan"], before)
    restored = make_policy().eval()
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored.sample(batch)["point_plan"], before)
    changed = dict(batch, language_embedding=batch["language_embedding"] + 1)
    assert not torch.allclose(policy.sample(changed)["point_plan"], before)
    batch["target_mask"].fill_(False)
    batch["target_points"].fill_(float("nan"))
    assert policy(batch)[0].item() == 0


def test_data_temporal_alignment_and_normalization(tmp_path):
    stats = tmp_path / "norm_stats_suite.json"
    stats.write_text(json.dumps({"level": "suite", "norm_stats": {
        "camera_xyz_track": {"q01": [0, 0, 0], "q99": [2, 2, 2]}}}))
    transform = PointPolicyTransform(stats, history_horizon=1, future_horizon=3,
                                     actor_point_indices=(0, 1), num_points=2, max_objects=2)
    points = torch.arange(5.)[:, None, None].expand(5, 6, 3).clone()
    data = {"node_points_xyz_track": torch.ones(2, 2, 2, 3),
            "valid_node_mask": torch.tensor([[True, False], [True, False]]),
            "gripper_points_xyz": points, "gripper_points_xyz_is_pad": torch.tensor([True, False, False, False, True]),
            "action": torch.tensor([[0.] * 6 + [-1.], [0.] * 6 + [1.], [0.] * 6 + [1.]]),
            "action_is_pad": torch.tensor([False, False, False]), "language": "move bowl"}
    batch = transform(data)
    assert batch["point_tracks"].shape == (2, 6, 3)
    assert not batch["point_mask"][:, -2:].any()
    assert batch["point_tracks"][:, -2:].count_nonzero() == 0
    assert batch["target_mask"].tolist() == [True, True, False]
    assert batch["target_gripper"].flatten().tolist() == [0, 1, 1]
    output = PointPolicyOutputTransform(stats)({"outputs": {"point_plan": batch["target_points"][None]}})
    torch.testing.assert_close(output["outputs"]["point_plan"][0], points[2:, :2])


def test_online_points_preserve_identity_and_match_training(tmp_path):
    from script.server import PointPolicyPreprocessor, InferenceSession
    stats = tmp_path / "norm_stats_suite.json"
    stats.write_text(json.dumps({"norm_stats": {"camera_xyz_track": {"q01": [0, 0, 0], "q99": [4, 4, 4]}}}))
    transform = PointPolicyTransform(stats, history_horizon=1, future_horizon=3,
                                     actor_point_indices=(0, 1), num_points=2, max_objects=2)
    preprocessor = PointPolicyPreprocessor.__new__(PointPolicyPreprocessor)
    preprocessor.point_transform = transform
    preprocessor.num_points = 2
    session = InferenceSession("test", "libero", "move bowl")
    session.object_nodes = [{"name": "bowl"}]
    tracks = np.array([[[1, 1, 0], [6, 6, 1]]], dtype=np.float32)
    frame = {"tracks": tracks, "metric_depth": np.ones((4, 4)), "intrinsic": np.eye(3),
             "gripper_points_xyz": np.ones((6, 3))}
    online = preprocessor._build_model_input(session, [frame, frame], {})
    objects = torch.zeros(2, 2, 2, 3)
    objects[:, 0, 0] = 1
    mask = torch.zeros(2, 2, 2, dtype=torch.bool)
    mask[:, 0, 0] = True
    offline = transform.observations(objects, mask, torch.ones(2, 6, 3))
    for name, value in offline.items():
        torch.testing.assert_close(torch.as_tensor(online[name])[0], value)
    assert not torch.as_tensor(online["point_mask"])[0, 0, 3]


def test_point_server_load_and_output_denormalization(tmp_path):
    from script.server import InferenceModel
    config = ModelConfig(num_points=2, max_objects=2, actor_point_indices=(0, 1), history_frames=(-1,),
                         future_horizon=3, repr_dim=16, hidden_dim=16, num_layers=2, num_heads=2,
                         dropout=0.2, language_dim=8, language_model_path=None)
    model = make_policy().eval()
    path = tmp_path / "step_1.pt"
    torch.save(model.state_dict(), path)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"norm_stats": {"camera_xyz_track": {"q01": [1, 2, 3], "q99": [3, 4, 5]}}}))
    output_transform = PointPolicyOutputTransform(stats)
    wrapper = InferenceModel(model_config=config, ckpt_path=path, device=torch.device("cpu"),
                             data_kwargs={"out_transforms": (output_transform,)})
    batch = make_batch()
    expected = output_transform({"outputs": model.sample(batch)})["outputs"]
    batch["language"] = ["test", "test"]
    actual = wrapper.infer(batch)
    torch.testing.assert_close(torch.tensor(actual["point_plan"]), expected["point_plan"])
    assert "subtask_progress" not in actual


def test_environment_controlled_server_and_language_reset(tmp_path):
    from types import SimpleNamespace
    from script.server import PointPolicyInferenceServer, REQUIRED_REQUEST_FIELDS
    seen = []

    def preprocess(request, session, structure):
        seen.append((session.language, len(session.feature_history)))
        session.feature_history.append({})
        return {}

    # There is deliberately no planner.update_after_inference method.
    server = PointPolicyInferenceServer(
        host="localhost", port=8007, execute_chunk_len=2, ckpt_path=tmp_path / "step_1.pt",
        planner=SimpleNamespace(_taskstructure=lambda language: {"subtasks": []}),
        preprocessor=SimpleNamespace(build=preprocess),
        inference=SimpleNamespace(infer=lambda data, **kwargs: {}),
        embodiment=SimpleNamespace(future_horizon=3, to_action=lambda *args: [[0.] * 7] * 3),
    )
    request = {k: [] for k in REQUIRED_REQUEST_FIELDS}
    request.update(benchmark="libero", session_id="test", language="move bowl")
    for _ in range(2):
        response = server.infer_from_observation(request)
        assert len(response["action"]) == 2 and response["episode_done"] is False
    request["language"] = "move plate"
    server.infer_from_observation(request)
    assert seen == [("move bowl", 0), ("move bowl", 1), ("move plate", 0)]
    assert server.server_info()["action_delta"] is False
    assert isinstance(server.server_info()["progress_threshold"], float)
