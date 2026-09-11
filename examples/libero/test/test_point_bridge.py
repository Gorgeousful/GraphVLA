from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from examples.libero.config.point_bridge.model_config import ModelConfig
from src.policy.point_bridge.data import PointBridgeTransform, PointBridgeOutputTransform, camera_pose_to_world_action
from src.policy.registry import build_policy


def small_config(mode):
    return ModelConfig(action_mode=mode, num_points=2, max_objects=2, history_frames=(-1,),
                       future_horizon=3, repr_dim=16, hidden_dim=16, num_layers=2,
                       num_heads=4, dropout=0.2, language_dim=8, language_model_path=None)


def batch_for(mode):
    return {"robot_points": torch.rand(2, 2, 6, 3), "object_points": torch.rand(2, 2, 2, 2, 3),
            "object_mask": torch.ones(2, 2, 2, 2, dtype=torch.bool), "language_embedding": torch.rand(2, 8),
            "target_actions": torch.rand(2, 3, 10 if mode == "pose" else 21),
            "target_mask": torch.ones(2, 3, dtype=torch.bool)}


@pytest.mark.parametrize("mode", ["pose", "points"])
def test_language_gradients_checkpointing_mask_permutation_and_roundtrip(mode):
    torch.set_num_threads(1)
    policy = build_policy(small_config(mode))
    batch = batch_for(mode)
    initial = deepcopy(policy.state_dict())
    runs = []
    for enabled in (False, True):
        policy.load_state_dict(initial)
        policy.zero_grad(set_to_none=True)
        policy.set_gradient_checkpointing(enabled)
        torch.manual_seed(9)
        loss, _ = policy(batch)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
        runs.append((loss.detach(), {k: p.grad.clone() for k, p in policy.named_parameters()}))
    torch.testing.assert_close(runs[0][0], runs[1][0])
    for name in runs[0][1]:
        torch.testing.assert_close(runs[0][1][name], runs[1][1][name])
    assert runs[1][1]["language_projector.0.weight"].abs().sum() > 0
    policy.eval()
    key = "pose_plan" if mode == "pose" else "point_plan"
    before = policy.sample(batch)[key]
    permuted = dict(batch, object_points=batch["object_points"].flip(2).flip(3))
    torch.testing.assert_close(policy.sample(permuted)[key], before)
    assert not torch.allclose(policy.sample(dict(batch, language_embedding=batch["language_embedding"] + 1))[key], before)
    batch["object_mask"][:, :, -1] = False
    before = policy.sample(batch)[key]
    batch["object_points"][:, :, -1] = float("nan")
    torch.testing.assert_close(policy.sample(batch)[key], before)
    restored = build_policy(small_config(mode)).eval()
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored.sample(batch)[key], before)
    batch["object_mask"].fill_(False)
    assert torch.isfinite(policy.sample(batch)[key]).all()
    batch["target_mask"].fill_(False)
    batch["target_actions"].fill_(float("nan"))
    assert policy(batch)[0].item() == 0


def fixture_data(tmp_path):
    (tmp_path / "meta").mkdir(exist_ok=True)
    stats = tmp_path / "meta/norm_stats_suite.json"
    stats.write_text(json.dumps({"norm_stats": {"camera_xyz": {"q01": [-2, -2, -2], "q99": [3, 3, 3]}}}))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()
    extrinsic[:3, 3] = [.5, -.4, .2]
    (tmp_path / "meta/cameras.json").write_text(json.dumps([
        {"task_index": 0, "cameras": {"agentview": {"extrinsic": extrinsic.tolist()}}}]))
    states = torch.tensor([[.2, .3, .7, .4, -.3, .2, .01, -.01],
                           [.3, .2, .8, -.2, .5, .3, .02, -.02],
                           [.4, .1, .9, .1, .2, -.4, .03, -.03]])
    data = {"node_points_xyz": torch.ones(2, 2, 2, 3),
            "valid_node_mask": torch.tensor([[True, False], [True, False]]),
            "gripper_points_xyz": torch.rand(5, 6, 3) + .1,
            "gripper_points_xyz_is_pad": torch.tensor([True, False, False, False, True]),
            "observation.state": states, "observation.state_is_pad": torch.tensor([False, False, True]),
            "action": torch.tensor([[0.] * 6 + [-1.], [0.] * 6 + [1.], [0.] * 6 + [1.]]),
            "action_is_pad": torch.zeros(3, dtype=torch.bool), "task_index": torch.tensor(0),
            "language": "put the bowl on the plate"}
    return stats, extrinsic, data


@pytest.mark.parametrize("mode", ["pose", "points"])
def test_future_alignment_normalization_and_pose_frame_roundtrip(tmp_path, mode):
    stats, extrinsic, data = fixture_data(tmp_path)
    transform = PointBridgeTransform(stats, tmp_path, mode, history_horizon=1,
                                     future_horizon=3, num_points=2, max_objects=2)
    result = transform(data)
    assert result["target_mask"].tolist() == [True, True, False]
    assert result["object_points"][:, 1].count_nonzero() == 0
    assert result["target_actions"][:, -1].tolist() == [0, 1, 1]
    if mode == "pose":
        output = PointBridgeOutputTransform(stats, mode)({"outputs": {"pose_plan": result["target_actions"][None]}})
        actions = camera_pose_to_world_action(output["outputs"]["pose_plan"][0], extrinsic)
        np.testing.assert_allclose(actions[:, :6], data["observation.state"][:, :6], atol=2e-6)
        np.testing.assert_allclose(actions[:, -1], data["action"][:, -1], atol=1e-6)
    else:
        xyz = result["target_actions"][:, :-3].reshape(1, 3, 6, 3)
        output = PointBridgeOutputTransform(stats, mode)({"outputs": {"point_plan": xyz}})
        torch.testing.assert_close(output["outputs"]["point_plan"][0], data["gripper_points_xyz"][2:], atol=2e-6, rtol=1e-5)


def test_online_sam_matches_offline_without_role_or_track_identity(tmp_path):
    from script.server import PointBridgePreprocessor, InferenceSession
    stats, _, _ = fixture_data(tmp_path)
    transform = PointBridgeTransform(stats, tmp_path, history_horizon=1, future_horizon=3,
                                     num_points=2, max_objects=2)
    preprocessor = PointBridgePreprocessor.__new__(PointBridgePreprocessor)
    preprocessor.point_transform, preprocessor.num_points = transform, 2
    session = InferenceSession("test", "libero", "move bowl")
    session.object_nodes = [{"name": "bowl"}, {"name": "plate"}]
    tracks = np.array([[[1, 1, 1], [6, 6, 1]], [[0, 0, 0], [0, 0, 0]]], dtype=np.float32)
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


@pytest.mark.parametrize("mode", ["pose", "points"])
def test_server_loads_native_outputs_and_uses_saved_normalization(tmp_path, mode):
    from script.server import InferenceModel
    stats, _, _ = fixture_data(tmp_path)
    config = small_config(mode)
    model = build_policy(config).eval()
    path = tmp_path / "step_1.pt"
    torch.save(model.state_dict(), path)
    transform = PointBridgeOutputTransform(stats, mode)
    server = InferenceModel(model_config=config, data_kwargs={"out_transforms": (transform,)},
                            ckpt_path=path, device=torch.device("cpu"))
    batch = batch_for(mode)
    batch["language"] = ["move bowl", "move plate"]
    actual = server.infer(batch)
    expected = transform({"outputs": model.sample(batch)})["outputs"]
    for key in expected:
        torch.testing.assert_close(torch.as_tensor(actual[key]), expected[key])


def test_pose_adapter_executes_absolute_actions_without_point_fitting(tmp_path):
    from script.server import PointBridgeEmbodimentAdapter, InferenceSession
    stats, extrinsic, data = fixture_data(tmp_path)
    transform = PointBridgeTransform(stats, tmp_path, history_horizon=1, future_horizon=3,
                                     num_points=2, max_objects=2)
    prediction = transform(data)["target_actions"][None]
    outputs = PointBridgeOutputTransform(stats)({"outputs": {"pose_plan": prediction}})["outputs"]
    adapter = PointBridgeEmbodimentAdapter(future_horizon=3, actor_point_indices=(0, 1, 2, 3, 4, 5),
                                           action_delta=False, robot_cls=object)
    actions = adapter.to_action(outputs, {"camera.extrinsics": [extrinsic.tolist()]},
                                InferenceSession("test", "libero", "move bowl"))
    np.testing.assert_allclose(np.asarray(actions)[:, :6], data["observation.state"][:, :6], atol=2e-6)
    assert adapter.robot is None


@pytest.mark.parametrize("mode", ["pose", "points"])
def test_shared_sam_server_chunk_execution_and_language_reset(tmp_path, mode):
    from types import SimpleNamespace
    from script.server import (PointBridgePreprocessor, PointBridgeInferenceServer,
                               REQUIRED_REQUEST_FIELDS)
    stats, _, _ = fixture_data(tmp_path)
    transform = PointBridgeTransform(stats, tmp_path, mode, history_horizon=1,
                                     future_horizon=3, num_points=6, max_objects=2)
    preprocessor = PointBridgePreprocessor(point_transform=transform, dataset_dir=tmp_path,
                                           history_horizon=1, future_horizon=3, num_points=6,
                                           actor_point_indices=(0, 1, 2, 3, 4, 5), robot_cls=object)
    assert preprocessor.sam_only
    seen = []

    def preprocess(request, session, structure):
        seen.append((session.language, len(session.feature_history)))
        session.feature_history.append({})
        return {}

    preprocessor.build = preprocess
    server = PointBridgeInferenceServer(
        host="localhost", port=8008, execute_chunk_len=2, ckpt_path=tmp_path / "step_1.pt",
        planner=SimpleNamespace(_taskstructure=lambda language: {"subtasks": []}), preprocessor=preprocessor,
        inference=SimpleNamespace(infer=lambda *args, **kwargs: {}, to_json=lambda value: value,
                                  model=SimpleNamespace(action_mode=mode)),
        embodiment=SimpleNamespace(future_horizon=3, actor_num_points=6,
                                    to_action=lambda *args: [[0.] * 7] * 3))
    request = {k: [] for k in REQUIRED_REQUEST_FIELDS}
    request.update(benchmark="libero", session_id="test", language="move bowl")
    for _ in range(2):
        response = server.infer_from_observation(request)
        assert len(response["action"]) == 2 and response["episode_done"] is False
    request["language"] = "move plate"
    server.infer_from_observation(request)
    assert seen == [("move bowl", 0), ("move bowl", 1), ("move plate", 0)]
    assert server.server_info()["action_mode"] == mode
    assert server.server_info()["action_delta"] is False
