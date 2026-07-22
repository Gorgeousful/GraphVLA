from __future__ import annotations

import json
import numpy as np
import pytest

from script.server import InferenceSession, InputPreprocessor


class SixPointRobot:
    def __init__(self, **_: object) -> None: pass
    def project_gripper_to_uvd(self, *_: object, gripper_width: float, **__: object):
        values = [
            [128.0, 128.0, 0.1], [138.0, 128.0, 0.2], [118.0, 128.0, 0.3],
            [128.0, 138.0, 0.4], [128.0, 118.0, 0.5], [128.0, 148.0, 0.6],
        ]
        names = ("root_uvd", "left_base_uvd", "right_base_uvd", "left_fingertip_uvd", "right_fingertip_uvd", "tcp_uvd")
        return {name: np.asarray(value, dtype=np.float32) for name, value in zip(names, values, strict=True)}


class LightweightInputPreprocessor(InputPreprocessor):
    def _initialize_perception(self, session, frame):
        session.object_nodes = []
        session.tracked_points = np.zeros((0, self.num_points, 3), dtype=np.float32)
    def _update_perception(self, session, frame): pass
    def _predict_depth(self, session, frame):
        depth = np.zeros((256, 256), dtype=np.float32)
        for u, v, value in ((128,128,.1),(138,128,.2),(118,128,.3),(128,138,.4),(128,118,.5),(128,148,.6)):
            depth[v, u] = value
        return depth


def test_online_input_uses_four_semantic_actor_rays(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "norm_stats_suite.json").write_text(json.dumps({"norm_stats": {
        "depths.depth_rel": {"q01": [0.0], "q99": [1.0]},
        "gripper_d": {"q01": [0.0], "q99": [1.0]},
    }}))
    preprocessor = LightweightInputPreprocessor(
        history_horizon=1, future_horizon=2, num_points=4,
        robot_cls=SixPointRobot, dataset_dir=tmp_path,
    )
    request = {
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.asarray([0.0] * 6 + [0.02, -0.02]),
        "camera.intrinsics": np.asarray([[100.0,0,128.0],[0,100.0,128.0],[0,0,1.0]]),
        "camera.extrinsics": np.eye(4),
    }
    model_input = preprocessor.build(
        request, InferenceSession("episode-1", "libero", "test"), {"nodes": []}
    )
    points = np.asarray(model_input["entity_points"])
    assert points.shape == (1, 2, 3, 4, 3)
    np.testing.assert_allclose(points[0, 0, 0, :, 0], [0.0, 0.1, -0.1, 0.0])
    np.testing.assert_allclose(points[0, 0, 0, :, 2], [0.1, 0.2, 0.3, 0.45])
    assert np.asarray(model_input["actor_metric_history"]).shape == (1, 2, 4, 1)
    assert np.asarray(model_input["gripper_closedness_history"])[0, 0, 0] == pytest.approx(0.0)
    assert model_input["scene_condition_texts"] == ["", None]
    assert model_input["entity_role_condition_texts"] == ["actor", "patient", "target"]


def test_online_input_applies_ray_scale(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "norm_stats_suite.json").write_text(json.dumps({"norm_stats": {
        "depths.depth_rel": {"q01": [0.0], "q99": [1.0]},
        "gripper_d": {"q01": [0.0], "q99": [1.0]},
    }}))
    preprocessor = LightweightInputPreprocessor(
        history_horizon=1, future_horizon=2, num_points=4,
        robot_cls=SixPointRobot, dataset_dir=tmp_path, ray_scale=2.0,
    )
    request = {
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.asarray([0.0] * 6 + [0.02, -0.02]),
        "camera.intrinsics": np.asarray([[100.0,0,128.0],[0,100.0,128.0],[0,0,1.0]]),
        "camera.extrinsics": np.eye(4),
    }
    model_input = preprocessor.build(
        request, InferenceSession("episode-1", "libero", "test"), {"nodes": []}
    )
    points = np.asarray(model_input["entity_points"])
    np.testing.assert_allclose(points[0, 0, 0, :, 0], [0.0, 0.2, -0.2, 0.0])


def test_online_object_features_use_semantic_target_slot(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "norm_stats_suite.json").write_text(json.dumps({"norm_stats": {
        "depths.depth_rel": {"q01": [0.0], "q99": [1.0]},
        "gripper_d": {"q01": [0.0], "q99": [1.0]},
    }}))
    preprocessor = LightweightInputPreprocessor(
        history_horizon=1, future_horizon=2, num_points=4,
        robot_cls=SixPointRobot, dataset_dir=tmp_path,
    )
    tracks = np.zeros((1, 4, 3), dtype=np.float32)
    tracks[..., :2] = 128.0
    depth = np.ones((256, 256), dtype=np.float32)
    intrinsic = np.asarray([[100.0, 0, 128.0], [0, 100.0, 128.0], [0, 0, 1.0]])
    points = preprocessor._object_feats(tracks, ["target"], depth, intrinsic, 256, 256)
    assert not points[0].any()
    assert points[1].any()
