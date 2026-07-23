from __future__ import annotations

import json
import numpy as np
import pytest

from script.server import InferenceSession, InputPreprocessor


class ThreePointRobot:
    def __init__(self, **_: object) -> None: pass

    def project_gripper_to_xyz(self, *_: object, **__: object) -> np.ndarray:
        return np.asarray([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [-0.1, 0.0, 0.5]], dtype=np.float32)


class LightweightInputPreprocessor(InputPreprocessor):
    def _initialize_perception(self, session, frame):
        session.object_nodes = []
        session.tracked_points = np.zeros((0, self.num_points, 3), dtype=np.float32)

    def _update_perception(self, session, frame): pass


def make_preprocessor(tmp_path, *, points=4):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "norm_stats_suite.json").write_text(json.dumps({"norm_stats": {
        "camera_xyz": {"q01": [-1.0, -1.0, 0.0], "q99": [1.0, 1.0, 1.0]},
    }}))
    return LightweightInputPreprocessor(
        history_horizon=1, future_horizon=2, num_points=points,
        robot_cls=ThreePointRobot, dataset_dir=tmp_path,
    )


def make_request():
    return {
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.depth.metric": np.ones((256, 256), dtype=np.float32) * 0.5,
        "observation.state": np.asarray([0.0] * 6 + [0.02, -0.02]),
        "camera.intrinsics": np.asarray([[100.0,0,128.0],[0,100.0,128.0],[0,0,1.0]]),
        "camera.extrinsics": np.eye(4),
    }


def test_online_input_uses_three_normalized_actor_xyz_points(tmp_path) -> None:
    preprocessor = make_preprocessor(tmp_path)
    subtaskstructure = {"nodes": []}
    session = InferenceSession(
        "episode-1", "libero", "test", taskstructure={"subtasks": [subtaskstructure]}
    )
    model_input = preprocessor.build(make_request(), session, subtaskstructure)
    points = np.asarray(model_input["entity_points"])
    assert points.shape == (1, 2, 3, 4, 3)
    np.testing.assert_allclose(
        points[0, 0, 0, :3],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        atol=1e-6,
    )
    assert np.asarray(model_input["gripper_closedness_history"])[0, 0, 0] == pytest.approx(0.0)
    assert "actor_metric_history" not in model_input
    assert "robot_metric_mask" not in model_input


def test_task_object_nodes_keep_duplicate_names_in_structure_order(tmp_path) -> None:
    preprocessor = make_preprocessor(tmp_path)
    taskstructure = {
        "subtasks": [
            {"nodes": [
                {"name": "gripper", "role": "actor"},
                {"name": "mug", "role": "patient"},
                {"name": "plate", "role": "target"},
            ]},
            {"nodes": [
                {"name": "gripper", "role": "actor"},
                {"name": "box", "role": "patient"},
                {"name": "plate", "role": "target"},
            ]},
        ]
    }
    session = InferenceSession("episode-1", "libero", "test", taskstructure=taskstructure)
    session.object_nodes = preprocessor._task_object_nodes(taskstructure)
    assert [node["name"] for node in session.object_nodes] == ["mug", "plate", "box", "plate"]
    assert preprocessor._subtask_object_indices(session, taskstructure["subtasks"][0]) == [0, 1]
    session.subtask_index = 1
    assert preprocessor._subtask_object_indices(session, taskstructure["subtasks"][1]) == [2, 3]


def test_online_object_features_backproject_metric_depth_and_repeat_valid_points(tmp_path) -> None:
    preprocessor = make_preprocessor(tmp_path)
    tracks = np.zeros((1, 4, 3), dtype=np.float32)
    tracks[0, :, :2] = [[128, 128], [138, 128], [400, 128], [128, 128]]
    tracks[0, :2, 2] = 1.0
    depth = np.ones((256, 256), dtype=np.float32) * 0.5
    intrinsic = np.asarray([[100.0, 0, 128.0], [0, 100.0, 128.0], [0, 0, 1.0]])
    points = preprocessor._object_feats(tracks, ["target"], depth, intrinsic, 256, 256)
    assert not points[0].any()
    assert points[1].shape == (4, 3)
    np.testing.assert_allclose(points[1, :, 0], [0.0, 0.05, 0.0, 0.05], atol=1e-6)
