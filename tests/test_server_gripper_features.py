from __future__ import annotations

import json

import numpy as np

from script.server import InferenceSession, InputPreprocessor


class SixPointRobot:
    def __init__(self, **_: object) -> None:
        pass

    def project_gripper_to_uvd(
        self,
        *_: object,
        gripper_width: float,
        **__: object,
    ) -> dict[str, np.ndarray]:
        assert gripper_width == 0.04
        values = (
            ("root_uvd", [10.0, 10.0, 0.1]),
            ("left_base_uvd", [20.0, 20.0, 0.2]),
            ("right_base_uvd", [30.0, 30.0, 0.3]),
            ("left_fingertip_uvd", [40.0, 40.0, 0.4]),
            ("right_fingertip_uvd", [50.0, 50.0, 0.5]),
            ("tcp_uvd", [60.0, 60.0, 0.6]),
        )
        return {name: np.asarray(value, dtype=np.float32) for name, value in values}


class LightweightInputPreprocessor(InputPreprocessor):
    def _initialize_perception(self, session, frame) -> None:
        session.object_nodes = []
        session.tracked_points = np.zeros((0, self.num_points, 3), dtype=np.float32)

    def _update_perception(self, session, frame) -> None:
        pass

    def _predict_depth(self, session, frame) -> np.ndarray:
        depth = np.zeros((256, 256), dtype=np.float32)
        for point_index, coordinate in enumerate((10, 20, 30, 40, 50, 60)):
            depth[coordinate, coordinate] = 0.1 * (point_index + 1)
        return depth


def test_online_model_input_uses_six_gripper_points_and_fingertip_mean_for_tcp_depth(tmp_path) -> None:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "norm_stats_suite.json").write_text(json.dumps({
        "level": "suite",
        "norm_stats": {
            "depths.depth_rel": {"q01": [0.0], "q99": [1.0]},
            "gripper_d": {"q01": [0.0], "q99": [1.0]},
        },
    }))
    preprocessor = LightweightInputPreprocessor(
        history_horizon=1,
        future_horizon=2,
        num_points=2,
        robot_cls=SixPointRobot,
        dataset_dir=tmp_path,
    )
    request = {
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.asarray([0.0] * 6 + [0.02, -0.02]),
        "camera.intrinsics": np.eye(3),
        "camera.extrinsics": np.eye(4),
    }
    session = InferenceSession(session_id="episode-1", benchmark="libero", language="test")

    model_input = preprocessor.build(request, session, {"nodes": []})

    actor_feats = np.asarray(model_input["actor_feats"], dtype=np.float32)
    assert actor_feats.shape == (1, 2, 1, 6, 8)
    np.testing.assert_allclose(
        actor_feats[0, 0, 0, :, 2],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.45],
    )
    assert model_input["object_id"] == [[0] * 12]
    assert model_input["point_id"] == [[0, 1, 2, 3, 4, 5] * 2]
    assert model_input["frame_id"] == [[1] * 6 + [2] * 6]
    assert model_input["actor_query_frame_id"] == [[1, 2]]

    input_object_id = np.asarray(model_input["input_object_id"])
    input_frame_id = np.asarray(model_input["input_frame_id"])
    assert set(input_object_id.reshape(-1).tolist()) == {0, 1, 2}
    assert set(input_frame_id.reshape(-1).tolist()) == {-1, 0}
    assert model_input["frame_query_frame_id"] == [[0]]
