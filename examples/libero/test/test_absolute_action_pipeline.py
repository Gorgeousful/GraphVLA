from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.dataset.pipeline import OfflinePipeline


def test_absolute_action_uses_tcp_pose_and_preserves_gripper_command() -> None:
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]
    world_rotation = R.from_euler("y", 45, degrees=True)
    frame = pd.DataFrame({
        "state": [[1.0, 4.0, 3.0, *world_rotation.as_rotvec(), 0.01, -0.01]],
        "actions": [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, -1.0]],
    })
    pipeline = object.__new__(OfflinePipeline)
    pipeline._load_libero_cameras = lambda: {
        5: {"agentview": {"extrinsic": camera_to_world.tolist()}},
    }

    action = np.asarray(pipeline._build_absolute_actions_camera(frame, 5)[0])

    np.testing.assert_allclose(action[:3], [2.0, 0.0, 0.0], atol=1e-6)
    expected_rotation = R.from_matrix(
        camera_to_world[:3, :3].T @ world_rotation.as_matrix()
    ).as_rotvec()
    np.testing.assert_allclose(action[3:6], expected_rotation, atol=1e-6)
    assert action[6] == -1.0
