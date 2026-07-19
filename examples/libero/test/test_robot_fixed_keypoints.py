from __future__ import annotations

import numpy as np

from examples.libero.embodiment.robot import GeomFrankaPanda


INTRINSIC = np.asarray(
    [
        [220.0, 0.0, 128.0],
        [0.0, 220.0, 128.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
EXTRINSIC = np.eye(4, dtype=np.float64)


def test_projected_pose_keypoints_are_fully_open_finger_bases() -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    uvd = geometry.project_gripper_to_uvd(
        tcp_state=tcp_state,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
    )
    assert tuple(uvd) == ("root_uvd", "left_base_uvd", "right_base_uvd")

    points_camera = []
    for value in uvd.values():
        u, v, z = value
        points_camera.append([
            (u - INTRINSIC[0, 2]) * z / INTRINSIC[0, 0],
            (v - INTRINSIC[1, 2]) * z / INTRINSIC[1, 1],
            z,
        ])
    root, left_base, right_base = np.asarray(points_camera)
    np.testing.assert_allclose(left_base - root, [0.0, 0.04, 0.0524], atol=1e-10)
    np.testing.assert_allclose(right_base - root, [0.0, -0.04, 0.0524], atol=1e-10)


def test_fixed_keypoints_recover_pose_with_independent_gripper_width() -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.08, -0.04, 1.1, 0.2, -0.1, 0.3], dtype=np.float64)
    expected_width = 0.023
    uvd = geometry.project_gripper_to_uvd(
        tcp_state=tcp_state,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
    )

    recovered = geometry.project_uvd_to_gripper(
        uvd,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
        gripper_width=expected_width,
    )

    np.testing.assert_allclose(recovered[:6], tcp_state, atol=1e-7)
    np.testing.assert_allclose(recovered[6], expected_width, atol=1e-12)
