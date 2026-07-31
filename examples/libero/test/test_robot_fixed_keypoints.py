from __future__ import annotations

import numpy as np
import pytest

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


def test_projected_fingertips_and_tcp_follow_gripper_width() -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    gripper_width = 0.03

    uvd = geometry.project_gripper_to_uvd(
        tcp_state=tcp_state,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
        gripper_width=gripper_width,
    )
    assert tuple(uvd) == (
        "root_uvd",
        "left_base_uvd",
        "right_base_uvd",
        "left_fingertip_uvd",
        "right_fingertip_uvd",
        "tcp_uvd",
    )

    points_camera = {}
    for name, (u, v, z) in uvd.items():
        points_camera[name] = np.asarray([
            (u - INTRINSIC[0, 2]) * z / INTRINSIC[0, 0],
            (v - INTRINSIC[1, 2]) * z / INTRINSIC[1, 1],
            z,
        ])

    root = points_camera["root_uvd"]
    left = points_camera["left_fingertip_uvd"]
    right = points_camera["right_fingertip_uvd"]
    tcp = points_camera["tcp_uvd"]
    np.testing.assert_allclose(
        left - root,
        [0.0, gripper_width / 2.0, geometry._FINGERTIP_CONTACT_Z],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        right - root,
        [0.0, -gripper_width / 2.0, geometry._FINGERTIP_CONTACT_Z],
        atol=1e-10,
    )
    np.testing.assert_allclose(tcp, (left + right) / 2.0, atol=1e-10)

    recovered = geometry.project_uvd_to_gripper(
        uvd,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
        gripper_width=gripper_width,
    )
    np.testing.assert_allclose(recovered[:6], tcp_state, atol=1e-7)


def test_projected_fingertip_width_is_clipped_and_nonfinite_width_is_rejected() -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    clipped = geometry.project_gripper_to_uvd(
        tcp_state=tcp_state,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
        gripper_width=1.0,
    )
    max_width = geometry.project_gripper_to_uvd(
        tcp_state=tcp_state,
        intrinsic=INTRINSIC,
        extrinsic=EXTRINSIC,
        gripper_width=geometry._MAX_GRIPPER_WIDTH,
    )
    np.testing.assert_allclose(clipped["left_fingertip_uvd"], max_width["left_fingertip_uvd"])
    np.testing.assert_allclose(clipped["right_fingertip_uvd"], max_width["right_fingertip_uvd"])

    with pytest.raises(ValueError, match="gripper_width must be finite"):
        geometry.project_gripper_to_uvd(
            tcp_state=tcp_state,
            intrinsic=INTRINSIC,
            extrinsic=EXTRINSIC,
            gripper_width=np.nan,
        )


@pytest.mark.parametrize("gripper_width", [0.0, 0.03, 0.08])
def test_six_xyz_keypoints_recover_pose(gripper_width: float) -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.08, -0.04, 1.1, 0.2, -0.1, 0.3], dtype=np.float64)

    xyz = geometry.project_gripper_to_xyz(
        tcp_state=tcp_state,
        extrinsic=EXTRINSIC,
        gripper_width=gripper_width,
    )
    recovered, residual = geometry.project_xyz_to_gripper(
        xyz,
        gripper_width=gripper_width,
        extrinsic=EXTRINSIC,
        return_residual=True,
    )

    assert xyz.shape == (6, 3)
    np.testing.assert_allclose(xyz[5], (xyz[3] + xyz[4]) / 2.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(xyz[3] - xyz[4]), gripper_width, atol=1e-12)
    np.testing.assert_allclose(recovered[:6], tcp_state, atol=1e-7)
    np.testing.assert_allclose(recovered[6], gripper_width, atol=1e-12)
    assert residual < 1e-12


@pytest.mark.parametrize("gripper_width", [0.01, 0.03, 0.08])
def test_actor_xyz_keypoints_recover_pose(gripper_width: float) -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.08, -0.04, 1.1, 0.2, -0.1, 0.3], dtype=np.float64)
    xyz = geometry.project_gripper_to_xyz(
        tcp_state=tcp_state,
        extrinsic=EXTRINSIC,
        gripper_width=gripper_width,
    )[[0, 3, 4]]

    recovered, residual = geometry.project_actor_xyz_to_gripper(
        xyz,
        gripper_width=np.linalg.norm(xyz[1] - xyz[2]),
        extrinsic=EXTRINSIC,
        return_residual=True,
    )

    np.testing.assert_allclose(recovered[:6], tcp_state, atol=1e-7)
    np.testing.assert_allclose(recovered[6], gripper_width, atol=1e-12)
    assert residual < 1e-12


def test_xyz_projection_rejects_invalid_width_and_point_count() -> None:
    geometry = GeomFrankaPanda()
    tcp_state = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="gripper_width must be finite"):
        geometry.project_gripper_to_xyz(
            tcp_state=tcp_state,
            extrinsic=EXTRINSIC,
            gripper_width=np.nan,
        )
    with pytest.raises(ValueError, match="six gripper XYZ keypoints"):
        geometry.project_xyz_to_gripper(
            np.zeros((3, 3)),
            gripper_width=0.03,
            extrinsic=EXTRINSIC,
        )
