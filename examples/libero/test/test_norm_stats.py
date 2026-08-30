from __future__ import annotations

import numpy as np

from src.common.schema import GRIPPER_TCP_POINT_INDEX
from src.dataset.lerobot_compute_norm_stats import _extract_tcp_relative_xyz_by_sample


def test_tcp_relative_stats_use_current_tcp_for_history_and_future() -> None:
    gripper = np.zeros((1, 4, 6, 3), dtype=np.float32)
    gripper[0, :, :, 0] = np.arange(4, dtype=np.float32)[:, None] * 10.0
    gripper[0, :, :, 1] = np.arange(6, dtype=np.float32)[None]
    nodes = np.zeros((1, 4, 2, 2, 3), dtype=np.float32)
    nodes[0, :, 0, :, 0] = np.arange(4, dtype=np.float32)[:, None] * 10.0 + 2.0
    nodes[0, :, 1] = 999.0
    valid = np.zeros((1, 4, 2), dtype=bool)
    valid[:, :, 0] = True

    values = _extract_tcp_relative_xyz_by_sample(
        {
            "node_points_xyz": nodes,
            "valid_node_mask": valid,
            "subtask_node_mask": valid.copy(),
            "gripper_points_xyz": gripper,
            "subtask_id": np.ones((1, 4), dtype=np.int64),
        },
        current_index=1,
    )[0]

    origin = gripper[0, 1, GRIPPER_TCP_POINT_INDEX]
    expected_nodes = (nodes[0, :2, 0] - origin).reshape(-1, 3)
    expected_gripper = (gripper[0] - origin).reshape(-1, 3)
    np.testing.assert_allclose(values, np.concatenate([expected_nodes, expected_gripper]))
    assert not np.any(np.all(values == 999.0 - origin, axis=-1))
    np.testing.assert_allclose(
        values[len(expected_nodes) + 6 + GRIPPER_TCP_POINT_INDEX], np.zeros(3),
    )


def test_tcp_relative_stats_apply_subtask_boundary_padding_after_centering() -> None:
    gripper = np.zeros((1, 4, 6, 3), dtype=np.float32)
    gripper[0, :, :, 0] = np.asarray([0.0, 10.0, 20.0, 100.0])[:, None]
    nodes = np.zeros((1, 4, 1, 1, 3), dtype=np.float32)
    nodes[0, :, 0, 0, 0] = np.asarray([1.0, 11.0, 21.0, 101.0])
    valid = np.ones((1, 4, 1), dtype=bool)

    values = _extract_tcp_relative_xyz_by_sample(
        {
            "node_points_xyz": nodes,
            "valid_node_mask": valid,
            "subtask_node_mask": valid.copy(),
            "gripper_points_xyz": gripper,
            "subtask_id": np.asarray([[0, 1, 1, 2]], dtype=np.int64),
        },
        current_index=1,
    )[0]

    expected_indices = np.asarray([1, 1, 2, 2])
    origin = gripper[0, 1, GRIPPER_TCP_POINT_INDEX]
    expected_nodes = (nodes[0, expected_indices[:2]] - origin).reshape(-1, 3)
    expected_gripper = (gripper[0, expected_indices] - origin).reshape(-1, 3)
    np.testing.assert_allclose(values, np.concatenate([expected_nodes, expected_gripper]))


def test_tcp_relative_stats_exclude_nodes_outside_current_subtask() -> None:
    gripper = np.zeros((1, 2, 6, 3), dtype=np.float32)
    nodes = np.zeros((1, 2, 2, 1, 3), dtype=np.float32)
    nodes[:, :, 0] = 1.0
    nodes[:, :, 1] = 999.0
    valid = np.ones((1, 2, 2), dtype=bool)
    active = np.zeros((1, 2, 2), dtype=bool)
    active[:, :, 0] = True

    values = _extract_tcp_relative_xyz_by_sample(
        {
            "node_points_xyz": nodes,
            "valid_node_mask": valid,
            "subtask_node_mask": active,
            "gripper_points_xyz": gripper,
            "subtask_id": np.zeros((1, 2), dtype=np.int64),
        },
        current_index=1,
    )[0]

    assert not np.any(np.all(values == 999.0, axis=-1))
