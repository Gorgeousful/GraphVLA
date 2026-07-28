from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.common.schema import NodeRole
from src.dataset.pipeline import OfflinePipeline


def test_metric_backprojection_processes_all_tracked_nodes_and_repeats_valid_points() -> None:
    pipeline = object.__new__(OfflinePipeline)
    tracks = np.zeros((1, 2, 4, 3), dtype=np.float32)
    tracks[0, 0, :, :2] = [[2, 2], [3, 2], [20, 2], [2, 2]]
    tracks[0, 0, :2, 2] = 1.0
    tracks[0, 1, :, :2] = 2.0
    tracks[0, 1, :, 2] = 1.0
    depth = np.ones((1, 8, 8), dtype=np.float32) * 0.5
    intrinsic = np.asarray([[10.0, 0, 2.0], [0, 10.0, 2.0], [0, 0, 1.0]])
    xyz, valid_node_mask = pipeline._build_node_points_xyz(tracks, depth, intrinsic)

    assert valid_node_mask.tolist() == [[True, True]]
    np.testing.assert_allclose(xyz[0, 0, :, 0], [0.0, 0.05, 0.0, 0.05], atol=1e-7)
    np.testing.assert_allclose(xyz[0, 1], [[0.0, 0.0, 0.5]] * 4, atol=1e-7)


def test_metric_backprojection_disables_node_without_any_valid_depth() -> None:
    pipeline = object.__new__(OfflinePipeline)
    tracks = np.zeros((1, 1, 4, 3), dtype=np.float32)
    depth = np.ones((1, 8, 8), dtype=np.float32)
    xyz, valid_node_mask = pipeline._build_node_points_xyz(tracks, depth, np.eye(3))
    assert valid_node_mask.tolist() == [[False]]
    assert not xyz.any()


def test_subtask_node_mask_follows_flattened_taskstructure_order() -> None:
    pipeline = object.__new__(OfflinePipeline)
    pipeline.config = SimpleNamespace(dataset_type="libero", max_nodes=5)
    actor = SimpleNamespace(role=NodeRole.ACTOR)
    patient = SimpleNamespace(role=NodeRole.PATIENT)
    target = SimpleNamespace(role=NodeRole.TARGET)
    taskstructure = SimpleNamespace(subtask_list=[
        SimpleNamespace(node_list=[actor, patient, target]),
        SimpleNamespace(node_list=[actor, patient]),
    ])
    df = pd.DataFrame({"subtask_id": [1, 2]})

    mask = pipeline._build_subtask_node_mask(df, taskstructure)

    assert mask.tolist() == [
        [True, True, False, False, False],
        [False, False, True, False, False],
    ]
