from __future__ import annotations

import cv2
import numpy as np
import torch

from examples.libero.eval.client_offline import build_actor_diagnostics, build_observation_request, draw_point_grid


def test_build_observation_request_sends_only_history_in_server_image_coordinates() -> None:
    images = torch.zeros((4, 3, 2, 3), dtype=torch.float32)
    images[0, 0] = torch.tensor([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]])
    states = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    intrinsic = np.eye(3, dtype=np.float64)
    extrinsic = np.eye(4, dtype=np.float64)

    request = build_observation_request(
        images=images,
        states=states,
        prompt="pick up the mug",
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        history_horizon=1,
        execute_chunk_len=2,
        session_id="offline-sample-7",
    )

    assert request["benchmark"] == "libero"
    assert request["session_id"] == "offline-sample-7"
    assert request["language"] == "pick up the mug"
    assert request["execute_chunk_len"] == 2
    assert request["reset"] is True
    assert np.asarray(request["observation.images.image"]).shape == (2, 2, 3, 3)
    assert request["observation.images.image"][0][0][0] == [255, 0, 0]
    assert request["observation.state"] == states[:2].tolist()
    assert request["camera.intrinsics"] == [intrinsic.tolist(), intrinsic.tolist()]
    assert request["camera.extrinsics"] == [extrinsic.tolist(), extrinsic.tolist()]



def test_build_actor_diagnostics_compares_new_gripper_heads() -> None:
    response = {
        "point": [[
            [10.0, 20.0, 0.2, 1.0],
            [8.0, 20.0, 0.2, 1.0],
            [12.0, 20.0, 0.2, 1.0],
            [99.0, 99.0, 0.2, 1.0],
        ]],
        "metric_depth": [[[0.50], [0.40], [0.40], [0.40]]],
        "object_id": [[0, 0, 0, 1]],
        "point_id": [[0, 1, 2, 0]],
        "frame_id": [[1, 1, 1, 1]],
        "actor_query_frame_id": [[-1, 0, 1]],
        "gripper_openness": [[[0.1], [0.5], [0.7]]],
        "gripper_action": [[[0.0], [0.0], [1.4]]],
        "action": [[0.0] * 6 + [1.0]],
    }
    gt_gripper_uvd = np.zeros((3, 3, 3), dtype=np.float32)
    gt_gripper_uvd[2] = np.asarray(
        [[10.0, 20.0, 0.50], [9.0, 20.0, 0.40], [11.0, 20.0, 0.40]],
        dtype=np.float32,
    )
    gt_openness = np.asarray([0.2, 0.4, 0.6], dtype=np.float32)
    gt_action = np.zeros((3, 7), dtype=np.float32)
    gt_action[:, -1] = np.asarray([1.0, 0.0, 1.0])

    rows = build_actor_diagnostics(
        response=response,
        gt_gripper_uvd=gt_gripper_uvd,
        gt_gripper_openness=gt_openness,
        gt_action=gt_action,
        history_horizon=1,
    )

    assert len(rows) == 1
    assert rows[0]["frame_id"] == 1
    assert rows[0]["predicted_openness"] == 0.7
    assert rows[0]["target_openness"] == 0.6
    assert rows[0]["openness_abs_error"] == 0.1
    assert rows[0]["predicted_action"] == 1.4
    assert rows[0]["target_action"] == -1.0
    assert rows[0]["action_abs_error"] == 2.4
    assert rows[0]["executed_action"] == 1.0
    assert rows[0]["pred_uvd"]["left"] == [8.0, 20.0, 0.4]
    assert rows[0]["gt_uvd"]["right"] == [11.0, 20.0, 0.4]


def test_draw_point_grid_draws_actor_and_object_points(tmp_path) -> None:
    output_path = tmp_path / "tracking.png"
    images = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    draw_point_grid(
        output_path=output_path,
        images=images,
        points=np.asarray([[[10.0, 25.0, 0.5, 1.0], [20.0, 25.0, 0.5, 1.0]]]),
        object_id=np.asarray([[0, 1]]),
        point_frame_id=np.asarray([[0, 0]]),
        frame_ids=[0],
        history_horizon=1,
        label="tracking",
    )
    image = cv2.imread(str(output_path))
    assert image.shape == (128, 128, 3)
    actor_color = image[25, 10].tolist()
    object_color = image[25, 20].tolist()
    assert actor_color != [0, 0, 0]
    assert object_color != [0, 0, 0]
    assert actor_color != object_color
