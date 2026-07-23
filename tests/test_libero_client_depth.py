from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.libero.eval import client


def test_prepare_observation_squeezes_metric_depth_for_server(monkeypatch) -> None:
    metric_depth = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    monkeypatch.setattr(client, "get_real_depth_map", lambda *_: metric_depth)
    observation = {
        "agentview_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "agentview_depth": np.zeros((2, 3, 1), dtype=np.float32),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }

    prepared = client._prepare_observation(observation, SimpleNamespace(sim=object()))
    assert prepared["agentview_metric_depth"].shape == (2, 3)
    np.testing.assert_array_equal(prepared["agentview_metric_depth"], metric_depth[::-1, :, 0])

    buffer = client.ObservationDeltaBuffer()
    buffer.append(prepared)
    request = buffer.to_request_fields(intrinsic=np.eye(3), extrinsic=np.eye(4))
    assert np.asarray(request["observation.depth.metric"]).shape == (1, 2, 3)


def test_inference_client_records_server_episode_done(monkeypatch) -> None:
    inference_client = client.InferenceClient(host="localhost", port=0)
    monkeypatch.setattr(
        inference_client,
        "_call_server",
        lambda _: {
            "episode_done": True,
            "action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
        },
    )
    observation = {
        "agentview_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "agentview_metric_depth": np.ones((2, 3), dtype=np.float32),
        "state": np.zeros(8, dtype=np.float64),
    }

    inference_client.infer(observation, "task")

    assert inference_client.episode_done is True
