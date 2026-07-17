from __future__ import annotations

import json

import numpy as np
import torch

from examples.libero.eval.client_offline import (
    build_observation_request,
    load_episode_observation_prefix,
    save_model_input_capture,
)
from examples.libero.test.expr0.run_expr0 import permute_patient_points
from script.server import InferenceModel


class FakePointModel:
    def infer(self, **inputs):
        return {"point": inputs["point_feats"].reshape(1, -1, inputs["point_feats"].shape[-1])}


def test_observation_request_can_warm_up_to_sample_without_future_leakage() -> None:
    images = np.zeros((37, 4, 4, 3), dtype=np.uint8)
    states = np.arange(37 * 8, dtype=np.float32).reshape(37, 8)
    for frame_index in range(len(images)):
        images[frame_index, :, :, 0] = frame_index

    request = build_observation_request(
        images=images,
        states=states,
        prompt="pick up the mug",
        intrinsic=np.eye(3),
        extrinsic=np.eye(4),
        history_horizon=15,
        observation_count=29,
        execute_chunk_len=15,
        session_id="test",
    )

    assert len(request["observation.images.image"]) == 29
    assert len(request["observation.state"]) == 29
    assert len(request["camera.intrinsics"]) == 29
    assert request["observation.state"][-1] == states[28].tolist()
    assert np.asarray(request["observation.images.image"])[-1].max() == 28


def test_episode_observation_prefix_stops_at_requested_sample() -> None:
    dataset = [
        {
            "observation.images.image": np.full((3, 2, 2), index, dtype=np.uint8),
            "observation.state": np.full(8, index, dtype=np.float32),
        }
        for index in range(37)
    ]

    images, states = load_episode_observation_prefix(dataset, list(range(37)), local_index=28)

    assert images.shape == (29, 3, 2, 2)
    assert states.shape == (29, 8)
    assert images[-1].max() == 28
    assert states[-1, 0] == 28


def test_inference_model_optionally_returns_exact_tensor_inputs() -> None:
    wrapper = InferenceModel.__new__(InferenceModel)
    wrapper.device = torch.device("cpu")
    wrapper.out_transforms = ()
    wrapper.model = FakePointModel()

    data = {
        "point_feats": np.zeros((1, 1, 1, 2, 6), dtype=np.float32),
        "actor_feats": np.zeros((1, 1, 1, 3, 6), dtype=np.float32),
        "object_condition": np.ones((1, 1, 4), dtype=np.float32),
        "actor_condition": np.ones((1, 1, 4), dtype=np.float32),
        "object_id": np.asarray([[0, 1]], dtype=np.int64),
        "point_id": np.asarray([[0, 0]], dtype=np.int64),
        "frame_id": np.asarray([[0, 0]], dtype=np.int64),
        "frame_query_frame_id": np.asarray([[0]], dtype=np.int64),
        "head_names": "point",
    }
    outputs, captured = wrapper.infer(data, return_model_input=True)

    assert "point" in outputs
    assert np.asarray(captured["point_feats"]).shape == (1, 1, 1, 2, 6)
    assert np.array_equal(captured["object_id"], data["object_id"])
    assert captured["head_names"] == "point"


def test_patient_permutation_is_fixed_across_frames_and_preserves_other_slot() -> None:
    points = torch.arange(1 * 3 * 2 * 4 * 2, dtype=torch.float32).reshape(1, 3, 2, 4, 2)
    permuted, permutation = permute_patient_points(points, seed=0)

    assert torch.equal(permuted[:, :, 1], points[:, :, 1])
    assert torch.equal(permuted[:, :, 0], points[:, :, 0, permutation])
    assert not torch.equal(permuted[:, :, 0], points[:, :, 0])


def test_capture_writer_saves_numeric_arrays_and_metadata(tmp_path) -> None:
    path = tmp_path / "capture.npz"
    save_model_input_capture(
        path,
        {
            "point_feats": [[[1.0, 2.0]]],
            "object_id": [[0]],
            "head_names": "point",
            "actor_condition_text": {"role": "actor"},
        },
    )

    with np.load(path) as capture:
        assert capture["point_feats"].shape == (1, 1, 2)
        assert capture["object_id"].dtype.kind in "iu"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["non_array"]["head_names"] == "point"
    assert metadata["non_array"]["actor_condition_text"]["role"] == "actor"
