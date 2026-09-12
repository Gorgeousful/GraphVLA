import asyncio
import json
import struct
import threading

import numpy as np
import pytest
import websockets

from examples.libero.eval.client import InferenceClient, ObservationDeltaBuffer, _server_info
from script.server import InferenceServer, InputPreprocessor
from src.common.observation_wire import decode_observation, encode_observation


def test_array_roundtrip_preserves_values_dtype_shape_and_metadata():
    request = {
        "language": "put the bowl on the plate 碗", "reset": True, "worker_id": 2,
        "rgb": np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, ::2],
        "depth": np.array([0, np.nan, np.inf, -0.0], dtype=np.float32),
        "state": np.array([1.23456789012345], dtype=">f8"),
        "scalar": np.array(4, dtype=np.int32), "empty": np.empty((0, 3), np.float32),
    }
    restored = decode_observation(encode_observation(request))
    for key, value in request.items():
        if isinstance(value, np.ndarray):
            assert restored[key].dtype == value.dtype
            assert restored[key].shape == value.shape
            assert restored[key].tobytes() == value.tobytes()
            assert restored[key].flags.writeable
        else:
            assert restored[key] == value
    assert decode_observation('{"type":"server_info"}') == {"type": "server_info"}


def test_invalid_binary_messages_are_rejected():
    message = encode_observation({"state": np.ones((2, 8), np.float64)})
    for broken in (b"", message[:7], message[:12], message[:-1], message + b"x"):
        with pytest.raises(ValueError):
            decode_observation(broken)
    with pytest.raises(ValueError):
        encode_observation({"bad": np.array([object()], dtype=object)})
    header = json.dumps({"metadata": {}, "arrays": {"x": {"dtype": "O", "shape": [1]}}}).encode()
    with pytest.raises(ValueError):
        decode_observation(b"GVO1" + struct.pack("!I", len(header)) + header + bytes(8))


@pytest.mark.parametrize("policy,steps", [("graphpoint", None), ("point_policy", None),
                                         ("point_bridge", None), ("act", 1), ("dp", 2), ("dp3", 2)])
def test_policy_fields_and_history_preserve_model_observations(policy, steps):
    buffer = ObservationDeltaBuffer()
    for i in range(10):
        buffer.append({"agentview_image": np.full((4, 5, 3), i, np.uint8),
                       "wrist_image": np.full((4, 5, 3), i + 20, np.uint8),
                       "agentview_metric_depth": np.full((4, 5), i + 0.25, np.float32),
                       "state": np.full(8, i + 0.123456789, np.float64)})
    keys = ("observation.depth.metric", "camera.intrinsics", "observation.state") if policy == "dp3" else None
    fields = buffer.to_request_fields(intrinsic=np.eye(3), extrinsic=np.eye(4),
                                     image_obs_steps=steps, observation_keys=keys)
    restored = decode_observation(encode_observation(fields))
    expected_steps = steps or 10
    np.testing.assert_array_equal(restored["observation.state"], np.stack(buffer.states[-expected_steps:]))
    assert ("observation.images.wrist_image" in restored) == (policy in ("act", "dp"))
    if policy in ("act", "dp"):
        np.testing.assert_array_equal(restored["observation.images.image"], np.stack(buffer.images[-steps:]))
        np.testing.assert_array_equal(restored["observation.images.wrist_image"], np.stack(buffer.wrist_images[-steps:]))
    if policy == "dp3":
        assert set(restored) == set(keys)
        np.testing.assert_array_equal(restored["observation.depth.metric"], np.stack(buffer.metric_depths[-2:]))
    elif steps is None:
        # Compare actual GP/PP/PB frame parsing with the previous list/repeated-camera representation.
        legacy = {k: v.tolist() for k, v in fields.items()}
        legacy["camera.intrinsics"] = [np.eye(3).tolist()] * 10
        legacy["camera.extrinsics"] = [np.eye(4).tolist()] * 10
        preprocessor = object.__new__(InputPreprocessor)
        before = preprocessor._frames_from_request(legacy)
        after = preprocessor._frames_from_request(restored)
        assert len(before) == len(after) == 10
        for a, b in zip(before, after):
            for field in ("image", "metric_depth", "state", "intrinsic", "extrinsic"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_real_websocket_accepts_binary_observations_and_text_metadata():
    async def run():
        captured = []
        server = object.__new__(InferenceServer)
        server.session_locks = {}
        server.session_locks_guard = threading.Lock()
        server.server_info = lambda: {"policy_name": "point_policy"}
        server.infer_from_observation = lambda request: (captured.append(request) or
                                                       {"action": [[0.0] * 7], "episode_done": False})
        async with websockets.serve(server.handle_connection, "127.0.0.1", 0, max_size=None) as listener:
            uri = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
            info = await InferenceClient._websocket_json(uri, {"type": "server_info"})
            assert info["policy_name"] == "point_policy"
            state = np.arange(80, dtype=np.float64).reshape(10, 8)
            response = await InferenceClient._websocket_json(uri, {"session_id": "test", "observation.state": state})
            assert response["action"] == [[0.0] * 7]
            np.testing.assert_array_equal(captured[0]["observation.state"], state)
    asyncio.run(run())


def test_dp3_handshake_selects_depth_history(monkeypatch):
    async def info(uri, request):
        return {"ckpt_path": "step.pt", "progress_threshold": 0.9, "policy_name": "dp3",
                "obs_steps": 2, "observation_keys": ["observation.depth.metric", "camera.intrinsics", "observation.state"]}
    monkeypatch.setattr(InferenceClient, "_websocket_json", staticmethod(info))
    _, _, steps, keys = _server_info(host="localhost", port=8001)
    assert steps == 2
    assert keys == ("observation.depth.metric", "camera.intrinsics", "observation.state")
