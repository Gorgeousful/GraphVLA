"""Bridge an OpenPI policy to the shared LIBERO oracle evaluation protocol."""

import argparse

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from script.server import DP3InferenceServer, EmbodimentAdapter, InferenceServer, TopLevelTaskPlanner


class Pi05InferenceServer(DP3InferenceServer):
    def __init__(self, args):
        InferenceServer.__init__(
            self, host="0.0.0.0", port=args.port,
            planner=TopLevelTaskPlanner(dataset_dir=args.dataset_dir),
            execute_chunk_len=10, preprocessor=None, inference=None,
            embodiment=EmbodimentAdapter(future_horizon=10, actor_point_indices=(0,),
                                        action_mode="delta_action", action_delta=True),
            ckpt_path=args.ckpt_path,
        )
        self.policy_name = "pi05"
        self.obs_steps = 1
        self.camera_keys = ("observation.images.image", "observation.images.wrist_image")
        self.upstream = websocket_client_policy.WebsocketClientPolicy(args.upstream_host, args.upstream_port)

    def _predict_action(self, request):
        payload = {"prompt": str(request["language"])}
        for source, destination in zip(self.camera_keys, ("observation/image", "observation/wrist_image")):
            value = np.asarray(request[source], dtype=np.uint8)
            if value.ndim == 4:
                value = value[-1]
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"Invalid RGB image: {source} {value.shape}")
            payload[destination] = image_tools.convert_to_uint8(image_tools.resize_with_pad(value, 224, 224))
        state = np.asarray(request["observation.state"], dtype=np.float32)
        payload["observation/state"] = state[-1] if state.ndim == 2 else state
        if payload["observation/state"].shape != (8,):
            raise ValueError(f"Invalid robot state: {state.shape}")
        actions = np.asarray(self.upstream.infer(payload)["actions"])
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 10 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid OpenPI actions: {actions.shape}")
        return {"action": actions[:10].tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8010)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--dataset-dir", default="/data0/luokang/dataset/luokang/lerobot/libero/libero_custom_0902_20hz")
    Pi05InferenceServer(parser.parse_args()).serve_forever()
