"""Shared SAM observations and aligned future measured pose/point targets for LIBERO."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.dataset.transform import Normalize, Unnormalize


@dataclass
class PointBridgeTransform:
    norm_stats_path: str | Path
    dataset_dir: str | Path
    action_mode: str = "pose"
    history_horizon: int = 0
    future_horizon: int = 10
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    num_points: int = 32
    max_objects: int = 10

    def __post_init__(self):
        if self.action_mode not in ("pose", "points"):
            raise ValueError("action_mode must be 'pose' or 'points'")
        self.normalizer = Normalize(
            self.norm_stats_path, use_quantiles=True, quantile_to_neg_one_one=False,
            field_map={"robot_points": "camera_xyz", "object_points": "camera_xyz"})
        self.target_normalizer = Normalize(
            self.norm_stats_path, use_quantiles=True, quantile_to_neg_one_one=False,
            field_map={"target_xyz": "camera_xyz"})
        self.cameras = None

    def observations(self, objects, object_mask, gripper):
        objects = torch.as_tensor(objects, dtype=torch.float32)
        mask = torch.as_tensor(object_mask, dtype=torch.bool)
        gripper = torch.as_tensor(gripper, dtype=torch.float32)
        h = self.history_horizon + 1
        if objects.shape != (h, self.max_objects, self.num_points, 3) or mask.shape != objects.shape[:-1]:
            raise ValueError("Unexpected SAM object point/history shape or mask")
        if gripper.shape != (h, 6, 3) or not torch.isfinite(gripper).all():
            raise ValueError("Expected finite gripper points [history,6,3]")
        mask = mask & torch.isfinite(objects).all(-1) & (objects[..., 2] > 0)
        result = self.normalizer({"object_points": torch.where(mask[..., None], objects, 0.0),
                                  "robot_points": gripper[:, self.actor_point_indices]})
        result["object_points"] = torch.where(mask[..., None], result["object_points"], 0.0)
        result["object_mask"] = mask
        return result

    def __call__(self, data):
        h, q = self.history_horizon + 1, self.future_horizon
        objects = torch.as_tensor(data["node_points_xyz"], dtype=torch.float32)
        valid = torch.as_tensor(data["valid_node_mask"], dtype=torch.bool)
        gripper = torch.as_tensor(data["gripper_points_xyz"], dtype=torch.float32)
        result = self.observations(objects[:h], valid[:h, :, None].expand(-1, -1, self.num_points), gripper[:h])
        result["language"] = str(data["language"])
        actions = torch.as_tensor(data["action"], dtype=torch.float32)
        grip = (actions[:, -1:] + 1) * 0.5
        mask = ~torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
        if self.action_mode == "points":
            xyz = gripper[h:h + q, self.actor_point_indices]
            mask &= ~torch.as_tensor(data["gripper_points_xyz_is_pad"], dtype=torch.bool)[h:h + q]
            xyz = self.target_normalizer({"target_xyz": xyz})["target_xyz"]
            target = torch.cat((xyz.flatten(1), grip.expand(-1, 3)), dim=-1)
        else:
            # observation.state holds measured world-frame TCP poses at t+1,...,t+Q.
            states = torch.as_tensor(data["observation.state"], dtype=torch.float32).numpy()
            mask &= ~torch.as_tensor(data["observation.state_is_pad"], dtype=torch.bool)
            if self.cameras is None:
                rows = json.loads((Path(self.dataset_dir) / "meta" / "cameras.json").read_text())
                self.cameras = {int(row["task_index"]): np.asarray(row["cameras"]["agentview"]["extrinsic"])
                                for row in rows}
            camera_from_world = np.linalg.inv(self.cameras[int(torch.as_tensor(data["task_index"]).item())])
            xyz = states[:, :3] @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
            rotation = camera_from_world[:3, :3] @ Rotation.from_rotvec(states[:, 3:6]).as_matrix()
            # Official continuous 6D rotation convention: first two matrix rows.
            rot6 = torch.as_tensor(rotation[:, :2, :].reshape(-1, 6), dtype=torch.float32)
            xyz = self.target_normalizer({"target_xyz": torch.as_tensor(xyz, dtype=torch.float32)})["target_xyz"]
            target = torch.cat((xyz, (rot6 + 1) * 0.5, grip), dim=-1)
        if target.shape[0] != q or mask.shape != (q,):
            raise ValueError("Missing future pose/point targets")
        result["target_actions"] = target
        result["target_mask"] = mask
        return result


@dataclass
class PointBridgeOutputTransform:
    norm_stats_path: str | Path
    action_mode: str = "pose"

    def __call__(self, data):
        outputs = data["outputs"]
        unnormalize = Unnormalize(self.norm_stats_path, field_map={"xyz": "camera_xyz"},
                                 use_quantiles=True, quantile_to_neg_one_one=False)
        if self.action_mode == "points":
            outputs["point_plan"] = unnormalize({"xyz": outputs["point_plan"]})["xyz"]
        elif self.action_mode == "pose":
            pose = outputs["pose_plan"].clone()
            pose[..., :3] = unnormalize({"xyz": pose[..., :3]})["xyz"]
            pose[..., 3:9] = 2 * pose[..., 3:9] - 1
            pose[..., 9] = (2 * pose[..., 9] - 1).clamp(-1, 1)
            outputs["pose_plan"] = pose
        else:
            raise ValueError("action_mode must be 'pose' or 'points'")
        return data


def camera_pose_to_world_action(pose, extrinsic):
    """Decode camera XYZ + row-6D rotation + gripper into absolute LIBERO actions."""
    pose = np.asarray(pose, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[-1] != 10 or not np.isfinite(pose).all():
        raise ValueError("Expected finite camera pose [Q,10]")
    first, second = pose[:, 3:6], pose[:, 6:9]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - (first * second).sum(-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    rotation = np.stack((first, second, np.cross(first, second)), axis=-2)
    world_rotation = extrinsic[:3, :3] @ rotation
    xyz = pose[:, :3] @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    return np.concatenate((xyz, Rotation.from_matrix(world_rotation).as_rotvec(),
                           np.clip(pose[:, -1:], -1, 1)), axis=-1)
