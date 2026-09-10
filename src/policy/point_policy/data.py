"""TAPNext trajectory formatting and shared train/inference normalization."""

from dataclasses import dataclass
from pathlib import Path

import torch

from src.common.schema import LIBERO_GRIPPER_MAX_WIDTH
from src.dataset.transform import Normalize, Unnormalize


@dataclass
class PointPolicyTransform:
    norm_stats_path: str | Path
    history_horizon: int = 9
    future_horizon: int = 10
    actor_point_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    num_points: int = 32
    max_objects: int = 10

    def __post_init__(self):
        self.input_normalizer = Normalize(self.norm_stats_path, field_map={"point_tracks": "camera_xyz_track"},
                                          use_quantiles=True, quantile_to_neg_one_one=False)
        self.target_normalizer = Normalize(self.norm_stats_path, field_map={"target_points": "camera_xyz_track"},
                                           use_quantiles=True, quantile_to_neg_one_one=False)

    def observations(self, objects, object_mask, gripper):
        objects = torch.as_tensor(objects, dtype=torch.float32)
        object_mask = torch.as_tensor(object_mask, dtype=torch.bool)
        gripper = torch.as_tensor(gripper, dtype=torch.float32)
        if objects.shape != (self.history_horizon + 1, self.max_objects, self.num_points, 3):
            raise ValueError(f"Unexpected tracked object shape: {objects.shape}")
        if object_mask.shape != objects.shape[:-1]:
            raise ValueError("Tracked point mask shape does not match tracked points")
        if gripper.shape != (self.history_horizon + 1, 6, 3) or not torch.isfinite(gripper).all():
            raise ValueError("Expected finite gripper points [history,6,3]")
        actor = gripper[:, self.actor_point_indices]
        mask = object_mask & torch.isfinite(objects).all(-1) & (objects[..., 2] > 0)
        tracks = torch.cat((actor, objects.flatten(1, 2)), dim=1)
        point_mask = torch.cat((torch.ones(actor.shape[:-1], dtype=torch.bool), mask.flatten(1, 2)), dim=1)
        width = torch.linalg.vector_norm(gripper[:, 3] - gripper[:, 4], dim=-1)
        closedness = 1 - (width / LIBERO_GRIPPER_MAX_WIDTH).clamp(0, 1)
        result = {"point_tracks": tracks, "point_mask": point_mask,
                  "gripper_history": closedness[:, None]}
        result = self.input_normalizer(result)
        result["point_tracks"] = torch.where(point_mask[..., None], result["point_tracks"], 0.0)
        return result

    def __call__(self, data):
        h = self.history_horizon + 1
        objects = torch.as_tensor(data["node_points_xyz_track"], dtype=torch.float32)[:h]
        valid = torch.as_tensor(data["valid_node_mask"], dtype=torch.bool)[:h]
        gripper = torch.as_tensor(data["gripper_points_xyz"], dtype=torch.float32)
        result = self.observations(objects, valid[..., None].expand(*valid.shape, self.num_points), gripper[:h])
        result["language"] = str(data["language"])
        target = gripper[h:h + self.future_horizon, self.actor_point_indices]
        if target.shape[0] != self.future_horizon:
            raise ValueError("Missing future gripper points")
        result["target_points"] = target
        result = self.target_normalizer(result)
        # Pose at t+1 is paired with the command issued at t, including the gripper.
        actions = torch.as_tensor(data["action"], dtype=torch.float32)
        result["target_gripper"] = (actions[:, -1:] + 1) * 0.5
        result["target_mask"] = (~torch.as_tensor(data["gripper_points_xyz_is_pad"], dtype=torch.bool)[h:h + self.future_horizon]
                                 & ~torch.as_tensor(data["action_is_pad"], dtype=torch.bool))
        return result


@dataclass
class PointPolicyOutputTransform:
    norm_stats_path: str | Path

    def __call__(self, data):
        data["outputs"] = Unnormalize(self.norm_stats_path, field_map={"point_plan": "camera_xyz_track"},
                                      use_quantiles=True, quantile_to_neg_one_one=False)(data["outputs"])
        return data
