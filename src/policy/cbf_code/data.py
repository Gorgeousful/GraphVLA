"""Convert LIBERO segmented entity points to the released CbF three-node contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class CbFCodeBatchTransform:
    num_points: int = 32
    obs_steps: int = 2
    horizon: int = 16

    def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
        points = torch.as_tensor(data["node_points_xyz"], dtype=torch.float32)
        valid = torch.as_tensor(data["valid_node_mask"], dtype=torch.bool)
        active = torch.as_tensor(data["subtask_node_mask"], dtype=torch.bool)
        gripper = torch.as_tensor(data["gripper_points_xyz"], dtype=torch.float32)
        state = torch.as_tensor(data["state"], dtype=torch.float32)
        actions = torch.as_tensor(data["action"], dtype=torch.float32)
        if points.ndim != 4 or points.shape[2:] != (self.num_points, 3):
            raise ValueError(f"Expected node_points_xyz [T,N,{self.num_points},3], got {tuple(points.shape)}")
        if valid.shape != points.shape[:2] or active.shape != points.shape[:2]:
            raise ValueError("valid_node_mask and subtask_node_mask must match [T,N]")
        if gripper.ndim != 3 or gripper.shape[1:] != (6, 3):
            raise ValueError(f"Expected gripper_points_xyz [T,6,3], got {tuple(gripper.shape)}")
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"Expected state [T,8], got {tuple(state.shape)}")
        if actions.shape != (self.horizon, 7):
            raise ValueError(f"Expected action [{self.horizon},7], got {tuple(actions.shape)}")
        current = int(data.get("history_horizon", self.obs_steps - 1))
        selected = torch.nonzero(active[current], as_tuple=False).flatten()
        roles = [
            str(node.get("role", "")) for node in data["subtaskstructure"].get("nodes", [])
            if isinstance(node, Mapping) and str(node.get("role", "")) != "actor"
        ]
        entities = points.new_zeros((self.obs_steps, 3, self.num_points, 3))
        # The dataset stores six analytic gripper points. Repeating them gives the
        # fixed-size actor cloud required by the shared CbF PointNet.
        entities[:, 0] = gripper[:self.obs_steps, torch.arange(self.num_points) % 6]
        for role_index, node_index in enumerate(selected[:len(roles)]):
            role = roles[role_index]
            slot = 1 if role == "patient" else (2 if role == "target" else role_index + 1)
            if slot >= 3:
                continue
            node_points = points[:self.obs_steps, node_index]
            node_valid = valid[:self.obs_steps, node_index] & active[:self.obs_steps, node_index]
            entities[:, slot] = torch.where(node_valid[:, None, None], node_points, 0.0)
        if not torch.isfinite(entities).all():
            raise ValueError("Segmented node point cloud contains non-finite coordinates")
        language = str(data["subtaskstructure"].get("subtask", data.get("language", "")))
        return {
            "node_point_clouds": entities,
            "state": state[:self.obs_steps],
            "actions": actions,
            "language": language,
        }
