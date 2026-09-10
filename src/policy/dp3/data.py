"""Observation-history inputs and the dataset's camera-frame FPS recipe."""

from dataclasses import dataclass
import numpy as np
import torch


@torch.inference_mode()
def depth_to_point_cloud(depth, intrinsic, *, num_points=512, max_candidates=8192, device="cpu"):
    """Match lerobot_add_scene_pcd.py at a953944: candidate linspace, then FPS."""
    from pytorch3d.ops import sample_farthest_points

    depth = torch.as_tensor(np.asarray(depth).copy(), dtype=torch.float32, device=device)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if depth.ndim != 2 or intrinsic.shape != (3, 3):
        raise ValueError("Expected metric depth [H,W] and intrinsic [3,3]")
    if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    height, width = depth.shape
    total = height * width
    indices = (torch.arange(total, device=device) if max_candidates <= 0 or max_candidates >= total
               else torch.linspace(0, total - 1, steps=max_candidates, device=device).round().long())
    u = (indices % width).float()
    v = torch.div(indices, width, rounding_mode="floor").float()
    # Python scalars match the source script's CUDA arithmetic (and FPS tie breaks).
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    rays = torch.stack(((u - cx) / fx, (v - cy) / fy, torch.ones_like(u)), -1)
    z = depth.flatten()[indices]
    valid = torch.isfinite(z) & (z > 0)
    points = (rays * z[:, None])[valid]
    if num_points <= 0 or len(points) < num_points:
        raise ValueError(f"Need {num_points} valid depth candidates, got {len(points)}")
    sampled, _ = sample_farthest_points(points[None], K=num_points, random_start_point=False)
    return sampled[0]


@dataclass
class DP3BatchTransform:
    num_points: int = 512
    obs_steps: int = 2

    def __call__(self, data):
        points = torch.as_tensor(data["observation.point_cloud"], dtype=torch.float32)
        state = torch.as_tensor(data["observation.state"], dtype=torch.float32)
        actions = torch.as_tensor(data["action"], dtype=torch.float32)
        is_pad = torch.as_tensor(data["action_is_pad"], dtype=torch.bool)
        if points.shape != (self.obs_steps, self.num_points, 3) or state.shape != (self.obs_steps, 8):
            raise ValueError("DP3 requires point_cloud [T,N,3] and state [T,8]")
        if not torch.isfinite(points).all():
            raise ValueError("Point cloud contains non-finite coordinates")
        if actions.ndim != 2 or actions.shape[-1] != 7 or is_pad.shape != actions.shape[:1]:
            raise ValueError("Expected actions [H,7] and is_pad [H]")
        return {"point_cloud": points, "state": state, "actions": actions,
                "is_pad": is_pad, "language": str(data["language"])}
