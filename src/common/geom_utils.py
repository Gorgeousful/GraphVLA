"""Pure-function geometry utilities for mesh projection and rendering.

These functions are stateless and have no dependency on any robot class.
They operate on pre-transformed camera-space meshes.
"""

from typing import Iterable
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def uv_to_normalized_ray(uv: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates to camera rays ``[x_n, y_n]``."""
    uv = np.asarray(uv)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if uv.shape[-1] != 2 or intrinsic.shape != (3, 3):
        raise ValueError(f"Expected uv [...,2] and intrinsic [3,3], got {uv.shape}, {intrinsic.shape}")
    ray = uv.astype(np.float64, copy=True)
    ray[..., 0] = (ray[..., 0] - intrinsic[0, 2]) / intrinsic[0, 0]
    ray[..., 1] = (ray[..., 1] - intrinsic[1, 2]) / intrinsic[1, 1]
    return ray.astype(uv.dtype, copy=False)


def uv_to_normalized_ray_torch(uv, intrinsic):
    """Torch equivalent of :func:`uv_to_normalized_ray` preserving gradients."""
    import torch

    if not isinstance(uv, torch.Tensor) or not isinstance(intrinsic, torch.Tensor):
        raise TypeError("uv and intrinsic must be torch tensors")
    if uv.shape[-1] != 2 or intrinsic.shape != (3, 3):
        raise ValueError(f"Expected uv [...,2] and intrinsic [3,3], got {uv.shape}, {intrinsic.shape}")
    ray = uv.clone()
    ray[..., 0] = (uv[..., 0] - intrinsic[0, 2]) / intrinsic[0, 0]
    ray[..., 1] = (uv[..., 1] - intrinsic[1, 2]) / intrinsic[1, 1]
    return ray


def normalized_ray_depth_to_xyz(ray_depth: np.ndarray) -> np.ndarray:
    """Convert ``[x_n, y_n, z]`` camera rays to camera-space XYZ."""
    ray_depth = np.asarray(ray_depth)
    if ray_depth.shape[-1] != 3:
        raise ValueError(f"Expected ray_depth [...,3], got {ray_depth.shape}")
    xyz = ray_depth.copy()
    xyz[..., 0] *= ray_depth[..., 2]
    xyz[..., 1] *= ray_depth[..., 2]
    return xyz


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) or (N, 6) points."""
    points = np.asarray(points)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError(f"Expected points shape (N, 3) or (N, 6), got {points.shape}.")

    xyz = points[:, :3].astype(np.float64, copy=False)
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    transformed_xyz = (transform @ xyz_h.T).T[:, :3]
    if points.shape[1] == 3:
        return transformed_xyz.astype(points.dtype, copy=False)
    return np.concatenate([transformed_xyz, points[:, 3:]], axis=1).astype(
        points.dtype, copy=False
    )


def make_pose(position: Iterable[float], rotation_matrix: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous pose from position and 3x3 rotation matrix."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
    pose[:3, 3] = np.asarray(tuple(position), dtype=np.float64)
    return pose


def project_meshes_to_mask(
    meshes_camera: Iterable[tuple[np.ndarray, np.ndarray]],
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize camera-space triangle meshes into a binary mask via OpenCV."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    clip_margin = max(int(height), int(width)) * 8

    for vertices_camera, faces in meshes_camera:
        vertices_camera = np.asarray(vertices_camera, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if len(vertices_camera) == 0 or len(faces) == 0:
            continue

        for face in faces:
            triangle = vertices_camera[face]
            if np.any(triangle[:, 2] <= 1e-6):
                continue
            projected = (intrinsic @ triangle.T).T
            pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
            pixels[:, 0] = np.clip(pixels[:, 0], -clip_margin, width + clip_margin)
            pixels[:, 1] = np.clip(pixels[:, 1], -clip_margin, height + clip_margin)
            cv2.fillConvexPoly(mask, pixels, 1)

    return mask.astype(bool)


def project_meshes_to_depth(
    meshes_camera: Iterable[tuple[np.ndarray, np.ndarray]],
    intrinsic: np.ndarray,
    height: int,
    width: int,
    raster_context,
    device: str = "cuda:0",
) -> np.ndarray:
    """Render camera-space triangle meshes into a z-buffer depth map via nvdiffrast."""
    import torch
    import nvdiffrast.torch as dr

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    height = int(height)
    width = int(width)
    vertices_parts = []
    faces_parts = []
    vertex_offset = 0

    for vertices_camera, faces in meshes_camera:
        vertices_camera = np.asarray(vertices_camera, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        if len(vertices_camera) == 0 or len(faces) == 0:
            continue

        visible_faces = faces[
            np.all(vertices_camera[faces][:, :, 2] > 1e-6, axis=1)
        ]
        if len(visible_faces) == 0:
            continue
        vertices_parts.append(vertices_camera)
        faces_parts.append(visible_faces + vertex_offset)
        vertex_offset += len(vertices_camera)

    depth = np.full((height, width), np.inf, dtype=np.float32)
    if not vertices_parts or not faces_parts:
        return depth

    vertices = np.concatenate(vertices_parts, axis=0)
    faces = np.concatenate(faces_parts, axis=0).astype(np.int32, copy=False)
    z = vertices[:, 2].astype(np.float32, copy=False)
    near = float(np.min(z) - 1e-3)
    far = float(np.max(z) + 1e-3)
    if far <= near:
        far = near + 1e-3

    projected = (intrinsic @ vertices.astype(np.float64).T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    x_ndc = 2.0 * pixels[:, 0] / float(width - 1) - 1.0
    y_ndc = 2.0 * pixels[:, 1] / float(height - 1) - 1.0
    z_ndc = 2.0 * (z.astype(np.float64) - near) / (far - near) - 1.0
    clip = np.stack(
        [x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=1
    ).astype(np.float32)

    torch_device = torch.device(device)
    with torch.no_grad():
        pos = torch.as_tensor(clip, dtype=torch.float32, device=torch_device)[None]
        tri = torch.as_tensor(faces, dtype=torch.int32, device=torch_device)
        attr = torch.as_tensor(z[:, None], dtype=torch.float32, device=torch_device)[
            None
        ]
        rast, _ = dr.rasterize(
            raster_context, pos, tri, resolution=[height, width], grad_db=False
        )
        depth_attr, _ = dr.interpolate(attr, rast, tri)
        mask = (rast[0, :, :, 3] > 0).detach().cpu().numpy()
        values = (
            depth_attr[0, :, :, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        )

    depth[mask] = values[mask]
    return depth


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    """Extract [x_min, y_min, x_max, y_max] bounding box from a binary mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def sample_mesh_pcd(meshes, n_points: int, seed: int = 0) -> np.ndarray:
    """Sample *n_points* from mesh surfaces via area-weighted random + FPS.

    Triangles are sampled proportionally to their area so that regions with
    larger (or fewer) triangles still receive a fair share of points.
    Candidate points are then filtered through Farthest Point Sampling for
    good spatial coverage.

    Args:
        meshes: list of dicts with "vertices" (N,3) and "faces" (M,3),
                or a single (V,3) vertices + (F,3) faces pair.
        n_points: number of points to sample.
        seed: random seed for reproducibility.

    Returns:
        (n_points, 3) point cloud in the same frame as input vertices.
    """
    # Normalize input
    if isinstance(meshes, tuple) and len(meshes) == 2:
        meshes = [{"vertices": meshes[0], "faces": meshes[1]}]

    rng = np.random.default_rng(seed)

    # 1. Collect all triangles and their areas
    all_verts = []
    all_faces = []
    offset = 0
    for mesh in meshes:
        v = np.asarray(mesh["vertices"], dtype=np.float64)
        f = np.asarray(mesh["faces"], dtype=np.int64)
        if len(v) == 0 or len(f) == 0:
            continue
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float64)

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(np.sum(areas))
    if not np.isfinite(total_area) or total_area < 1e-12:
        return np.zeros((0, 3), dtype=np.float64)

    # 2. Sample triangles proportional to area (~20x oversampling for FPS)
    n_cand = max(n_points * 20, n_points + 1)
    probs = areas / total_area
    tri_idx = rng.choice(len(faces), size=n_cand, replace=True, p=probs)

    a = v0[tri_idx]
    b = v1[tri_idx]
    c = v2[tri_idx]

    # 3. Uniform random points on each selected triangle (sqrt trick)
    r1 = rng.random(n_cand)
    r2 = rng.random(n_cand)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2
    candidates = w0[:, None] * a + w1[:, None] * b + w2[:, None] * c

    # 4. Farthest Point Sampling for spatial coverage
    n_select = min(n_points, len(candidates))
    selected = [0]
    min_dist = np.full(len(candidates), np.inf)
    for _ in range(n_select - 1):
        last = candidates[selected[-1]]
        dist = np.sum((candidates - last) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        selected.append(int(np.argmax(min_dist)))

    return candidates[selected].astype(np.float64)


def rot_transform(input_data, input_format="rot6d", target_format="matrix") -> np.ndarray:
    """将旋转表示法在不同格式之间转换。

    Args:
        input_data: 输入旋转数据
        input_format: 输入格式，可选 "quat", "matrix", "rot6d", "euler", "axis_angle"
        target_format: 目标格式，可选 "quat", "matrix", "rot6d", "euler", "axis_angle"
            quat 格式为 (x, y, z, w)
            euler 格式为 (roll, pitch, yaw) 以弧度表示
            axis_angle 格式为 (ax, ay, az) 轴角表示，向量方向为旋转轴，长度为旋转角度（弧度）

    Returns:
        转换后的旋转表示
    """
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    # 根据输入格式转换为旋转矩阵
    if input_format == "rot6d":
        if input_data.shape != (6,):
            raise ValueError(f"rot6d 应该是6维向量，但得到的是 {input_data.shape}")

        a1, a2 = input_data[:3], input_data[3:]
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        rotation_matrix = np.column_stack([b1, b2, b3])

    elif input_format == "quat":
        if input_data.shape != (4,):
            raise ValueError(f"四元数应该是4维向量，但得到的是 {input_data.shape}")
        rotation_matrix = R.from_quat(input_data).as_matrix()

    elif input_format == "matrix":
        if input_data.shape != (3, 3):
            raise ValueError(f"旋转矩阵应该是3x3矩阵，但得到的是 {input_data.shape}")
        rotation_matrix = input_data

    elif input_format == "euler":
        if input_data.shape != (3,):
            raise ValueError(f"欧拉角应该是3维向量，但得到的是 {input_data.shape}")
        rotation_matrix = R.from_euler('xyz', input_data).as_matrix()

    elif input_format == "axis_angle":
        if input_data.shape != (3,):
            raise ValueError(f"轴角应该是3维向量，但得到的是 {input_data.shape}")
        rotation_matrix = R.from_rotvec(input_data).as_matrix()

    else:
        raise ValueError(f"不支持的输入格式: {input_format}。支持 'quat', 'matrix', 'rot6d', 'euler', 'axis_angle'")

    # 根据目标格式进行转换
    if target_format == "matrix":
        return rotation_matrix

    elif target_format == "quat":
        return R.from_matrix(rotation_matrix).as_quat()

    elif target_format == "euler":
        return R.from_matrix(rotation_matrix).as_euler('xyz')

    elif target_format == "rot6d":
        col1 = rotation_matrix[:, 0]
        col2 = rotation_matrix[:, 1]
        return np.concatenate([col1, col2])

    elif target_format == "axis_angle":
        return R.from_matrix(rotation_matrix).as_rotvec()

    else:
        raise ValueError(f"不支持的输出格式: {target_format}。支持 'quat', 'matrix', 'rot6d', 'euler', 'axis_angle'")


def sample_points_from_mask(mask: np.ndarray, num_points: int = 128, erode_pixel: int = 0) -> np.ndarray:
    """Sample points from a binary mask using farthest point sampling.

    Args:
        mask: np.ndarray, shape (H, W), non-zero values are valid pixels.
        num_points: Number of points to return.
        erode_pixel: Pixels to erode before sampling. Falls back to the original
            mask if erosion removes all valid pixels.

    Returns:
        np.ndarray, shape (num_points, 2), float32, coordinates in (x, y).
    """
    mask = np.asarray(mask, dtype=bool)
    if erode_pixel > 0 and mask.any():
        kernel = np.ones((2 * int(erode_pixel) + 1, 2 * int(erode_pixel) + 1), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        if eroded.any():
            mask = eroded

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("mask contains no valid pixels")

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(points) <= num_points:
        repeats = int(np.ceil(num_points / len(points)))
        return np.tile(points, (repeats, 1))[:num_points]

    center = points.mean(axis=0, keepdims=True)
    first_idx = np.argmin(np.sum((points - center) ** 2, axis=1))
    selected = np.empty((num_points, 2), dtype=np.float32)
    selected[0] = points[first_idx]

    min_dist_sq = np.sum((points - selected[0]) ** 2, axis=1)
    for i in range(1, num_points):
        idx = np.argmax(min_dist_sq)
        selected[i] = points[idx]
        dist_sq = np.sum((points - selected[i]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return selected
