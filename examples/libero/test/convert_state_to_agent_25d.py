"""Convert LIBERO robot states to structured two-finger gripper 2.5D state.

Output arrays are saved as an ``.npz`` with one row per frame:
  center_uvd, root_uvd, open_axis_2p5d, width_2p5d,
  left_tip_uvd, right_tip_uvd, states, episode_index, frame_index.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from src.common.schema import LIBERO_GRIPPER_MAX_WIDTH


DEFAULT_DATASET_ROOT = Path(
    "/data0/luokang/dataset/luokang/lerobot/libero/"
    "libero_all_no_noops_1.0.0_lerobot_10hz"
)
DEFAULT_ROBOT_XML = Path(
    "/data0/luokang/research/GraphVLA/examples/libero/embodiment/"
    "franka_panda/robot.xml"
)
DEFAULT_FINGER_STL = DEFAULT_ROBOT_XML.parent / "meshes" / "panda_gripper" / "finger_vis.stl"
TCP_TO_HAND_OFFSET = np.array([0.0, 0.0, -0.097], dtype=np.float64)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_episode(root: Path, task_index: int, trajectory_index: int) -> dict:
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    task = next(item for item in tasks if item["task_index"] == task_index)
    matches = [ep for ep in load_jsonl(root / "meta" / "episodes.jsonl") if task["task"] in ep["tasks"]]
    if trajectory_index >= len(matches):
        raise IndexError(f"task {task_index} has {len(matches)} trajectories, got {trajectory_index}")
    return matches[trajectory_index]


def episode_data_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def load_camera(root: Path, task_index: int, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    cameras = json.loads((root / "meta" / "cameras.json").read_text(encoding="utf-8"))
    record = next(item for item in cameras if item["task_index"] == task_index)
    camera = record["cameras"][camera_name]
    return np.asarray(camera["intrinsic"], dtype=np.float64), np.asarray(camera["extrinsic"], dtype=np.float64)


def load_states(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(parquet_path, columns=["observation.state", "frame_index"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    return states, frame_index


def quat_wxyz_to_matrix(quat: list[float]) -> np.ndarray:
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def make_pose(pos: list[float] | np.ndarray, quat_wxyz: list[float] | None = None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    if quat_wxyz is not None:
        pose[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    return pose


def load_binary_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n_triangles = struct.unpack("<I", data[80:84])[0]
    vertices = []
    offset = 84
    for _ in range(n_triangles):
        offset += 12
        tri = struct.unpack("<9f", data[offset : offset + 36])
        vertices.extend(np.asarray(tri, dtype=np.float64).reshape(3, 3))
        offset += 38
    return np.asarray(vertices, dtype=np.float64)


def finger_mesh_tip(path: Path) -> np.ndarray:
    vertices = load_binary_stl_vertices(path)
    z_max = vertices[:, 2].max()
    return vertices[np.isclose(vertices[:, 2], z_max, atol=1e-5)].mean(axis=0)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (transform @ np.append(point, 1.0))[:3]


def local_tip_points(opening: float, mesh_tip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opening = float(np.clip(opening, 0.0, LIBERO_GRIPPER_MAX_WIDTH))
    half_open = opening / 2.0

    hand_from_gripper = make_pose([0.0, 0.0, 0.0], [0.707107, 0.0, 0.0, -0.707107])
    gripper_from_finger = make_pose([0.0, 0.0, 0.0524], [0.707107, 0.0, 0.0, 0.707107])
    right_finger_geom = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    left_slide = make_pose([0.0, half_open, 0.0])
    right_slide = make_pose([0.0, -half_open, 0.0])

    left_tip = transform_point(hand_from_gripper @ gripper_from_finger @ left_slide, mesh_tip)
    right_tip = transform_point(
        hand_from_gripper @ gripper_from_finger @ right_slide @ right_finger_geom, mesh_tip
    )
    root = np.zeros(3, dtype=np.float64)
    return left_tip, right_tip, root


def project_camera_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pixels_h = (intrinsic @ points_camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return np.concatenate([pixels, points_camera[:, 2:3]], axis=1)


def tcp_state_to_hand_pose(state: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = state[:3]
    pose[:3, :3] = R.from_rotvec(state[3:6]).as_matrix()
    offset = np.eye(4, dtype=np.float64)
    offset[:3, 3] = TCP_TO_HAND_OFFSET
    return pose @ offset


def convert_states(states: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, finger_stl: Path) -> dict[str, np.ndarray]:
    mesh_tip = finger_mesh_tip(finger_stl)
    camera_from_world = np.linalg.inv(extrinsic)
    left_uvd, right_uvd, root_uvd = [], [], []

    for state in states:
        opening = abs(state[6]) + abs(state[7])
        left_local, right_local, root_local = local_tip_points(opening, mesh_tip)
        camera_from_hand = camera_from_world @ tcp_state_to_hand_pose(state)
        points_camera = np.stack([
            transform_point(camera_from_hand, left_local),
            transform_point(camera_from_hand, right_local),
            transform_point(camera_from_hand, root_local),
        ])
        uvd = project_camera_points(points_camera, intrinsic)
        left_uvd.append(uvd[0])
        right_uvd.append(uvd[1])
        root_uvd.append(uvd[2])

    left_uvd = np.asarray(left_uvd, dtype=np.float64)
    right_uvd = np.asarray(right_uvd, dtype=np.float64)
    root_uvd = np.asarray(root_uvd, dtype=np.float64)
    center_uvd = 0.5 * (left_uvd + right_uvd)
    delta = right_uvd - left_uvd
    width = np.linalg.norm(delta, axis=1)
    open_axis = delta / np.maximum(width[:, None], 1e-8)
    return {
        "center_uvd": center_uvd,
        "root_uvd": root_uvd,
        "open_axis_2p5d": open_axis,
        "width_2p5d": width,
        "left_tip_uvd": left_uvd,
        "right_tip_uvd": right_uvd,
        "gripper_opening": np.clip(np.abs(states[:, 6]) + np.abs(states[:, 7]), 0.0, LIBERO_GRIPPER_MAX_WIDTH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task-index", type=int, default=31)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--finger-stl", type=Path, default=DEFAULT_FINGER_STL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GraphVLA/examples/libero/test/task31_traj0_agent_state_25d.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = find_episode(args.dataset_root, args.task_index, args.trajectory_index)
    episode_index = int(episode["episode_index"])
    states, frame_index = load_states(episode_data_path(args.dataset_root, episode_index))
    intrinsic, extrinsic = load_camera(args.dataset_root, args.task_index, args.camera_name)
    agent_state = convert_states(states, intrinsic, extrinsic, args.finger_stl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **agent_state,
        states=states,
        frame_index=frame_index,
        episode_index=np.asarray(episode_index, dtype=np.int64),
        task_index=np.asarray(args.task_index, dtype=np.int64),
    )
    print(f"task_index: {args.task_index}")
    print(f"trajectory_index: {args.trajectory_index}")
    print(f"episode_index: {episode_index}")
    print(f"states shape: {states.shape}")
    print(f"center_uvd shape: {agent_state['center_uvd'].shape}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
