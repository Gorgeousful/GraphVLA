from pathlib import Path
from typing import Iterable
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
file_dir = Path(__file__).resolve().parent
from src.common.geom_utils import (
    bbox_from_mask,
    make_pose,
    project_meshes_to_depth,
    project_meshes_to_mask,
    sample_mesh_pcd,
    transform_points,
)
from rich.console import Console
cs = Console()


class GeomFrankaPanda:
    """EEF-only Franka gripper geometry helper.

    The robot state is the 6-D TCP pose ``[x, y, z, rx, ry, rz]``
    (axis-angle).  Finger opening can be provided via ``gripper_state``.
    """

    _FINGER_GEOM_NAMES = frozenset({"finger1_visual", "finger2_visual"})
    _GRIPPER_FRAME_BODY = "right_hand"
    # Offline measurement of the center of the inner fingertip contact surface
    # in the right_hand frame. The fingers move symmetrically along local y.
    _FINGERTIP_CONTACT_Z = 0.097 # 0.097397454
    _TCP_OFFSET = np.array([0.0, 0.0, -_FINGERTIP_CONTACT_Z], dtype=np.float64)
    _MAX_GRIPPER_WIDTH = 0.08


    def __init__(
        self,
        mjcf_path: str | Path = file_dir / "franka_panda" / "robot.xml",
        with_fingers: bool = True,
        points_per_mesh: int = 512,
    ):
        self.mjcf_path = Path(mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self._raster_contexts: dict[str, object] = {}

        gripper_geom_names = (
            ("hand_visual", "finger1_visual", "finger2_visual")
            if with_fingers else ("hand_visual",)
        )

        # Separate static (hand) from dynamic (finger) geom names
        all_names = tuple(gripper_geom_names)
        self._hand_geom_names = tuple(n for n in all_names if n not in self._FINGER_GEOM_NAMES)
        self._finger_geom_names = tuple(n for n in all_names if n in self._FINGER_GEOM_NAMES)

        # Load all meshes at default pose
        mujoco.mj_forward(self.model, self.data)
        all_meshes = self._load_meshes_for_geoms(tuple(gripper_geom_names))
        self._hand_meshes = [m for m in all_meshes if m["name"] not in self._FINGER_GEOM_NAMES]

        # Pre-sample PCD via FPS for each mesh (gripper frame, default pose)
        self._hand_pcd = np.zeros((0, 3), dtype=np.float64)
        self._finger_pcds = {}  # geom_name → (N, 3) in gripper frame
        self._finger_default_g2g = {}  # geom_name → inv(hand_from_geom_default)

        for mesh in all_meshes:
            pts = sample_mesh_pcd([mesh], points_per_mesh)
            if mesh["name"] in self._FINGER_GEOM_NAMES:
                self._finger_pcds[mesh["name"]] = pts
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, mesh["name"])
                # Store the inverse of the geom pose *in the reference body
                # frame* (right_hand), NOT in the world frame.  The stored
                # PCD is in the reference-body frame, so the delta transform
                # used at projection time must also be in that frame.
                frame_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self._GRIPPER_FRAME_BODY,
                )
                frame_from_world = np.linalg.inv(make_pose(
                    self.data.xpos[frame_id],
                    self.data.xmat[frame_id].reshape(3, 3),
                ))
                hand_from_geom = frame_from_world @ make_pose(
                    self.data.geom_xpos[geom_id],
                    self.data.geom_xmat[geom_id].reshape(3, 3),
                )
                self._finger_default_g2g[mesh["name"]] = np.linalg.inv(hand_from_geom)
            else:
                self._hand_pcd = np.concatenate([self._hand_pcd, pts]) if len(self._hand_pcd) else pts

        self._pose_keypoints_local = self._build_pose_keypoints_local()

    #: public
    def project_gripper_to_mask(
        self,
        tcp_state: Iterable[float],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image_size: tuple[int, int],
        world_transform: np.ndarray | None = None,
        gripper_state: float | None = None,
    ) -> np.ndarray:
        """Project the gripper mesh into a binary camera mask."""
        meshes_cam = self._prepare_camera_projection(
            tcp_state, extrinsic, world_transform, gripper_state,
        )
        height, width = image_size
        return project_meshes_to_mask(meshes_cam, intrinsic, height, width)

    def project_gripper_to_bbox(
        self,
        tcp_state: Iterable[float],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image_size: tuple[int, int],
        world_transform: np.ndarray | None = None,
        gripper_state: float | None = None,
    ) -> list[int] | None:
        """Project the gripper mesh and return ``[x_min, y_min, x_max, y_max]``."""
        mask = self.project_gripper_to_mask(
            tcp_state, intrinsic, extrinsic, image_size, world_transform, gripper_state,
        )
        return bbox_from_mask(mask)

    def project_gripper_to_tcp(
        self,
        tcp_state: Iterable[float],
        extrinsic: np.ndarray,
        world_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return TCP pose in camera coordinates as ``[pos(3) + rot6d(6)]``."""
        state = np.asarray(tcp_state, dtype=np.float64).ravel()
        tcp_pose = make_pose(state[:3], R.from_rotvec(state[3:6]).as_matrix())
        if world_transform is not None:
            tcp_pose = np.asarray(world_transform, dtype=np.float64) @ tcp_pose
        cam_from_tcp = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64)) @ tcp_pose
        pos = cam_from_tcp[:3, 3]
        rot6d = cam_from_tcp[:3, :2].T.ravel()  # first two columns
        return np.concatenate([pos, rot6d])

    def project_gripper_to_pcd(
        self,
        tcp_state: Iterable[float],
        extrinsic: np.ndarray,
        world_transform: np.ndarray | None = None,
        gripper_state: float | None = None,
    ) -> np.ndarray:
        """Return a point cloud (N, 3) in camera coordinates (FPS-sampled at init)."""
        local_to_camera = self._camera_transform(tcp_state, extrinsic, world_transform)

        # Hand: rigid transform
        pcd_parts = [transform_points(self._hand_pcd, local_to_camera)]

        # Fingers: per-finger delta transform from default pose
        if gripper_state is not None and self._finger_pcds:
            self._set_finger_state(gripper_state)
            mujoco.mj_forward(self.model, self.data)
            # Compute delta in the reference-body (right_hand) frame to match
            # the frame the stored PCD lives in.  geom_xpos/geom_xmat give
            # the world-frame pose; we must left-multiply by frame_from_world
            # so the delta acts on right_hand-frame points.
            frame_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self._GRIPPER_FRAME_BODY,
            )
            frame_from_world = np.linalg.inv(make_pose(
                self.data.xpos[frame_id],
                self.data.xmat[frame_id].reshape(3, 3),
            ))
            for name, pcd in self._finger_pcds.items():
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                hand_from_geom = frame_from_world @ make_pose(
                    self.data.geom_xpos[geom_id],
                    self.data.geom_xmat[geom_id].reshape(3, 3),
                )
                delta = hand_from_geom @ self._finger_default_g2g[name]
                pcd_parts.append(transform_points(pcd, local_to_camera @ delta))

        return np.concatenate(pcd_parts) if len(pcd_parts) > 1 else pcd_parts[0]

    def project_gripper_to_depth(
        self,
        tcp_state: Iterable[float],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image_size: tuple[int, int],
        world_transform: np.ndarray | None = None,
        gripper_state: float | None = None,
        device: str = "cuda:0",
    ) -> np.ndarray:
        """Project the gripper mesh into a z-buffer depth map (nvdiffrast)."""
        meshes_cam = self._prepare_camera_projection(
            tcp_state, extrinsic, world_transform, gripper_state,
        )
        height, width = image_size
        return project_meshes_to_depth(
            meshes_cam, intrinsic, height, width,
            raster_context=self._raster_context(device),
            device=device,
        )

    def project_gripper_to_uvd(
        self,
        tcp_state: Iterable[float],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        world_transform: np.ndarray | None = None,
        gripper_width: float | None = None,
    ) -> dict:
        local_to_camera = self._camera_transform(tcp_state, extrinsic, world_transform)
        points_local = self._pose_keypoints_local
        point_names = ("root_uvd", "left_base_uvd", "right_base_uvd")
        if gripper_width is not None:
            width = float(gripper_width)
            if not np.isfinite(width):
                raise ValueError(f"gripper_width must be finite, got {gripper_width!r}")
            width = float(np.clip(width, 0.0, self._MAX_GRIPPER_WIDTH))
            half_width = width / 2.0
            fingertips_local = np.asarray(
                [
                    [0.0, half_width, self._FINGERTIP_CONTACT_Z],
                    [0.0, -half_width, self._FINGERTIP_CONTACT_Z],
                ],
                dtype=np.float64,
            )
            tcp_local = fingertips_local.mean(axis=0, keepdims=True)
            points_local = np.concatenate([points_local, fingertips_local, tcp_local], axis=0)
            point_names += ("left_fingertip_uvd", "right_fingertip_uvd", "tcp_uvd")

        points_camera = transform_points(points_local, local_to_camera)
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        pixels_h = (intrinsic @ points_camera.T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        uvd = np.concatenate([pixels, points_camera[:, 2:3]], axis=1)
        return dict(zip(point_names, uvd))

    def project_uvd_to_gripper(
        self,
        uvd_dict: dict,
        intrinsic: np.ndarray,
        gripper_width: float,
        extrinsic: np.ndarray | None = None,
        world_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        intrinsic = np.asarray(intrinsic, dtype=np.float64)

        root_uvd = np.asarray(uvd_dict["root_uvd"], dtype=np.float64)
        left_base_uvd = np.asarray(uvd_dict["left_base_uvd"], dtype=np.float64)
        right_base_uvd = np.asarray(uvd_dict["right_base_uvd"], dtype=np.float64)

        uvd = np.stack([root_uvd, left_base_uvd, right_base_uvd])
        z = uvd[:, 2]
        points_camera = np.stack([
            (uvd[:, 0] - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (uvd[:, 1] - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ], axis=1)
        points_local = self._pose_keypoints_local

        local_center = points_local.mean(axis=0)
        camera_center = points_camera.mean(axis=0)
        local_zero = points_local - local_center
        camera_zero = points_camera - camera_center
        u, _, vt = np.linalg.svd(local_zero.T @ camera_zero)
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1] *= -1
            rot = vt.T @ u.T
        trans = camera_center - rot @ local_center
        camera_from_gripper = np.eye(4, dtype=np.float64)
        camera_from_gripper[:3, :3] = rot
        camera_from_gripper[:3, 3] = trans

        if extrinsic is None:
            pose = camera_from_gripper
        else:
            pose = np.asarray(extrinsic, dtype=np.float64) @ camera_from_gripper
            if world_transform is not None:
                pose = np.linalg.inv(np.asarray(world_transform, dtype=np.float64)) @ pose

        offset = np.eye(4, dtype=np.float64)
        offset[:3, 3] = self._TCP_OFFSET
        tcp_pose = pose @ np.linalg.inv(offset)
        return np.concatenate([
            tcp_pose[:3, 3],
            R.from_matrix(tcp_pose[:3, :3]).as_rotvec(),
            np.asarray([gripper_width], dtype=np.float64),
        ])

    #: private
    def _build_pose_keypoints_local(self) -> np.ndarray:
        """Return root and fully-open finger-base anchors in the gripper frame."""
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("finger_joint1", "finger_joint2")
        ]
        max_width = float(
            self.model.jnt_range[joint_ids[0], 1]
            - self.model.jnt_range[joint_ids[1], 0]
        )
        self._set_finger_state(max_width)
        mujoco.mj_forward(self.model, self.data)

        frame_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self._GRIPPER_FRAME_BODY,
        )
        frame_from_world = np.linalg.inv(make_pose(
            self.data.xpos[frame_id],
            self.data.xmat[frame_id].reshape(3, 3),
        ))
        keypoints = [np.zeros(3, dtype=np.float64)]
        for body_name in ("leftfinger", "rightfinger"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            keypoints.append(transform_points(
                self.data.xpos[body_id][None], frame_from_world,
            )[0])

        self._set_finger_state(0.0)
        mujoco.mj_forward(self.model, self.data)
        return np.stack(keypoints)

    def _load_meshes_for_geoms(self, geom_names: tuple[str, ...]) -> list[dict]:
        """Load meshes for the given geom names in the current gripper frame."""
        frame_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._GRIPPER_FRAME_BODY)
        world_from_frame = make_pose(
            self.data.xpos[frame_id],
            self.data.xmat[frame_id].reshape(3, 3),
        )
        frame_from_world = np.linalg.inv(world_from_frame)

        meshes = []
        for geom_name in geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue

            vertices, faces = self._mesh_vertices_faces(mesh_id)
            if len(vertices) == 0 or len(faces) == 0:
                continue

            world_from_geom = make_pose(
                self.data.geom_xpos[geom_id],
                self.data.geom_xmat[geom_id].reshape(3, 3),
            )
            verts_local = transform_points(
                transform_points(vertices, world_from_geom), frame_from_world
            )
            meshes.append(
                {
                    "name": geom_name,
                    "vertices": verts_local.astype(np.float32, copy=False),
                    "faces": faces.astype(np.int64, copy=False),
                }
            )
        return meshes

    def _get_meshes(self, gripper_state: float | None) -> list[dict]:
        """Return all meshes, recomputing fingers if ``gripper_state`` is given."""
        if gripper_state is not None and self._finger_geom_names:
            self._set_finger_state(gripper_state)
            mujoco.mj_forward(self.model, self.data)
            finger_meshes = self._load_meshes_for_geoms(self._finger_geom_names)
            return self._hand_meshes + finger_meshes
        return self._hand_meshes

    def _set_finger_state(self, gripper_state: float) -> None:
        """Set finger joint positions for the given opening value."""
        half_open = float(gripper_state) / 2.0
        # finger_joint1 range=[0, 0.04], finger_joint2 range=[-0.04, 0]
        fingers = {
            "finger_joint1": half_open,
            "finger_joint2": -half_open,
        }
        for name, value in fingers.items():
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = int(self.model.jnt_qposadr[jnt_id])
            self.data.qpos[qpos_adr] = value

    def _mesh_vertices_faces(
        self, mesh_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        vert_adr = int(self.model.mesh_vertadr[mesh_id])
        vert_num = int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            self.model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32,
        ).reshape(-1, 3)

        face_adr = int(self.model.mesh_faceadr[mesh_id])
        face_num = int(self.model.mesh_facenum[mesh_id])
        faces = np.asarray(
            self.model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64,
        ).reshape(-1, 3)
        return vertices, faces

    def _tcp_state_to_gripper_pose(self, tcp_state: Iterable[float]) -> np.ndarray:
        """Return the 4x4 gripper pose in world coordinates from a 6-D TCP state."""
        state = np.asarray(tcp_state, dtype=np.float64).ravel()
        tcp_pose = make_pose(state[:3], R.from_rotvec(state[3:6]).as_matrix())
        offset = np.eye(4, dtype=np.float64)
        offset[:3, 3] = self._TCP_OFFSET
        return tcp_pose @ offset

    def _camera_transform(
        self,
        tcp_state: Iterable[float],
        extrinsic: np.ndarray,
        world_transform: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the 4x4 gripper-local → camera-space transform."""
        gripper_pose = self._tcp_state_to_gripper_pose(tcp_state)
        if world_transform is not None:
            gripper_pose = np.asarray(world_transform, dtype=np.float64) @ gripper_pose
        # extrinsic is the camera pose in world coordinates: T_world_from_camera.
        return np.linalg.inv(np.asarray(extrinsic, dtype=np.float64)) @ gripper_pose

    def _prepare_camera_projection(
        self,
        tcp_state: Iterable[float],
        extrinsic: np.ndarray,
        world_transform: np.ndarray | None,
        gripper_state: float | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Common pipeline: TCP → gripper → camera-space meshes."""
        local_to_camera = self._camera_transform(tcp_state, extrinsic, world_transform)
        meshes = self._get_meshes(gripper_state)
        return [
            (transform_points(m["vertices"], local_to_camera), m["faces"])
            for m in meshes
        ]

    def _raster_context(self, device: str):
        import torch
        import nvdiffrast.torch as dr

        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError(
                f"nvdiffrast depth rendering requires a CUDA device, got {device!r}."
            )
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device.index)

        key = str(torch_device)
        if key not in self._raster_contexts:
            self._raster_contexts[key] = dr.RasterizeCudaContext(device=torch_device)
        return self._raster_contexts[key]


# 外部接口
class GeomRobot:
    """Factory wrapper for embodiment-specific geometry helpers."""

    _REGISTRY = {
        "franka_panda": GeomFrankaPanda,
    }

    def __new__(cls, embodiment="franka_panda", *args, **kwargs):
        key = str(embodiment).lower()
        if key not in cls._REGISTRY:
            supported = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unsupported embodiment={embodiment!r}. Supported: {supported}."
            )
        return cls._REGISTRY[key](*args, **kwargs)


if __name__ == "__main__":
    import json
    import subprocess
    import cv2

    # --- paths ---
    REPO_ROOT = file_dir.parents[2]
    TASK_STEM = "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_ep0"
    DATA_DIR = REPO_ROOT / "__test__" / "v1" / "data"
    STATE_JSON = DATA_DIR / f"{TASK_STEM}_state.json"
    CAMERA_JSON = DATA_DIR / f"{TASK_STEM}_camera.json"
    VIDEO_PATH = DATA_DIR / f"{TASK_STEM}_image.mp4"
    OUTPUT_PATH = file_dir / "gripper_overlay.mp4"

    # --- load state ---
    state_record = json.loads(STATE_JSON.read_text())
    frames_data = {
        item["frame_index"]: {
            "eef_state": item["eef_state"],
            "gripper": abs(item["state"][6]) + abs(item["state"][7]),
        }
        for item in state_record["frames"]
    }

    # --- load camera ---
    camera_record = json.loads(CAMERA_JSON.read_text())
    intrinsic = np.asarray(camera_record["cameras"]["agentview"]["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(camera_record["cameras"]["agentview"]["extrinsic"], dtype=np.float64)
    image_size = tuple(camera_record["image_size"])

    # --- create geometry ---
    geometry = GeomRobot(embodiment="franka_panda", with_fingers=True)

    # --- process video ---
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in frames_data:
            data = frames_data[frame_idx]
            eef_state = data["eef_state"]

            # --- mask overlay ---
            mask = geometry.project_gripper_to_mask(
                tcp_state=eef_state,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                image_size=image_size,
                gripper_state=data["gripper"],
            )
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

            overlay = frame.copy()
            overlay[mask] = (0, 165, 255)  # orange
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            mask_u8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

            # --- PCD back-projection overlay ---
            pcd = geometry.project_gripper_to_pcd(
                tcp_state=eef_state,
                extrinsic=extrinsic,
                gripper_state=data["gripper"],
            )
            # Scale intrinsic if camera image_size differs from video size
            scale_x = width / image_size[1]
            scale_y = height / image_size[0]
            K = intrinsic.copy()
            K[0, :] *= scale_x
            K[1, :] *= scale_y

            # Project 3D → 2D
            valid = pcd[:, 2] > 1e-6
            proj = (K @ pcd[valid].T).T
            pixels = (proj[:, :2] / proj[:, 2:3]).astype(int)

            # Draw points (only within frame bounds)
            in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & \
                        (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            for px, py in pixels[in_bounds]:
                cv2.circle(frame, (px, py), 1, (255, 0, 0), -1)  # blue dots

            # --- TCP center overlay ---
            tcp_cam = geometry.project_gripper_to_tcp(
                tcp_state=eef_state, extrinsic=extrinsic,
            )
            tcp_pos_cam = tcp_cam[:3]
            if tcp_pos_cam[2] > 1e-6:
                p = (K @ tcp_pos_cam).astype(np.float64)
                tx, ty = int(p[0] / p[2]), int(p[1] / p[2])
                if 0 <= tx < width and 0 <= ty < height:
                    cv2.circle(frame, (tx, ty), 4, (0, 0, 255), -1)  # red dot

        frames_bgr.append(frame)
        frame_idx += 1

    cap.release()

    # --- write video (same pattern as depth_predictor.draw_on_image) ---
    h, w = frames_bgr[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "26", "-g", "2",
            "-pix_fmt", "yuv420p",
            str(OUTPUT_PATH),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for frame in frames_bgr:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    cs.print(f"Wrote {frame_idx} frames → {OUTPUT_PATH}")
