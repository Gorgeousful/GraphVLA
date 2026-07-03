import importlib.util
import os
import sys
import trimesh
import numpy as np
from rich.console import Console
cs = Console()
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

class ScaleEstimator:
    """Scale estimation for GLB meshes and depth calibration.

    Uses camera intrinsics/extrinsics to relate mesh space and depth space
    to real-world 3D coordinates.

    Usage::

        estimator = ScaleEstimator(intrinsic, extrinsic)

        # scale a GLB mesh to match real-world depth
        scale = estimator.estimate_glb(mesh, depth, mask)

        # calibrate predicted depth to metric depth
        result = estimator.estimate_depth(predicted_depth, eef_state)
    """

    def __init__(
        self,
        intrinsic,
        extrinsic,
        example="libero",
        embodiment="franka_panda",
        **robot_kwargs,
    ):
        """
        Args:
            intrinsic: np.ndarray, shape (3, 3), camera intrinsic matrix.
            extrinsic: np.ndarray, shape (4, 4), camera extrinsic matrix.
            example: str, examples subdirectory containing embodiment/robot.py.
            embodiment: str, concrete robot embodiment name for GeomRobot.
            **robot_kwargs: extra keyword arguments forwarded to GeomRobot.
        """
        self.intrinsic = np.asarray(intrinsic)
        self.extrinsic = np.asarray(extrinsic)
        self.example = example
        self.embodiment = embodiment
        self.gripper_geometry = self._load_geom(
            example=example,
            embodiment=embodiment,
            robot_kwargs=robot_kwargs,
        )

    def _load_geom(self, example, embodiment, robot_kwargs):
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        elif sys.path[0] != REPO_ROOT:
            sys.path.remove(REPO_ROOT)
            sys.path.insert(0, REPO_ROOT)

        example_py = os.path.normpath(os.path.join(EXAMPLES_DIR, example, "embodiment", "robot.py"))
        spec = importlib.util.spec_from_file_location(f"{example}_{embodiment}", example_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GeomRobot(
            embodiment=embodiment,
            **robot_kwargs,
        )

    #: public
    def estimate_glb(
        self,
        mesh,
        depth,
        mask,
        mode="median",
        quantile=0.95,
        num_points=10000,
        pair_count=100000,
        seed=0,
    ):
        """Estimate scale factor for a GLB mesh to match real-world depth.

        Compares pairwise distance distributions between mesh vertices
        and depth-backprojected 3D points within the mask region.

        Args:
            mesh: trimesh Scene or Trimesh object.
            depth: np.ndarray, shape (H, W), float, calibrated depth.
            mask: np.ndarray, shape (H, W), bool, object segmentation mask.
            mode: "median" or "quantile", statistic for pairwise distances.
            quantile: float, quantile value when mode="quantile".
            num_points: int, number of points to sample from each source.
            pair_count: int, number of random pairs for distance metric.
            seed: int, random seed.

        Returns:
            float: scale factor. Apply via apply_scale_about_center().
        """
        intrinsic = self.intrinsic

        def backproject(depth, mask):
            h, w = depth.shape
            valid = mask & np.isfinite(depth) & (depth > 0)
            v, u = np.where(valid)
            if len(u) == 0:
                return np.empty((0, 3), dtype=np.float64)
            z = depth[v, u].astype(np.float64)
            fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
            cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
            x = (u.astype(np.float64) - cx) / fx * z
            y = (v.astype(np.float64) - cy) / fy * z
            return np.stack([x, y, z], axis=1)

        def sample(points, count, seed):
            rng = np.random.default_rng(seed)
            if len(points) < 2:
                raise ValueError(f"At least 2 points required, got {len(points)}")
            idx = rng.choice(len(points), size=count, replace=len(points) < count)
            return points[idx]

        def pairwise_metric(points, mode, quantile, pair_count, seed):
            rng = np.random.default_rng(seed)
            idx_a = rng.integers(0, len(points), size=pair_count)
            idx_b = rng.integers(0, len(points), size=pair_count)
            same = idx_a == idx_b
            while np.any(same):
                idx_b[same] = rng.integers(0, len(points), size=int(np.sum(same)))
                same = idx_a == idx_b
            distances = np.linalg.norm(points[idx_a] - points[idx_b], axis=1)
            if mode == "median":
                return float(np.median(distances))
            elif mode == "quantile":
                return float(np.quantile(distances, quantile))
            raise ValueError(f"Unknown mode: {mode}")

        # collect mesh vertices
        if hasattr(mesh, "geometry"):
            verts = []
            for name, m in mesh.geometry.items():
                verts.append(np.asarray(m.vertices, dtype=np.float64))
            source_points = np.concatenate(verts, axis=0) if verts else np.zeros((0, 3))
        else:
            source_points = np.asarray(mesh.vertices, dtype=np.float64)

        # backproject depth within mask
        target_points = backproject(depth, mask)

        # sample and compute metrics
        source_pts = sample(source_points, num_points, seed)
        target_pts = sample(target_points, num_points, seed + 1)

        source_metric = pairwise_metric(source_pts, mode, quantile, pair_count, seed + 2)
        target_metric = pairwise_metric(target_pts, mode, quantile, pair_count, seed + 3)

        scale = float(target_metric / source_metric)
        cs.print(f"scale={scale:.6g} (source={source_metric:.4g}, target={target_metric:.4g})")
        return scale

    def estimate_depth(
        self,
        predicted_depth,
        eef_state,
        gripper_state=None,
        invert=False,
        calib_frame=0,
        fit_iterations=10,
        trim_quantile=0.9,
    ):
        """Estimate depth affine scale/shift using rendered gripper depth as anchor.

        Fits scale and shift in the prediction space:
            target = scale * predicted_depth + shift
        where target is rendered depth, or 1/rendered_depth if invert=True.

        Args:
            predicted_depth: np.ndarray, (H, W) or (T, H, W), model output.
            eef_state: dict mapping frame_index -> 6-D TCP pose, or one 6-D TCP pose.
            gripper_state: optional scalar or dict mapping frame_index -> scalar finger opening.
            invert: bool, fit against inverse rendered depth target while keeping predicted_depth raw.
            calib_frame: int, frame index used for initial fit.
            fit_iterations: int, iterations for trimmed least squares.
            trim_quantile: float, quantile for trimming outliers.

        Returns:
            dict: Contains scalar ``scale`` and ``shift``.
        """
        intrinsic = self.intrinsic
        extrinsic = self.extrinsic

        def robust_affine_fit(x, y):
            x = x.astype(np.float64)
            y = y.astype(np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]

            keep = np.ones(len(x), dtype=bool)
            scale, shift = 1.0, 0.0
            for _ in range(fit_iterations):
                if int(keep.sum()) < 2:
                    break
                design = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
                scale, shift = np.linalg.lstsq(design, y[keep], rcond=None)[0]
                residual = np.abs(scale * x + shift - y)
                threshold = float(np.quantile(residual[np.isfinite(residual)], trim_quantile))
                keep = np.isfinite(residual) & (residual <= threshold)

            pred = scale * x + shift
            valid_pred = np.isfinite(pred) & (pred > 0)
            error = pred[valid_pred] - y[valid_pred]
            return {
                "scale": float(scale),
                "shift": float(shift),
                "fit_points": int(keep.sum()),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
            }

        # --- main logic ---
        depths = np.asarray(predicted_depth, dtype=np.float32)
        if depths.ndim == 2:
            depths = depths[None]
        elif depths.ndim != 3:
            raise ValueError(
                f"Expected predicted_depth shape (H, W) or (T, H, W), got {depths.shape}."
            )
        frame_count, height, width = depths.shape

        # Keep source in raw prediction space. If invert=True, invert rendered targets instead.
        source = depths.copy()

        def make_target(rendered_depth):
            rendered_depth = np.asarray(rendered_depth, dtype=np.float32)
            if not invert:
                return rendered_depth
            target = np.full(rendered_depth.shape, np.nan, dtype=np.float32)
            valid = np.isfinite(rendered_depth) & (rendered_depth > 0)
            target[valid] = (1.0 / rendered_depth[valid].astype(np.float64)).astype(np.float32)
            return target

        # --- initial fit on calibration frame ---
        calib_state = eef_state.get(calib_frame) if isinstance(eef_state, dict) else eef_state
        if calib_state is None:
            raise RuntimeError(f"Missing eef_state for calibration frame {calib_frame}")
        calib_tcp_state = np.asarray(calib_state, dtype=np.float64).ravel()
        if calib_tcp_state.size != 6:
            raise ValueError(f"Expected 6 values for eef_state, got {calib_tcp_state.size}.")
        calib_gripper_state = gripper_state.get(calib_frame) if isinstance(gripper_state, dict) else gripper_state
        calib_gripper_state = None if calib_gripper_state is None else float(calib_gripper_state)

        rendered_z = self.gripper_geometry.project_gripper_to_depth(
            tcp_state=calib_tcp_state,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image_size=(height, width),
            device="cuda:0",
            gripper_state=calib_gripper_state,
        )
        target = make_target(rendered_z)
        anchor = np.isfinite(target) & np.isfinite(source[calib_frame])

        fit = robust_affine_fit(source[calib_frame][anchor], target[anchor])
        return {
            "scale": float(fit["scale"]),
            "shift": float(fit["shift"]),
        }

    def apply_scale_glb(self, mesh, scale):
        """Scale a mesh about its bounding box center.
        Args:
            mesh: trimesh.Trimesh object.
            scale: float, scale factor.

        Returns:
            trimesh.Trimesh: scaled mesh copy.
        """ 
        scaled = mesh.copy()
        center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
        transform = (
            trimesh.transformations.translation_matrix(center)
            @ np.diag([scale, scale, scale, 1.0])
            @ trimesh.transformations.translation_matrix(-center)
        )
        scaled.apply_transform(transform)
        return scaled

    def apply_scale_depth(self, depths, scales, shifts):
        """Apply per-frame affine depth scaling.

        Args:
            depths: np.ndarray, shape (H, W) or (T, H, W).
            scales: float or sequence of float, one per frame or one shared value.
            shifts: float or sequence of float, one per frame or one shared value.

        Returns:
            np.ndarray: scaled depth with the same shape convention as input.
        """
        depth_array = np.asarray(depths, dtype=np.float32)
        single_frame = depth_array.ndim == 2
        if single_frame:
            depth_array = depth_array[None]
        elif depth_array.ndim != 3:
            raise ValueError(
                f"Expected depths shape (H, W) or (T, H, W), got {depth_array.shape}."
            )

        frame_count = depth_array.shape[0]
        scale_array = np.asarray(scales, dtype=np.float64).ravel()
        shift_array = np.asarray(shifts, dtype=np.float64).ravel()
        if scale_array.size == 1:
            scale_array = np.full(frame_count, float(scale_array[0]), dtype=np.float64)
        if shift_array.size == 1:
            shift_array = np.full(frame_count, float(shift_array[0]), dtype=np.float64)
        if scale_array.size != frame_count or shift_array.size != frame_count:
            raise ValueError(
                "scales and shifts must be scalar or have one value per depth frame "
                f"(got {scale_array.size} scales, {shift_array.size} shifts, {frame_count} frames)."
            )

        scaled = np.full(depth_array.shape, np.nan, dtype=np.float32)
        for fi in range(frame_count):
            values = scale_array[fi] * depth_array[fi].astype(np.float64) + shift_array[fi]
            valid = np.isfinite(values) & (values > 0)
            scaled[fi][valid] = values[valid].astype(np.float32)

        return scaled[0] if single_frame else scaled


class RawDepthShiftCalibrator:
    def __init__(self, anchor_frame_depth, anchor_frame_mask, method="median"):
        if method not in {"median", "mean"}:
            raise ValueError(f"method must be 'median' or 'mean', got {method!r}")
        self.anchor_frame_depth = np.asarray(anchor_frame_depth, dtype=np.float32)
        self.anchor_frame_mask = np.asarray(anchor_frame_mask, dtype=bool)
        self.method = method
        if self.anchor_frame_depth.ndim != 2:
            raise ValueError(f"anchor_frame_depth must have shape (H, W), got {self.anchor_frame_depth.shape}")
        if self.anchor_frame_mask.shape != self.anchor_frame_depth.shape:
            raise ValueError(
                f"anchor_frame_mask shape {self.anchor_frame_mask.shape} does not match "
                f"anchor_frame_depth shape {self.anchor_frame_depth.shape}"
            )

    def calibrate(self, depth, mask):
        depth = np.asarray(depth, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if depth.shape != self.anchor_frame_depth.shape:
            raise ValueError(f"depth shape {depth.shape} does not match anchor shape {self.anchor_frame_depth.shape}")
        if mask.shape != depth.shape:
            raise ValueError(f"mask shape {mask.shape} does not match depth shape {depth.shape}")

        valid = (
            self.anchor_frame_mask
            & mask
            & np.isfinite(self.anchor_frame_depth)
            & np.isfinite(depth)
            & (self.anchor_frame_depth > 0)
            & (depth > 0)
        )
        if not valid.any():
            raise ValueError("No valid pixels for raw depth shift calibration.")

        residual = self.anchor_frame_depth[valid].astype(np.float64) - depth[valid].astype(np.float64)
        if self.method == "median":
            shift = float(np.median(residual))
        else:
            shift = float(np.mean(residual))
        # cs.print(f"raw_depth_shift={shift:.6g} valid_pixels={int(valid.sum())} method={self.method}")
        return (depth + shift).astype(np.float32, copy=False)


if __name__ == "__main__":
    pass