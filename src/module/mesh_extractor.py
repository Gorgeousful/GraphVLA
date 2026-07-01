import os
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import cv2
import pickle
from rich.console import Console
cs = Console()


class MeshExtractor:
    """Single-image mesh generation using sam-3d-objects.

    Takes an RGB image + binary mask, produces a 3D mesh (.glb).

    Usage::

        extractor = MeshExtractor()
        output = extractor.extract(image, mask)
        extractor.save_glb(output, "output.glb")
    """

    def __init__(
        self,
        config_path="/data0/luokang/dataset/luokang/ckpts/sam-3d-objects/checkpoints/pipeline.yaml",
        compile_model=False,
    ):
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        os.environ.setdefault("CUDA_HOME", conda_prefix)

        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = compile_model
        config.workspace_dir = os.path.dirname(config_path)

        self.pipeline = instantiate(config)

    def extract(self, image, mask, seed=42):
        """Generate 3D mesh from image + mask.

        Args:
            image: np.ndarray, shape (H, W, 3), uint8, RGB order.
            mask: np.ndarray, shape (H, W), bool.
            seed: int, random seed for generation.

        Returns:
            dict: Contains "glb" (trimesh Scene) and other outputs.
        """
        alpha = mask.astype(np.uint8)[:, :, None] * 255
        rgba = np.concatenate([image[:, :, :3], alpha], axis=-1)
        output = self.pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )
        return output

    def save_glb(self, output, save_path=None):
        """Export glb from pipeline output.

        Args:
            output: dict returned by extract().
            save_path: str or Path, output .glb file path.
        """
        glb = output.get("glb")
        if glb is None:
            raise RuntimeError(f"No 'glb' in output. Keys: {sorted(output.keys())}")
        if save_path:
            os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
            glb.export(str(save_path))
            cs.print(f"Saved mesh: {save_path}")
        return glb

    def sample_glb(self, glb, num=512, save_path=None):
        """Sample points from a glb mesh using farthest point sampling.

        Args:
            glb: trimesh Scene or Trimesh object.
            num: int, number of points to sample.
            save_path: optional str or Path, save sampled points as .npy.

        Returns:
            np.ndarray, shape (num, 3), float32, sampled point cloud (xyz only).
        """
        # collect all vertices (discard color)
        if hasattr(glb, "geometry"):
            vertices = []
            for name, mesh in glb.geometry.items():
                vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
            vertices = np.concatenate(vertices, axis=0) if vertices else np.zeros((0, 3))
        else:
            vertices = np.asarray(glb.vertices, dtype=np.float64)

        n = len(vertices)
        if n == 0:
            raise RuntimeError("Mesh has no vertices")
        if n <= num:
            cs.print(f"[warn] mesh has {n} vertices < {num}, padding with duplicates")
            indices = np.arange(n)
            extra = np.random.randint(0, n, size=num - n)
            indices = np.concatenate([indices, extra])
            sampled = vertices[indices].astype(np.float32)
        else:
            # farthest point sampling
            selected = np.zeros(num, dtype=np.int64)
            selected[0] = np.random.randint(n)
            min_dist = np.full(n, np.inf)

            for i in range(1, num):
                dist = np.linalg.norm(vertices - vertices[selected[i - 1]], axis=1)
                np.minimum(min_dist, dist, out=min_dist)
                selected[i] = np.argmax(min_dist)

            sampled = vertices[selected].astype(np.float32)

        if save_path:
            os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
            np.save(str(save_path), sampled)
            cs.print(f"Saved point cloud: {save_path} {sampled.shape}")

        return sampled


if __name__ == "__main__":
    video_path = "/data0/luokang/research/GraphVLA/__test__/data/data.mp4"
    tracking_pkl = "/data0/luokang/research/GraphVLA/__test__/data/0_0_point_tracking_frames.pkl"
    save_dir = "/data0/luokang/research/GraphVLA/__tmp__"
    os.makedirs(save_dir, exist_ok=True)
    frame_index = 0

    # read first frame
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    cs.print(f"Frame shape: {frame_rgb.shape}")

    # load masks from tracking pkl
    with open(tracking_pkl, "rb") as f:
        tracks = pickle.load(f)

    extractor = MeshExtractor()
    for track_id, item in tracks.items():
        node = item.get("node", f"track_{track_id}")
        if not item.get("is_object", True):
            continue
        frame_data = item.get("frames", {}).get(frame_index)
        if frame_data is None or frame_data.get("mask") is None:
            cs.print(f"[skip] {node}: no mask on frame {frame_index}")
            continue

        mask = frame_data["mask"]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        cs.print(f"Generating mesh for: {node} (mask pixels: {mask.sum()})")
        output = extractor.extract(frame_rgb, mask)
        safe_name = node.replace(" ", "_")
        extractor.save_glb(output, os.path.join(save_dir, f"mesh_{safe_name}.glb"))

    cs.print("\nDone.")
