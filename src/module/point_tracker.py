import pickle
import subprocess

import cv2
import numpy as np
import torch
from rich.console import Console

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import tracker_certainty
from src.common.geom_utils import sample_points_from_mask

cs = Console()


class PointTracker:
    """Online multi-object point tracking using TAPNext++.

    Usage::

        tracker = PointTracker()
        result = tracker.track(frame0, points=points, anchor_frame=True)
        result = tracker.track(frame1, anchor_frame=False)

    ``points`` should have shape (N, P, 2), where N is object count and P is
    points per object. Coordinates are original-image (x, y) pixels.
    """

    def __init__(
        self,
        model_path="/data0/luokang/dataset/luokang/ckpts/tapnextpp_ckpt.pt",
        device="cuda",
        image_size=256,
        use_certainty=False,
        threshold=0.5,
        certainty_radius=8,
    ):
        self.device = device
        self.image_size = image_size
        self.threshold = threshold
        self.certainty_radius = certainty_radius
        self.use_certainty = use_certainty

        self.model = TAPNext(image_size=(image_size, image_size))
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state_dict = {k.replace("tapnext.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        self.state = None

    def _preprocess(self, frame_rgb):
        frame = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(frame).to(self.device, dtype=torch.float32) / 255.0
        tensor = tensor * 2.0 - 1.0
        return tensor.unsqueeze(0).unsqueeze(0)

    def _make_queries(self, points, output_size):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError("points must have shape (N, P, 2)")

        h, w = output_size
        scaled = points.copy()
        scaled[..., 0] = scaled[..., 0] * self.image_size / max(w, 1)
        scaled[..., 1] = scaled[..., 1] * self.image_size / max(h, 1)
        scaled = np.clip(scaled, 0, self.image_size - 1)

        flat_points_xy = scaled.reshape(-1, 2)
        queries = np.zeros((1, flat_points_xy.shape[0], 3), dtype=np.float32)
        queries[0, :, 1] = flat_points_xy[:, 1]
        queries[0, :, 2] = flat_points_xy[:, 0]
        return torch.from_numpy(queries).to(self.device), points.shape[:2]

    def _format_output(self, tracks, track_logits, visible_logits):
        st = self.state
        h, w = st["output_size"]
        num_objects, points_per_object = st["points_shape"]

        scores = torch.sigmoid(visible_logits)
        if self.use_certainty:
            scores = scores * tracker_certainty(
                tracks, track_logits, radius=self.certainty_radius
            )
            visibles = scores > self.threshold
        else:
            visibles = visible_logits > 0

        tracks_yx = tracks[0, 0].detach().float().cpu().numpy()
        points = np.empty_like(tracks_yx)
        points[:, 0] = tracks_yx[:, 1] * w / self.image_size
        points[:, 1] = tracks_yx[:, 0] * h / self.image_size
        points = points.reshape(num_objects, points_per_object, 2).astype(np.float32)

        visibles = visibles[0, 0, :, 0].detach().cpu().numpy()
        visibles = visibles.reshape(num_objects, points_per_object).astype(bool)

        return {"points": points, "visibles": visibles}

    def reset(self, frame_rgb, points):
        """Initialize tracking from an anchor frame and object points.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), uint8, RGB order.
            points: np.ndarray, shape (N, P, 2), original-image (x, y) pixels.

        Returns:
            dict with points (N, P, 2) and visibles (N, P).
        """
        output_size = frame_rgb.shape[:2]
        query_points, points_shape = self._make_queries(points, output_size)
        video = self._preprocess(frame_rgb)

        with torch.no_grad():
            tracks, track_logits, visible_logits, tracking_state = self.model(
                video=video,
                query_points=query_points,
            )

        self.state = {
            "tracking_state": tracking_state,
            "output_size": output_size,
            "points_shape": points_shape,
            "frame_idx": 0,
        }
        return self._format_output(tracks, track_logits, visible_logits)

    def update(self, frame_rgb):
        """Track points on the next frame."""
        if self.state is None:
            raise RuntimeError("PointTracker must be reset before update")

        video = self._preprocess(frame_rgb)
        with torch.no_grad():
            tracks, track_logits, visible_logits, tracking_state = self.model(
                video=video,
                state=self.state["tracking_state"],
            )

        self.state["tracking_state"] = tracking_state
        self.state["frame_idx"] += 1
        return self._format_output(tracks, track_logits, visible_logits)

    def track(self, frame_rgb, points=None, anchor_frame=False):
        """Process one frame and return multi-object point tracks.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), uint8, RGB order.
            points: np.ndarray, shape (N, P, 2). Required for anchor frame.
            anchor_frame: If True, initialize with points; otherwise update.

        Returns:
            dict with points (N, P, 2) and visibles (N, P).
        """
        if anchor_frame:
            if points is None:
                raise ValueError("points are required when anchor_frame=True")
            return self.reset(frame_rgb, points)

        return self.update(frame_rgb)

    def draw_on_image(self, image, result, save_path=None):
        """Draw tracked points on an image.

        Args:
            image: np.ndarray, shape (H, W, 3), RGB or BGR.
            result: Output dict from track/reset/update.
            save_path: Optional path to save the drawn image.

        Returns:
            np.ndarray: Drawn image.
        """
        colors = [
            (60, 60, 255),
            (255, 144, 30),
            (50, 205, 50),
            (0, 215, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        drawn = image.copy()
        points = result["points"]
        visibles = result.get("visibles", np.ones(points.shape[:2], dtype=bool))

        for obj_idx in range(points.shape[0]):
            color = np.array(colors[obj_idx % len(colors)], dtype=np.float32)
            pale_color = tuple(np.round(color * 0.35 + 255 * 0.65).astype(int).tolist())
            color = tuple(color.astype(int).tolist())
            for point, is_visible in zip(points[obj_idx], visibles[obj_idx]):
                x, y = np.round(point).astype(int)
                draw_color = color if is_visible else pale_color
                cv2.circle(drawn, (x, y), 2, draw_color, -1, lineType=cv2.LINE_AA)

        if save_path is not None:
            cv2.imwrite(save_path, drawn)

        return drawn


def OnlineTest():
    video_path = "/data0/luokang/research/GraphVLA/__test__/data/data.mp4"
    points_path = "/data0/luokang/research/GraphVLA/__test__/data/0_0_point_tracking_frames.pkl"
    save_path = "/data0/luokang/research/GraphVLA/__tmp__/point_tracking_vis.mp4"
    tracker = PointTracker()

    with open(points_path, "rb") as f:
        tracking_frames = pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    interval = 1
    i = 0
    vis_frames = []
    points = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i % interval != 0:
            i += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if points is None:
            sampled_points = []
            frame_h, frame_w = frame_rgb.shape[:2]
            for obj_id in sorted(tracking_frames):
                item = tracking_frames[obj_id]
                node = str(item.get("node", ""))
                if not item.get("is_object", True) or "gripper" in node.lower():
                    cs.print(f"skip track_id={obj_id} node={node}")
                    continue
                mask = item["frames"][0]["mask"]
                obj_points = sample_points_from_mask(mask, num_points=128)
                mask_h, mask_w = mask.shape[:2]
                obj_points[:, 0] *= frame_w / mask_w
                obj_points[:, 1] *= frame_h / mask_h
                sampled_points.append(obj_points)
            points = np.stack(sampled_points, axis=0)
            result = tracker.track(frame_rgb, points=points, anchor_frame=True)
        else:
            result = tracker.track(frame_rgb, anchor_frame=False)

        vis_frame = tracker.draw_on_image(frame_bgr, result)
        vis_frames.append(vis_frame)
        i += 1
        cs.print(
            f"frame {i}: points={result['points'].shape}, "
            f"visibles={result['visibles'].sum()}/{result['visibles'].size}"
        )

    cap.release()

    if len(vis_frames) == 0:
        raise RuntimeError(f"no frames were read from {video_path}")

    fps = 10.0
    h, w = vis_frames[0].shape[:2]
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
            save_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for frame in vis_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    cs.print(f"Vis video: {save_path} ({len(vis_frames)} frames, {fps:.1f} fps)")


if __name__ == "__main__":
    OnlineTest()
