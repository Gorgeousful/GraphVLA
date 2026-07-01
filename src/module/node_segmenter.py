import json
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sam3.model_builder import build_sam3_video_model
from rich.console import Console
cs = Console()


class NodeSegmenter:
    """Real-time streaming node segmentation using SAM3 tracker.

    Frame-by-frame tracking with point prompts on the first frame.
    No complete video required — frames are fed one at a time.

    Usage::

        segmenter = NodeSegmenter()
        for i, frame in enumerate(frames):
            if i == 0:
                masks = segmenter.predict(frame, points=points_list, anchor_frame=True)
            else:
                masks = segmenter.predict(frame, anchor_frame=False)
    """

    def __init__(
        self,
        model_path="/data0/luokang/dataset/luokang/ckpts/sam3/sam3.pt",
        device="cuda:0",
    ):
        model = build_sam3_video_model(
            checkpoint_path=model_path,
            load_from_HF=False,
        )
        self.tracker = model.tracker
        self.tracker.backbone = model.detector.backbone

        # state — populated by reset()
        self.state = None

    def _preprocess(self, frame_rgb):
        """numpy RGB → float16 tensor, shape (3, S, S), normalized [-1, 1]."""
        img_size = self.tracker.image_size
        pil_img = Image.fromarray(frame_rgb)
        pil_img = TF.resize(pil_img, size=(img_size, img_size))
        tensor = TF.to_tensor(pil_img).half()       # float16, [0, 1]
        tensor = (tensor - 0.5) / 0.5                # [-1, 1]
        return tensor

    def _extract_masks(self):
        """Extract bool masks from self.state['video_res_masks']."""
        h, w = self.state["output_size"]
        obj_ids = self.state["obj_ids"]
        video_res_masks = self.state["video_res_masks"]

        masks = []
        for i in range(len(obj_ids)):
            mask_np = (video_res_masks[i, 0] > 0).cpu().numpy()
            if mask_np.shape != (h, w):
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            masks.append(mask_np)

        return masks

    def reset(self, frame_rgb, points):
        """Initialize tracker with first frame and point prompts.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), uint8, RGB order.
            points: list of point sets, one per node.
                Each element is a list of (x, y) pixel coordinates.

        Returns:
            list[np.ndarray]: One bool mask per node, each shape (H, W).
        """
        h, w = frame_rgb.shape[:2]

        # manually init state — no video_path required
        inference_state = self.tracker.init_state(
            video_height=h,
            video_width=w,
            num_frames=1,
            offload_video_to_cpu=True,
        )
        inference_state["images"] = [self._preprocess(frame_rgb)]

        # add point prompts for all nodes on frame 0
        for node_idx, node_points in enumerate(points):
            pts_array = np.array(node_points, dtype=np.float32)
            pts_array[:, 0] /= w
            pts_array[:, 1] /= h
            np.clip(pts_array, 0, 1, out=pts_array)

            _, obj_ids, _, video_res_masks = self.tracker.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=node_idx,
                points=torch.from_numpy(pts_array),
                labels=torch.ones(len(node_points), dtype=torch.int32),
            )

        # propagate frame 0 to get initial masks
        obj_ids = None
        video_res_masks = None
        for result in self.tracker.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            reverse=False,
            propagate_preflight=True,
            tqdm_disable=True,
        ):
            _, obj_ids, _, video_res_masks, _ = result

        self.state = {
            "inference_state": inference_state,
            "output_size": (h, w),
            "obj_ids": obj_ids,
            "video_res_masks": video_res_masks,
            "frame_idx": 0,
        }

        return self._extract_masks()

    def update(self, frame_rgb):
        """Advance tracker by one frame and return masks.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), uint8, RGB order.

        Returns:
            list[np.ndarray]: One bool mask per node, each shape (H, W).
        """
        st = self.state
        st["frame_idx"] += 1
        frame_idx = st["frame_idx"]

        # append new frame to state
        st["inference_state"]["images"].append(self._preprocess(frame_rgb))
        st["inference_state"]["num_frames"] = frame_idx + 1

        # propagate one frame
        for result in self.tracker.propagate_in_video(
            st["inference_state"],
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
            reverse=False,
            propagate_preflight=False,
            tqdm_disable=True,
        ):
            _, st["obj_ids"], _, st["video_res_masks"], _ = result

        return self._extract_masks()

    def predict(self, frame_rgb, points=None, anchor_frame=True):
        """Process one frame and return masks for all nodes.

        Args:
            frame_rgb: np.ndarray, shape (H, W, 3), uint8, RGB order.
            points: list of point sets, one per node. Required when anchor_frame=True.
            anchor_frame: If True, reset tracker (first frame + point prompts).
                          If False, advance tracking by one frame.

        Returns:
            list[np.ndarray]: One bool mask per node, each shape (H, W).
        """
        if anchor_frame:
            if points is None:
                raise ValueError("points are required when anchor_frame=True")
            return self.reset(frame_rgb, points)

        return self.update(frame_rgb)

    def draw_on_image(self, image, masks, labels=None, save_path=None):
        """Draw segmentation masks overlaid on the image.

        Args:
            image: np.ndarray, shape (H, W, 3), uint8, RGB or BGR.
            masks: list[np.ndarray], each shape (H, W), bool.
            labels: optional list[str], one label per mask.
            save_path: if provided, save the result to this path.

        Returns:
            np.ndarray: The drawn image (BGR).
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

        for i, mask in enumerate(masks):
            if mask is None or not mask.any():
                continue
            color = colors[i % len(colors)]

            overlay = np.zeros_like(drawn)
            overlay[mask] = color
            drawn = np.where(
                mask[..., None],
                (drawn * 0.55 + overlay * 0.45).astype(np.uint8),
                drawn,
            )

            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
                label = labels[i] if labels and i < len(labels) else str(i)
                cv2.putText(
                    drawn, label,
                    (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
                )

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            cv2.imwrite(save_path, drawn)
            print(f"Saved segmentation vis: {save_path}")

        return drawn


if __name__ == "__main__":
    video_path = "/data0/luokang/research/GraphVLA/__test__/data/data.mp4"
    localization_json = "/data0/luokang/research/GraphVLA/__test__/data/0_0_point_localization.json"
    save_dir = "/data0/luokang/research/GraphVLA/__tmp__"
    os.makedirs(save_dir, exist_ok=True)

    # load node points (only is_object=True nodes)
    with open(localization_json, "r") as f:
        loc_data = json.load(f)

    node_points = []
    node_labels = []
    for group in loc_data:
        for item in group:
            if item.get("is_object", True) and item.get("points"):
                node_points.append([tuple(p) for p in item["points"]])
                node_labels.append(item["node"])

    print(f"Tracking {len(node_points)} nodes: {node_labels}")

    # simulate streaming: read frames one by one
    cap = cv2.VideoCapture(video_path)
    segmenter = NodeSegmenter()

    masks = None
    last_frame_bgr = None
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        last_frame_bgr = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        masks = segmenter.predict(
            frame_rgb,
            points=node_points if i == 0 else None,
            anchor_frame=(i == 0),
        )

        if i % 25 == 0:
            print(f"frame {i}: {len(masks)} masks, "
                  f"pixels={[int(m.sum()) for m in masks]}")
        i += 1
    cap.release()

    print(f"\nDone. Tracked {i} frames.")

    # save last frame visualization
    segmenter.draw_on_image(
        last_frame_bgr, masks,
        labels=node_labels,
        save_path=os.path.join(save_dir, "node_tracking_vis.png"),
    )
