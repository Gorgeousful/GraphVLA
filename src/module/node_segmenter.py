import os
import contextlib
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sam3.model_builder import build_sam3_stream_model
from ultralytics.models.sam import SAM2VideoPredictor
from rich.console import Console
cs = Console()


def _validate_mask_prompts(masks, frame_rgb):
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[0] == 0 or masks.shape[1:] != frame_rgb.shape[:2]:
        raise ValueError("masks must have shape (N, H, W) at the original frame resolution")
    if not np.isin(masks, [0, 1]).all() or not masks.reshape(len(masks), -1).any(axis=1).all():
        raise ValueError("each mask must be nonempty and binary (0/1)")
    return masks.astype(bool, copy=True)


class NodeSegmenter:
    """Frame-by-frame node segmentation and video-level prompt segmentation with one SAM3 model."""

    def __init__(
        self,
        model_path="/data0/luokang/dataset/luokang/ckpts/sam3/sam3.pt",
        device="cuda",
        mode=None,
        model=None,
    ):
        self.model_path = model_path
        self.device = device
        self.model = model
        if self.model is None:
            with self._device_context():
                self.model = build_sam3_stream_model(
                    checkpoint_path=model_path,
                    load_from_HF=False,
                    device=device,
                )
                self.model.tracker.backbone = self.model.detector.backbone
        self.point_state = None
        self.prompt_state = None

    def _device_context(self):
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            return torch.cuda.device(self.device)
        return contextlib.nullcontext()

    def _preprocess(self, frame_rgb):
        img_size = self.model.image_size
        pil_img = Image.fromarray(frame_rgb)
        pil_img = TF.resize(pil_img, size=(img_size, img_size))
        tensor = TF.to_tensor(pil_img).half()
        tensor = (tensor - 0.5) / 0.5
        return tensor.to(self.device)

    def _extract_point_masks(self):
        h, w = self.point_state["output_size"]
        obj_ids = self.point_state["obj_ids"]
        video_res_masks = self.point_state["video_res_masks"]

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

    def _extract_output_masks(self, outputs, image_size: tuple[int, int]) -> list[np.ndarray]:
        if not outputs:
            return []
        masks = outputs.get("out_binary_masks")
        if masks is None:
            return []

        height, width = image_size
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None]

        result = []
        for mask in masks:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            result.append(mask)
        return result

    def _reset_point(self, frame_rgb, points=None, boxes=None, masks=None):
        self.prompt_state = None
        h, w = frame_rgb.shape[:2]
        tracker = self.model.tracker
        inference_state = tracker.init_state(
            video_height=h,
            video_width=w,
            num_frames=1,
            offload_video_to_cpu=True,
        )
        inference_state["images"] = [self._preprocess(frame_rgb)]

        if masks is not None:
            masks = _validate_mask_prompts(masks, frame_rgb)
            for node_idx, mask in enumerate(masks):
                tracker.add_new_mask(
                    inference_state=inference_state, frame_idx=0, obj_id=node_idx,
                    mask=torch.from_numpy(mask).to(self.device),
                )
        elif boxes is not None:
            for node_idx, node_box in enumerate(boxes):
                box = np.asarray(node_box, dtype=np.float32).copy()
                if box.shape != (4,):
                    raise ValueError("each object box must have shape (4,) in XYXY format")
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                np.clip(box, 0, 1, out=box)
                tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=node_idx,
                    box=torch.from_numpy(box),
                )
        else:
            for node_idx, node_points in enumerate(points):
                pts_array = np.array(node_points, dtype=np.float32)
                pts_array[:, 0] /= w
                pts_array[:, 1] /= h
                np.clip(pts_array, 0, 1, out=pts_array)

                tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=node_idx,
                    points=torch.from_numpy(pts_array).to(self.device),
                    labels=torch.ones(len(node_points), dtype=torch.int32, device=self.device),
                )

        obj_ids = None
        video_res_masks = None
        for result in tracker.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            reverse=False,
            propagate_preflight=True,
            tqdm_disable=True,
        ):
            _, obj_ids, _, video_res_masks, _ = result

        self.point_state = {
            "inference_state": inference_state,
            "output_size": (h, w),
            "obj_ids": obj_ids,
            "video_res_masks": video_res_masks,
            "frame_idx": 0,
        }
        return list(masks) if masks is not None else self._extract_point_masks()

    def _update_point(self, frame_rgb):
        if self.point_state is None:
            raise RuntimeError("Point tracker is not initialized; call reset(..., points=...) first")

        tracker = self.model.tracker
        st = self.point_state
        st["frame_idx"] += 1
        frame_idx = st["frame_idx"]
        st["inference_state"]["images"].append(self._preprocess(frame_rgb))
        st["inference_state"]["num_frames"] = frame_idx + 1

        for result in tracker.propagate_in_video(
            st["inference_state"],
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
            reverse=False,
            propagate_preflight=False,
            tqdm_disable=True,
        ):
            _, st["obj_ids"], _, st["video_res_masks"], _ = result

        return self._extract_point_masks()

    def _reset_prompt(self, frame_rgb, prompt):
        self.point_state = None
        self.prompt_state = self.model.init_stream_state()
        frame_idx = self.model.add_frame(self.prompt_state, frame_rgb)
        with torch.inference_mode():
            _, outputs = self.model.add_prompt(
                self.prompt_state,
                frame_idx=frame_idx,
                text_str=str(prompt),
            )
        return self._extract_output_masks(outputs, frame_rgb.shape[:2])

    def _update_prompt(self, frame_rgb):
        if self.prompt_state is None:
            raise RuntimeError("Prompt tracker is not initialized; call reset(..., prompt=...) first")

        frame_idx = self.model.add_frame(self.prompt_state, frame_rgb)
        with torch.inference_mode():
            outputs = self.model.run_single_frame_inference(self.prompt_state, frame_idx=frame_idx)
        return self._extract_output_masks(outputs, frame_rgb.shape[:2])

    def segment_prompt_video(self, frames, prompt: str) -> list[list[np.ndarray]]:
        with self._device_context():
            frames = [np.asarray(frame) for frame in frames]
            if not frames:
                raise ValueError("frames must not be empty")
            frame_masks = [self._reset_prompt(frames[0], prompt)]
            frame_masks.extend(self._update_prompt(frame) for frame in frames[1:])
            return frame_masks

    def reset(self, frame_rgb, points=None, boxes=None, prompt=None, masks=None):
        with self._device_context():
            if sum(value is not None for value in (points, boxes, prompt, masks)) != 1:
                raise ValueError("exactly one of points, boxes, masks, or prompt is required")
            if masks is not None:
                return self._reset_point(frame_rgb, masks=masks)
            if points is not None:
                return self._reset_point(frame_rgb, points)
            if boxes is not None:
                return self._reset_point(frame_rgb, boxes=boxes)
            if prompt is not None:
                return self._reset_prompt(frame_rgb, prompt)

    def update(self, frame_rgb):
        with self._device_context():
            if self.prompt_state is not None:
                return self._update_prompt(frame_rgb)
            return self._update_point(frame_rgb)

    def predict(self, frame_rgb, points=None, boxes=None, prompt=None, anchor_frame=True, masks=None):
        if anchor_frame:
            return self.reset(frame_rgb, points=points, boxes=boxes, prompt=prompt, masks=masks)
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


class NodeSegmenterSAM2:
    """Online SAM2 tracking initialized by labeled points, boxes, or binary masks."""

    class _OnlinePredictor(SAM2VideoPredictor):
        """Adapt Ultralytics' video predictor to repeated single-frame calls."""

        def __init__(self, *args, **kwargs):
            self._reset_requested = True
            self._frame_idx = 0
            super().__init__(*args, **kwargs)

        def init_state(self, predictor=None):
            # This callback runs at the start of every predictor call. Keep the
            # inference state across update calls and rebuild it only for reset.
            if self._reset_requested or not self.inference_state:
                self.inference_state = self._init_state(num_frames=1)
                self._reset_requested = False

        def inference(self, im, *args, **kwargs):
            self.dataset.frame = self._frame_idx
            if self.inference_state:
                self.inference_state["num_frames"] = self._frame_idx + 1
            result = super().inference(im, *args, **kwargs)
            self._frame_idx += 1
            return result

        def request_reset(self):
            self._reset_requested = True
            self._frame_idx = 0

        def _prepare_prompts(self, *args, **kwargs):
            points, labels, masks = super()._prepare_prompts(*args, **kwargs)
            # The video track_step requires a channel axis for mask inputs.
            if masks is not None and masks.ndim == 3:
                masks = masks[:, None]
            return points, labels, masks

    def __init__(self, model_path="/data0/luokang/dataset/luokang/ckpts/sam2/sam2.1_l.pt", device="cuda", mode=None, model=None):
        self.model_path = model_path
        self.device = device
        overrides = {
            "conf": 0.25, "task": "segment", "mode": "predict", "imgsz": 1024,
            "model": model_path, "device": device,
            "quantize": 16 if isinstance(device, str) and device.startswith("cuda") else None,
            "save": False, "verbose": False,
        }
        self.predictor = self._OnlinePredictor(overrides=overrides)
        self.predictor.setup_model(model=model, verbose=False)
        self.model = self.predictor.model
        self.num_objects = 0
        self.output_size = None

    def _extract_masks(self):
        frame_idx = self.predictor._frame_idx - 1
        outputs = self.predictor.inference_state["output_dict"]
        current = outputs["cond_frame_outputs"].get(frame_idx) or outputs["non_cond_frame_outputs"].get(frame_idx)
        if current is None:
            raise RuntimeError(f"SAM2 produced no tracking output for frame {frame_idx}")

        logits = current["pred_masks"]
        scores = current.get("object_score_logits")
        masks = torch.nn.functional.interpolate(
            logits.float(), size=self.output_size, mode="bilinear", align_corners=False
        )[:, 0] > self.model.mask_threshold
        return [
            masks[i].cpu().numpy()
            if scores is None or scores[i].item() > 0
            else np.zeros(self.output_size, dtype=bool)
            for i in range(self.num_objects)
        ]

    def reset(self, frame_rgb, points=None, boxes=None, labels=None, masks=None):
        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must have shape (H, W, 3)")
        if sum(value is not None for value in (points, boxes, masks)) != 1:
            raise ValueError("exactly one of points, boxes, or masks is required")

        predictor_kwargs = {}
        if masks is not None:
            masks = _validate_mask_prompts(masks, frame_rgb)
            self.num_objects = len(masks)
            predictor_kwargs["masks"] = masks.astype(np.uint8)
        elif boxes is not None:
            box_groups = np.asarray(boxes, dtype=np.float32)
            if box_groups.ndim != 2 or box_groups.shape[0] == 0 or box_groups.shape[1] != 4:
                raise ValueError("boxes must have shape (N, 4) in XYXY format")
            self.num_objects = len(box_groups)
            predictor_kwargs["bboxes"] = box_groups.tolist()
        else:
            point_groups = []
            for node_points in points:
                pts = np.asarray(node_points, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
                    raise ValueError("each object must have one or more (x, y) points")
                point_groups.append(pts.tolist())
            self.num_objects = len(point_groups)
            predictor_kwargs["points"] = point_groups
            if labels is None:
                labels = [[1] * len(node_points) for node_points in point_groups]
            if len(labels) != len(point_groups) or any(
                len(group) != len(pts) or any(label not in (0, 1) for label in group)
                for group, pts in zip(labels, point_groups)
            ):
                raise ValueError("labels must match point groups and contain only 0 or 1")
            predictor_kwargs["labels"] = labels

        self.output_size = frame_rgb.shape[:2]
        self.predictor.request_reset()
        self.predictor(
            source=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            **predictor_kwargs,
        )
        return list(masks) if masks is not None else self._extract_masks()

    def update(self, frame_rgb):
        if self.output_size is None:
            raise RuntimeError("SAM2 tracker is not initialized; call reset(..., points=...) first")
        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must have shape (H, W, 3)")
        if frame_rgb.shape[:2] != self.output_size:
            raise ValueError(f"frame size changed from {self.output_size} to {frame_rgb.shape[:2]}")

        self.predictor(source=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        return self._extract_masks()

    def predict(self, frame_rgb, points=None, boxes=None, anchor_frame=True, labels=None, masks=None):
        return self.reset(frame_rgb, points=points, boxes=boxes, labels=labels, masks=masks) if anchor_frame else self.update(frame_rgb)

    def draw_on_image(self, image, masks, labels=None, save_path=None):
        return NodeSegmenter.draw_on_image(self, image, masks, labels, save_path)


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
