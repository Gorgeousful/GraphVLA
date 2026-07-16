from types import SimpleNamespace

import numpy as np

from src.module.node_segmenter import NodeSegmenter


class FakeSam3StreamModel:
    image_size = 4

    def __init__(self):
        backbone = object()
        self.detector = SimpleNamespace(backbone=backbone)
        self.tracker = SimpleNamespace(backbone=backbone)

    def init_stream_state(self):
        return {"frames": [], "prompt": None}

    def add_frame(self, inference_state, raw_image):
        inference_state["frames"].append(np.asarray(raw_image))
        return len(inference_state["frames"]) - 1

    def add_prompt(self, inference_state, frame_idx, text_str):
        inference_state["prompt"] = text_str
        return frame_idx, self._output(frame_idx)

    def run_single_frame_inference(self, inference_state, frame_idx):
        assert inference_state["prompt"] == "table"
        return self._output(frame_idx)

    @staticmethod
    def _output(frame_idx):
        mask = np.zeros((4, 4), dtype=bool)
        mask[frame_idx : frame_idx + 2, 1:3] = True
        return {"out_binary_masks": mask[None]}


def test_prompt_video_and_online_tracking_produce_the_same_masks():
    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]

    offline = NodeSegmenter(device="cpu", model=FakeSam3StreamModel())
    offline_masks = offline.segment_prompt_video(frames, prompt="table")

    online = NodeSegmenter(device="cpu", model=FakeSam3StreamModel())
    online_masks = [online.predict(frames[0], prompt="table", anchor_frame=True)]
    online_masks.append(online.predict(frames[1], anchor_frame=False))

    assert len(offline_masks) == len(online_masks) == 2
    for offline_frame, online_frame in zip(offline_masks, online_masks):
        assert len(offline_frame) == len(online_frame) == 1
        np.testing.assert_array_equal(offline_frame[0], online_frame[0])
