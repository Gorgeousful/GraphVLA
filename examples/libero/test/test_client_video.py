from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from examples.libero.eval.client import _save_video_ffmpeg


class _FakeFfmpegProcess:
    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO()
        self.returncode = 0

    def wait(self) -> None:
        return None


def test_save_video_recreates_missing_output_directory(tmp_path, monkeypatch) -> None:
    save_path = tmp_path / "deleted-during-rollout" / "video.mp4"

    def fake_popen(command, **_kwargs):
        output_path = Path(command[-1])
        assert output_path == save_path
        assert output_path.parent.is_dir()
        return _FakeFfmpegProcess()

    monkeypatch.setattr("examples.libero.eval.client.subprocess.Popen", fake_popen)

    _save_video_ffmpeg(
        [np.zeros((4, 4, 3), dtype=np.uint8)],
        save_path,
        fps=10.0,
    )
