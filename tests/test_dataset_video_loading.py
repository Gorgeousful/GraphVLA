import inspect
from pathlib import Path
from types import SimpleNamespace
import src.dataset.dataset as dataset_module

def test_video_free_dataset_matches_lerobot_query_hook() -> None:
    base_parameters = inspect.signature(dataset_module.LeRobotDataset._query_videos).parameters
    override_parameters = inspect.signature(dataset_module._VideoFreeLeRobotDataset._query_videos).parameters
    assert tuple(override_parameters) == tuple(base_parameters)

    dataset = object.__new__(dataset_module._VideoFreeLeRobotDataset)
    assert dataset._query_videos({"camera": [0.0]}, 0) == {}


def test_make_lerobot_dataset_selects_video_behavior(monkeypatch) -> None:
    class RegularDataset:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class VideoFreeDataset(RegularDataset):
        pass

    monkeypatch.setattr(dataset_module, "LeRobotDataset", RegularDataset)
    monkeypatch.setattr(dataset_module, "_VideoFreeLeRobotDataset", VideoFreeDataset)

    regular = dataset_module.make_lerobot_dataset(Path("dataset"))
    video_free = dataset_module.make_lerobot_dataset(Path("dataset"), load_videos=False)

    assert type(regular) is RegularDataset
    assert type(video_free) is VideoFreeDataset


def test_generic_dataset_defaults_to_loading_videos(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeDataset:
        def __len__(self) -> int:
            return 0

    def fake_make_lerobot_dataset(dataset_dir, **kwargs):
        captured.update(kwargs)
        return FakeDataset()

    monkeypatch.setattr(dataset_module, "make_lerobot_dataset", fake_make_lerobot_dataset)
    config = SimpleNamespace(
        dataset_dir=tmp_path,
        episodes=None,
        video_backend="pyav",
        horizon=None,
        tasks=None,
        transforms=(),
    )

    dataset_module.GenericDataset(config)

    assert captured["load_videos"] is True
