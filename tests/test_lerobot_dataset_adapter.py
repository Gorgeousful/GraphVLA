import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import src.dataset.dataset as dataset_module

def test_feature_only_dataset_matches_lerobot_query_hooks() -> None:
    for method_name in ("_query_videos", "_query_hf_dataset"):
        base_parameters = inspect.signature(getattr(dataset_module.LeRobotDataset, method_name)).parameters
        override_parameters = inspect.signature(
            getattr(dataset_module._FeatureOnlyLeRobotDataset, method_name)
        ).parameters
        assert tuple(override_parameters) == tuple(base_parameters)

    dataset = object.__new__(dataset_module._FeatureOnlyLeRobotDataset)
    assert dataset._query_videos({"camera": [0.0]}, 0) == {}


def test_make_lerobot_dataset_selects_video_behavior(monkeypatch) -> None:
    class RegularDataset:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FeatureOnlyDataset(RegularDataset):
        pass

    monkeypatch.setattr(dataset_module, "LeRobotDataset", RegularDataset)
    monkeypatch.setattr(dataset_module, "_FeatureOnlyLeRobotDataset", FeatureOnlyDataset)

    regular = dataset_module.make_lerobot_dataset(Path("dataset"))
    feature_only = dataset_module.make_lerobot_dataset(Path("dataset"), load_videos=False)

    assert type(regular) is RegularDataset
    assert type(feature_only) is FeatureOnlyDataset


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


def test_feature_only_dataset_groups_equal_horizon_queries() -> None:
    class FakeHFDataset:
        def __init__(self) -> None:
            self.select_calls: list[list[int]] = []
            self.format_calls: list[str] = []
            self.formatted_batches: list[dict[str, np.ndarray]] = []

        def select(self, indices: list[int]):
            self.select_calls.append(list(indices))
            owner = self

            class Selection(dict):
                def with_format(self, format_name: str):
                    owner.format_calls.append(format_name)
                    formatted = {key: np.asarray(values) for key, values in self.items()}
                    owner.formatted_batches.append(formatted)
                    return formatted

            return Selection(
                first=[index for index in indices],
                second=[index + 10 for index in indices],
                different=[index + 20 for index in indices],
                camera=[index + 30 for index in indices],
            )

    dataset = object.__new__(dataset_module._FeatureOnlyLeRobotDataset)
    dataset.meta = SimpleNamespace(video_keys=["camera"])
    dataset.hf_dataset = FakeHFDataset()

    result = dataset._query_hf_dataset(
        {
            "first": [0, 1],
            "different": [1, 2],
            "second": [0, 1],
            "camera": [0, 1],
        }
    )

    assert dataset.hf_dataset.select_calls == [[0, 1], [1, 2]]
    assert dataset.hf_dataset.format_calls == ["numpy", "numpy"]
    assert list(result) == ["first", "different", "second"]
    torch.testing.assert_close(result["first"], torch.tensor([0, 1]))
    torch.testing.assert_close(result["second"], torch.tensor([10, 11]))
    torch.testing.assert_close(result["different"], torch.tensor([21, 22]))
    assert "camera" not in result

    result["first"][0] = -1
    assert dataset.hf_dataset.formatted_batches[0]["first"][0] == 0
