from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from src.training.checkpoint import TrainingCheckpoint


CONFIG_SOURCE = '''\
from dataclasses import dataclass, field
from pathlib import Path

LIBERO_DATASET_DIR = "default_dataset"


@dataclass
class NestedConfig:
    enabled: bool = True


@dataclass
class DataConfig:
    dataset_dir: Path
    tasks: list[int] | None = None
    nested: NestedConfig = field(default_factory=NestedConfig)

    def to_kwargs(self):
        return vars(self)


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    labels: tuple[str, ...] = ("default",)

    def to_kwargs(self):
        return vars(self)


@dataclass
class TrainingConfig:
    max_steps: int = 10
    save_dir: Path | None = None


LIBERO_DATA_CONFIG = DataConfig(dataset_dir=Path(LIBERO_DATASET_DIR))
LIBERO_MODEL_CONFIG = ModelConfig()
LIBERO_TRAINING_CONFIG = TrainingConfig()
'''


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assignment_count(source: str, symbol: str) -> int:
    count = 0
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == symbol for target in node.targets
        ):
            count += 1
    return count


def test_config_snapshots_replace_original_assignments_with_resolved_values(tmp_path: Path) -> None:
    source_path = tmp_path / "source_config.py"
    source_path.write_text(CONFIG_SOURCE, encoding="utf-8")
    source = load_module(source_path, "test_resolved_config_source")

    data_config = source.DataConfig(
        dataset_dir=Path("/resolved/dataset"),
        tasks=[2, 5],
        nested=source.NestedConfig(enabled=False),
    )
    model_config = source.ModelConfig(hidden_dim=512, labels=("final", "config"))
    training_config = source.TrainingConfig(max_steps=123, save_dir=tmp_path)

    checkpoint = TrainingCheckpoint(tmp_path, resume=True, keep_period=100)
    checkpoint.save_config_snapshots(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
    )

    for filename, symbol in (
        ("data_config.py", "LIBERO_DATA_CONFIG"),
        ("model_config.py", "LIBERO_MODEL_CONFIG"),
        ("training_config.py", "LIBERO_TRAINING_CONFIG"),
    ):
        snapshot_source = (tmp_path / "configs" / filename).read_text(encoding="utf-8")
        assert assignment_count(snapshot_source, symbol) == 1

    loaded_data, loaded_model, loaded_training = TrainingCheckpoint.load_config_snapshots(
        tmp_path / "checkpoints" / "step_1.pt"
    )
    assert loaded_data.dataset_dir == Path("/resolved/dataset")
    assert loaded_data.tasks == [2, 5]
    assert loaded_data.nested.enabled is False
    assert loaded_model.hidden_dim == 512
    assert loaded_model.labels == ("final", "config")
    assert loaded_training.max_steps == 123


def test_load_config_snapshots_requires_experiment_configs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data_config.py"):
        TrainingCheckpoint.load_config_snapshots(tmp_path / "checkpoints" / "step_1.pt")
