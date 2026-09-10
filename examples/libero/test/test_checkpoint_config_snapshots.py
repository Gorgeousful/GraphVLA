from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
from torch.utils.data import DataLoader, Dataset

from src.training.checkpoint import TrainingCheckpoint
from src.training.training import resolve_resume_configs


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
    resume: bool = True
    wandb_name: str | None = None


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


class SnapshotDataset(Dataset):
    def __init__(self, config):
        self.config = config

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"path": str(self.config.dataset_dir),
                "enabled": self.config.nested.enabled,
                "task": self.config.to_kwargs()["tasks"][0]}


def test_snapshot_config_survives_spawn_dataloader(tmp_path: Path) -> None:
    path = tmp_path / "data_config.py"
    path.write_text(CONFIG_SOURCE, encoding="utf-8")
    config = TrainingCheckpoint.load_config_symbol(path, "LIBERO_DATA_CONFIG")
    # Runtime changes must survive too, rather than reloading the saved instance.
    config.dataset_dir = Path("/runtime/dataset")
    config.tasks = [7]
    config.nested.enabled = False
    loader = DataLoader(SnapshotDataset(config), num_workers=1,
                        multiprocessing_context="spawn", timeout=30)
    batches = list(loader)
    assert batches[0]["path"] == ["/runtime/dataset"]
    assert batches[0]["task"].item() == 7
    assert not batches[0]["enabled"].item()


def test_resume_without_snapshots_uses_current_configs(tmp_path: Path) -> None:
    source_path = tmp_path / "source_config.py"
    source_path.write_text(CONFIG_SOURCE, encoding="utf-8")
    source = load_module(source_path, "test_resume_current_config_source")
    configs = (
        source.DataConfig(dataset_dir=Path("/current/dataset")),
        source.ModelConfig(hidden_dim=128),
        source.TrainingConfig(save_dir=tmp_path, wandb_name="run"),
    )

    assert resolve_resume_configs(*configs) == configs


@pytest.mark.parametrize("has_checkpoint", [False, True])
def test_resume_with_snapshots_requires_weights(tmp_path: Path, has_checkpoint: bool) -> None:
    source_path = tmp_path / "source_config.py"
    source_path.write_text(CONFIG_SOURCE, encoding="utf-8")
    source = load_module(source_path, "test_resume_saved_config_source")
    experiment_dir = tmp_path / "run"
    checkpoint = TrainingCheckpoint(experiment_dir, resume=True, keep_period=100)
    checkpoint.save_config_snapshots(
        data_config=source.DataConfig(dataset_dir=Path("/saved/dataset"), tasks=[3]),
        model_config=source.ModelConfig(hidden_dim=512),
        training_config=source.TrainingConfig(
            max_steps=30_000,
            save_dir=experiment_dir,
            resume=False,
            wandb_name="run",
        ),
    )
    current_configs = (
        source.DataConfig(dataset_dir=Path("/current/dataset"), tasks=[9]),
        source.ModelConfig(hidden_dim=128),
        source.TrainingConfig(max_steps=60_000, save_dir=tmp_path, wandb_name="run"),
    )
    # Only step-numbered checkpoint files participate in automatic resume.
    (checkpoint.ckpt_dir / "step_invalid.pt").touch()
    (checkpoint.ckpt_dir / "step_2.pt").mkdir()
    if has_checkpoint:
        (checkpoint.ckpt_dir / "step_1.pt").touch()
    else:
        assert resolve_resume_configs(*current_configs) == current_configs
        return

    data_config, model_config, training_config = resolve_resume_configs(*current_configs)

    assert data_config.dataset_dir == Path("/saved/dataset")
    assert data_config.tasks == [3]
    assert model_config.hidden_dim == 512
    assert training_config.max_steps == 60_000
    assert training_config.save_dir == experiment_dir
    assert training_config.resume is True


@pytest.mark.parametrize("has_checkpoint", [False, True])
def test_resume_with_partial_snapshots_fails_only_with_weights(tmp_path: Path, has_checkpoint: bool) -> None:
    source_path = tmp_path / "source_config.py"
    source_path.write_text(CONFIG_SOURCE, encoding="utf-8")
    source = load_module(source_path, "test_resume_partial_config_source")
    config_dir = tmp_path / "run" / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "data_config.py").write_text(CONFIG_SOURCE, encoding="utf-8")
    configs = (
        source.DataConfig(dataset_dir=Path("/current/dataset")),
        source.ModelConfig(),
        source.TrainingConfig(save_dir=tmp_path, wandb_name="run"),
    )
    if not has_checkpoint:
        assert resolve_resume_configs(*configs) == configs
        return
    ckpt_dir = tmp_path / "run" / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "step_1.pt").touch()
    with pytest.raises(FileNotFoundError, match="snapshots are incomplete"):
        resolve_resume_configs(*configs)
