"""Checkpoint loading, saving, and retention management."""

from __future__ import annotations

import random
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

cs = Console()


class TrainingCheckpoint:
    def __init__(
        self,
        save_dir: str | Path | None,
        *,
        resume: bool,
        keep_period: int,
        is_main_process: bool = True,
        barrier: Any | None = None,
    ) -> None:
        self.root = Path(save_dir) if save_dir is not None else Path("result")
        self.ckpt_dir = self.root / "checkpoints"
        self.config_dir = self.root / "configs"
        self.keep_period = keep_period
        self.is_main_process = is_main_process
        self.barrier = barrier
        if self.is_main_process:
            self.prepare_dir(resume=resume)
        if self.barrier is not None:
            self.barrier()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dir(self, *, resume: bool) -> None:
        existing = self.list_checkpoints()
        if existing and not resume:
            answer = input(f"{self.ckpt_dir} already has checkpoints. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                raise RuntimeError(f"Please use a different save_dir or set resume=True: {self.root}")
            for path in existing:
                path.unlink()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> list[Path]:
        if not self.ckpt_dir.exists():
            return []
        paths = [path for path in self.ckpt_dir.glob("step_*.pt") if self.step_from_path(path) is not None]
        return sorted(paths, key=lambda path: self.step_from_path(path) or -1)

    def latest_checkpoint(self) -> Path | None:
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def load_pretrained(self, ckpt_path: str | Path | None, model: torch.nn.Module, device: torch.device) -> None:
        if ckpt_path is None:
            return
        path = Path(ckpt_path)
        if not path.exists():
            if self.is_main_process:
                cs.print(f"[yellow]pretrained checkpoint not found, skip: {path}[/yellow]")
            return

        state = torch.load(path, map_location=device, weights_only=False)
        incompatible = self.model_for_state(model).load_state_dict(self.unwrap_model_state(state), strict=False)
        if self.is_main_process:
            cs.print(
                f"[green]loaded pretrained weights from {path}[/green] "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )

    def load_latest(self, model: torch.nn.Module, optimizer: Any, device: torch.device) -> int:
        path = self.latest_checkpoint()
        if path is None:
            return 0
        state = torch.load(path, map_location=device, weights_only=False)
        self.model_for_state(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        self.restore_rng_state(state.get("rng"))
        step = int(state.get("step", self.step_from_path(path) or 0))
        if self.is_main_process:
            cs.print(f"[green]resumed from {path} at step {step}[/green]")
        return step

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        step: int,
        *,
        data_config: Any,
        model_config: Any,
        training_config: Any,
    ) -> Path:
        configs = {
            "data": self.config_to_state(data_config),
            "model": self.config_to_state(model_config),
            "training": self.config_to_state(training_config),
        }
        self.save_configs(configs)
        path = self.ckpt_dir / f"step_{step}.pt"
        state = {
            "step": step,
            "model": self.model_for_state(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": self.rng_state(),
        }
        torch.save(state, path)
        self.cleanup(current_step=step)
        return path

    def cleanup(self, *, current_step: int) -> None:
        for path in self.list_checkpoints():
            step = self.step_from_path(path)
            if step is None or step == current_step:
                continue
            if self.keep_period > 0 and step % self.keep_period == 0:
                continue
            path.unlink()

    def save_configs(self, configs: Mapping[str, Any]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        for name, config in configs.items():
            path = self.config_dir / f"{name}_config.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def config_to_state(cls, config: Any) -> Any:
        if hasattr(config, "to_kwargs") and callable(config.to_kwargs):
            return cls.to_plain_value(config.to_kwargs())
        if is_dataclass(config):
            return cls.to_plain_value(asdict(config))
        return cls.to_plain_value(config)

    @classmethod
    def to_plain_value(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return cls.to_plain_value(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): cls.to_plain_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [cls.to_plain_value(item) for item in value]
        if isinstance(value, list):
            return [cls.to_plain_value(item) for item in value]
        if value is None or isinstance(value, str | int | float | bool):
            return value
        return repr(value)

    @staticmethod
    def step_from_path(path: Path) -> int | None:
        stem = path.stem
        if not stem.startswith("step_"):
            return None
        try:
            return int(stem.removeprefix("step_"))
        except ValueError:
            return None

    @staticmethod
    def model_for_state(model: torch.nn.Module) -> torch.nn.Module:
        current = model
        while True:
            if hasattr(current, "module"):
                current = current.module
                continue
            if hasattr(current, "_orig_mod"):
                current = current._orig_mod
                continue
            if hasattr(current, "model"):
                current = current.model
                continue
            return current

    @staticmethod
    def unwrap_model_state(state: Any) -> Mapping[str, torch.Tensor]:
        if isinstance(state, Mapping):
            for key in ("model", "state_dict", "model_state_dict"):
                value = state.get(key)
                if isinstance(value, Mapping):
                    return value
            return state
        raise TypeError(f"Unsupported checkpoint state type: {type(state)!r}")

    @staticmethod
    def rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def restore_rng_state(state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
