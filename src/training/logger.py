"""Training metric logging utilities."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any


class TrainingLogger:
    def __init__(
        self,
        training_config: Any,
        *,
        data_config: Any | None = None,
        model_config: Any | None = None,
        is_main_process: bool = True,
    ) -> None:
        self.backend = getattr(training_config, "log_backend", "wandb")
        self.run = None
        if not is_main_process or self.backend is None:
            return

        backend = str(self.backend).lower()
        if backend in {"none", "disabled", "off"}:
            return
        if backend != "wandb":
            raise ValueError(f"Unsupported log_backend={self.backend!r}")

        try:
            import wandb
        except ImportError as exc:
            raise ImportError("log_backend='wandb' requires wandb to be installed") from exc

        save_dir = getattr(training_config, "save_dir", None)
        project = getattr(training_config, "wandb_project", "GraphVLA")
        entity = getattr(training_config, "wandb_entity", None)
        name = getattr(training_config, "wandb_name", None)
        if name is None and save_dir is not None:
            name = Path(save_dir).name

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            dir=str(save_dir) if save_dir is not None else None,
            config={
                "data": self.config_to_plain(data_config),
                "model": self.config_to_plain(model_config),
                "training": self.config_to_plain(training_config),
            },
            resume="allow",
        )

    def log(self, *, step: int, metrics: dict[str, float], lr: float) -> None:
        if self.run is None:
            return
        payload = {"train/lr": lr}
        payload.update({f"train/{key}": value for key, value in metrics.items()})
        self.run.log(payload, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    @classmethod
    def config_to_plain(cls, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "to_kwargs") and callable(value.to_kwargs):
            return cls.config_to_plain(value.to_kwargs())
        if is_dataclass(value):
            return cls.config_to_plain(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls.config_to_plain(item) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return [cls.config_to_plain(item) for item in value]
        if isinstance(value, str | int | float | bool):
            return value
        return repr(value)
