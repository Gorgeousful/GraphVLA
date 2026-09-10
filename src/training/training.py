"""Training loop for GraphVLA policies."""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from copy import copy
from datetime import timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from rich.console import Console

from src.dataset.dataset import GenericDataLoader
from src.policy.registry import SUPPORTED_POLICIES, build_policy
from src.training.checkpoint import TrainingCheckpoint
from src.training.distributed import TrainingDistributed
from src.training.logger import TrainingLogger
from src.training.optimizer import TrainingOptimizer

cs = Console()


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def build_model(model_config: Any, training_config: Any, device: torch.device) -> torch.nn.Module:
    model = build_policy(model_config).to(device)
    if getattr(training_config, "gradient_checkpointing", False):
        model.set_gradient_checkpointing(True)
    if getattr(training_config, "compile_model", False):
        model = torch.compile(model, dynamic=False)
    return model


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def resolve_experiment_dir(training_config: Any) -> Path | None:
    if training_config.save_dir is None:
        return None
    if not training_config.wandb_name:
        raise ValueError("wandb_name is required when save_dir is set")
    save_dir = Path(training_config.save_dir)
    return save_dir if save_dir.name == training_config.wandb_name else save_dir / training_config.wandb_name


def resolve_resume_configs(
    data_config: Any,
    model_config: Any,
    training_config: Any,
) -> tuple[Any, Any, Any]:
    experiment_dir = resolve_experiment_dir(training_config)
    if not training_config.resume or experiment_dir is None:
        return data_config, model_config, training_config

    has_checkpoint = any(
        path.is_file() and TrainingCheckpoint.step_from_path(path) is not None
        for path in (experiment_dir / "checkpoints").glob("step_*.pt")
    )
    if not has_checkpoint:
        return data_config, model_config, training_config

    config_dir = experiment_dir / "configs"
    snapshot_paths = [
        config_dir / filename
        for filename in ("data_config.py", "model_config.py", "training_config.py")
    ]
    existing_snapshots = [path for path in snapshot_paths if path.is_file()]
    if not existing_snapshots:
        return data_config, model_config, training_config
    if len(existing_snapshots) != len(snapshot_paths):
        missing = [str(path) for path in snapshot_paths if not path.is_file()]
        raise FileNotFoundError(f"resume config snapshots are incomplete; missing: {missing}")

    loaded_data, loaded_model, loaded_training = TrainingCheckpoint.load_config_snapshots(
        experiment_dir / "checkpoints" / "resume.pt"
    )
    loaded_experiment_dir = resolve_experiment_dir(loaded_training)
    if loaded_experiment_dir != experiment_dir:
        raise ValueError(
            f"resume training snapshot points to {loaded_experiment_dir}, "
            f"expected {experiment_dir}"
        )
    loaded_training.resume = True
    loaded_training.max_steps = training_config.max_steps
    cs.print(f"[green]loaded resume config snapshots from {config_dir}[/green]")
    return loaded_data, loaded_model, loaded_training


def train(data_config: Any, model_config: Any, training_config: Any) -> torch.nn.Module:
    training_config = copy(training_config)
    if bool(getattr(data_config, "action_delta", True)) != bool(getattr(model_config, "action_delta", True)):
        raise ValueError("data_config.action_delta must match model_config.action_delta")
    data_camera_keys = getattr(data_config, "camera_keys", None)
    model_camera_keys = getattr(model_config, "camera_keys", None)
    if data_camera_keys is not None and tuple(data_camera_keys) != tuple(model_camera_keys or ()):
        raise ValueError("data_config.camera_keys must match model_config.camera_keys")
    training_config.save_dir = resolve_experiment_dir(training_config)

    distributed = TrainingDistributed(
        enabled=getattr(training_config, "use_ddp", True),
        backend=getattr(training_config, "distributed_backend", "nccl"),
    )
    seed = training_config.seed + distributed.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = distributed.device
    if distributed.is_main_process: 
        cs.print("Distributed init!")

    checkpoint = TrainingCheckpoint(
        save_dir=training_config.save_dir,
        resume=training_config.resume,
        keep_period=training_config.keep_period,
        is_main_process=distributed.is_main_process,
        barrier=distributed.barrier,
        rank=distributed.rank,
        world_size=distributed.world_size,
        gather_object=distributed.gather_object,
    )
    if distributed.is_main_process:
        cs.print("Checkpoint Manager init!")
    checkpoint.save_config_snapshots(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
    )

    dataloader = GenericDataLoader(
        data_config,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
        distributed=distributed.enabled,
        world_size=distributed.world_size,
        rank=distributed.rank,
        seed=training_config.seed,
    )
    if distributed.is_main_process: 
        cs.print("DataLoader init!")

    logger = TrainingLogger(
        training_config,
        data_config=data_config,
        model_config=model_config,
        is_main_process=distributed.is_main_process,
    )
    if distributed.is_main_process: 
        cs.print("Logger init!")

    base_model = build_model(model_config, training_config, device)
    checkpoint.load_pretrained(
        getattr(training_config, "ckpt_path", None),
        base_model,
        device,
    )
    model = distributed.wrap_model(base_model, device)
    train_optimizer = TrainingOptimizer(model, training_config)
    model.train()
    train_optimizer.zero_grad()
    if distributed.is_main_process: 
        cs.print("Model and Optimizer init!")


    step = checkpoint.load_latest(
        model, train_optimizer, device, seed=training_config.seed,
    ) if training_config.resume else 0
    accum_steps = max(1, training_config.gradient_accumulation_steps)
    micro_step = step * accum_steps
    data_epoch, batch_offset = dataloader.resume_position(micro_step)
    training_start_step = step
    training_start_time = time.perf_counter()
    while step < training_config.max_steps:
        for batch in dataloader.iter_epoch(data_epoch, skip_batches=batch_offset):
            batch_offset = 0
            lr = train_optimizer.set_step_lr(step)
            batch = move_to_device(batch, device)
            micro_step += 1

            with amp_context(device, training_config.use_amp):
                loss, metrics = model(batch)
                loss = loss / accum_steps
            loss.backward()

            if micro_step % accum_steps != 0:
                continue

            clip_norm = training_config.gradient_clip_norm
            if clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            train_optimizer.step()
            train_optimizer.zero_grad()
            step += 1

            if step % training_config.log_interval == 0:
                metrics = {**metrics, "grad_norm": grad_norm.detach()}
                logs = distributed.reduce_metrics(metrics)
                if distributed.is_main_process:
                    elapsed_seconds = time.perf_counter() - training_start_time
                    completed_steps = step - training_start_step
                    remaining_seconds = elapsed_seconds / completed_steps * (training_config.max_steps - step)
                    elapsed = timedelta(seconds=int(elapsed_seconds))
                    eta = timedelta(seconds=int(remaining_seconds))
                    log_text = " ".join(f"{key}={value:.4f}" for key, value in sorted(logs.items()))
                    cs.rule()
                    cs.print(f"step={step} lr={lr:.3e} elapsed={elapsed} eta={eta}\n{log_text}")
                    logger.log(step=step, metrics=logs, lr=lr)

            if step % training_config.save_interval == 0:
                path = checkpoint.save(
                    model,
                    train_optimizer,
                    step,
                )
                if distributed.is_main_process:
                    cs.print(f"[green]saved checkpoint {path}[/green]")

            if step >= training_config.max_steps:
                break
        data_epoch += 1

    logger.finish()
    return model


def load_example_configs(example: str, policy: str) -> tuple[Any, Any, Any]:
    example = example.lower()
    policy = policy.lower()
    if (example, policy) == ("libero", "point_policy"):
        from examples.libero.config.point_policy.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.point_policy.model_config import LIBERO_MODEL_CONFIG
        from examples.libero.config.point_policy.training_config import LIBERO_TRAINING_CONFIG

        return LIBERO_DATA_CONFIG, LIBERO_MODEL_CONFIG, LIBERO_TRAINING_CONFIG
    if (example, policy) == ("libero", "graphpoint"):
        from examples.libero.config.graphpoint.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.graphpoint.model_config import LIBERO_MODEL_CONFIG
        from examples.libero.config.graphpoint.training_config import (
            LIBERO_TRAINING_CONFIG,
        )
        return LIBERO_DATA_CONFIG, LIBERO_MODEL_CONFIG, LIBERO_TRAINING_CONFIG
    if (example, policy) == ("libero", "act"):
        from examples.libero.config.act.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.act.model_config import LIBERO_MODEL_CONFIG
        from examples.libero.config.act.training_config import LIBERO_TRAINING_CONFIG

        return LIBERO_DATA_CONFIG, LIBERO_MODEL_CONFIG, LIBERO_TRAINING_CONFIG
    if (example, policy) == ("libero", "dp"):
        from examples.libero.config.dp.data_config import (
            LIBERO_DATA_CONFIG,
        )
        from examples.libero.config.dp.model_config import (
            LIBERO_MODEL_CONFIG,
        )
        from examples.libero.config.dp.training_config import (
            LIBERO_TRAINING_CONFIG,
        )

        return LIBERO_DATA_CONFIG, LIBERO_MODEL_CONFIG, LIBERO_TRAINING_CONFIG
    if (example, policy) == ("libero", "dp3"):
        from examples.libero.config.dp3.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.dp3.model_config import LIBERO_MODEL_CONFIG
        from examples.libero.config.dp3.training_config import LIBERO_TRAINING_CONFIG

        return LIBERO_DATA_CONFIG, LIBERO_MODEL_CONFIG, LIBERO_TRAINING_CONFIG
    raise ValueError(f"Unsupported example/policy combination: {(example, policy)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphVLA.")
    parser.add_argument(
        "--example",
        default="libero",
        choices=("libero",),
        help="Example config to train with.",
    )
    parser.add_argument(
        "--policy",
        default="graphpoint",
        choices=SUPPORTED_POLICIES,
        help="Policy configuration to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_config, model_config, training_config = load_example_configs(args.example, args.policy)
    data_config, model_config, training_config = resolve_resume_configs(
        data_config,
        model_config,
        training_config,
    )
    train(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
    )
