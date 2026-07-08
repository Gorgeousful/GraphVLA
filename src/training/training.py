"""Training loop for GraphVLA models."""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
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
from src.model.model import PointQueryModel
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
    model = PointQueryModel(**model_config.to_kwargs()).to(device)
    if getattr(training_config, "gradient_checkpointing", False):
        model.set_gradient_checkpointing(True)
    if getattr(training_config, "compile_model", False):
        model = torch.compile(model)
    return model


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def train(data_config: Any, model_config: Any, training_config: Any) -> torch.nn.Module:
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
    checkpoint = TrainingCheckpoint(
        save_dir=training_config.save_dir,
        resume=training_config.resume,
        keep_period=training_config.keep_period,
        is_main_process=distributed.is_main_process,
        barrier=distributed.barrier,
    )
    base_model = build_model(model_config, training_config, device)
    checkpoint.load_pretrained(getattr(training_config, "ckpt_path", None), base_model, device)
    model = distributed.wrap_model(base_model, device)
    train_optimizer = TrainingOptimizer(model, training_config)
    step = checkpoint.load_latest(model, train_optimizer, device) if training_config.resume else 0
    logger = TrainingLogger(
        training_config,
        data_config=data_config,
        model_config=model_config,
        is_main_process=distributed.is_main_process,
    )
    accum_steps = max(1, training_config.gradient_accumulation_steps)
    micro_step = step * accum_steps
    data_epoch, batch_offset = dataloader.resume_position(micro_step)

    model.train()
    train_optimizer.zero_grad()

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

            clip_norm = training_config.gradient_clip_l2_norm
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
                    log_text = " ".join(f"{key}={value:.4f}" for key, value in sorted(logs.items()))
                    cs.print(f"step={step} lr={lr:.3e} {log_text}")
                    logger.log(step=step, metrics=logs, lr=lr)

            if step % training_config.save_interval == 0 and distributed.is_main_process:
                path = checkpoint.save(
                    model,
                    train_optimizer,
                    step,
                    data_config=data_config,
                    model_config=model_config,
                    training_config=training_config,
                )
                cs.print(f"[green]saved checkpoint {path}[/green]")

            if step >= training_config.max_steps:
                break
        data_epoch += 1

    logger.finish()
    return model


def load_example_configs(example: str) -> tuple[Any, Any, Any]:
    example = example.lower()
    if example == "libero":
        from examples.libero.config.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.model_config import LIBERO_MODEL_CONFIG
        from examples.libero.config.training_config import LIBERO_TRAINING_CONFIG
        return LIBERO_DATA_CONFIG, LIBERO_MODEL_CONFIG, LIBERO_TRAINING_CONFIG
    raise ValueError(f"Unsupported example: {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphVLA.")
    parser.add_argument(
        "--example",
        default="libero",
        choices=("libero",),
        help="Example config to train with.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_config, model_config, training_config = load_example_configs(args.example)
    train(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
    )
