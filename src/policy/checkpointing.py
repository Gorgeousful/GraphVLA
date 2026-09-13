"""Activation checkpointing shared by policies; checkpoint weights are unchanged."""

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class GradientCheckpointingMixin:
    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        for module in self.modules():
            module.gradient_checkpointing = bool(enabled)


@contextmanager
def _recompute_batch_norm(batch_norms):
    # Recompute with private running buffers. Do not mutate buffers that a prior
    # forward/backward may reference, or count the recompute as a training batch.
    saved = []
    try:
        for module in batch_norms:
            for name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, name)
                if value is not None:
                    saved.append((module, name, value))
                    setattr(module, name, value.clone())
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def checkpoint_module(module: nn.Module, *args, **kwargs):
    if not (getattr(module, "gradient_checkpointing", False) and module.training and torch.is_grad_enabled()):
        return module(*args, **kwargs)
    batch_norms = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
                  and m.training and m.track_running_stats]
    options = {}
    if batch_norms:
        options["context_fn"] = lambda: (nullcontext(), _recompute_batch_norm(batch_norms))
    return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=True, **options, **kwargs)
