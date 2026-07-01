"""Task heads for decoded query tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHeads(nn.Module):
    """A small collection of linear heads keyed by output name."""

    def __init__(self, hidden_dim: int, output_dims: dict[str, int]) -> None:
        super().__init__()
        if not output_dims:
            raise ValueError("output_dims must contain at least one head")
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, dim) for name, dim in output_dims.items()})

    def forward(self, decoded_queries: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(decoded_queries) for name, head in self.heads.items()}
