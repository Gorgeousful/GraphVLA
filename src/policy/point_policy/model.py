"""Language-conditioned LIBERO adaptation of Point-Policy's deterministic actor.

Architecture adapted from Point-Policy (Siddhant Haldar, MIT) and its nanoGPT
backbone (Andrej Karpathy, MIT). See LICENSE and LICENSE.nanoGPT in this folder.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.policy.language import FrozenBgeClsEncoder
from src.policy.checkpointing import GradientCheckpointingMixin, checkpoint_module


class PointBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.ln_1 = nn.LayerNorm(dim)
        self.c_attn = nn.Linear(dim, dim * 3)
        self.c_proj = nn.Linear(dim, dim)
        self.resid_dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        batch, count, dim = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, count, self.heads, dim // self.heads).transpose(1, 2)
                   for a in (q, k, v)]
        # Original Point-Policy uses non-causal attention over point identities.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(batch, count, dim)
        x = x + self.resid_dropout(self.c_proj(y))
        return x + self.mlp(self.ln_2(x))


class PointPolicy(GradientCheckpointingMixin, nn.Module):
    def __init__(self, *, num_points=32, max_objects=10, actor_point_indices=(0, 1, 2, 3, 4, 5),
                 history_horizon=9, future_horizon=10, repr_dim=512, hidden_dim=256,
                 num_layers=4, num_heads=2, dropout=0.1, stddev=0.1,
                 language_dim=384, language_model_path: str | Path | None = None):
        super().__init__()
        if min(num_points, max_objects, future_horizon, num_layers, num_heads) < 1:
            raise ValueError("Point counts, horizon and layer/head counts must be positive")
        if history_horizon < 0 or hidden_dim % num_heads or stddev <= 0:
            raise ValueError("Invalid history, hidden dimension or Gaussian stddev")
        self.num_robot_points = len(actor_point_indices)
        self.num_points = num_points
        self.max_objects = max_objects
        self.history_len = history_horizon + 1
        self.future_horizon = future_horizon
        self.stddev = stddev
        self.num_tracks = self.num_robot_points + max_objects * num_points
        self.point_projector = nn.Linear(3 * self.history_len, repr_dim)
        self.wte = nn.Linear(repr_dim, hidden_dim)
        # Robot points, object points, gripper token, and a task language token.
        self.wpe = nn.Embedding(self.num_tracks + 2, hidden_dim)
        self.language_projection = nn.Linear(language_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([PointBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3 * future_horizon),
        )
        self.language_encoder = (FrozenBgeClsEncoder(language_model_path)
                                 if language_model_path is not None else None)
        # Point-Policy applies orthogonal initialization to all actor Linear layers.
        self.apply(self._init_weights)
        nn.init.normal_(self.wpe.weight, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _predict(self, batch):
        tracks = batch["point_tracks"]
        mask = batch["point_mask"].bool()
        expected = (self.history_len, self.num_tracks, 3)
        if tracks.shape[1:] != expected or mask.shape != tracks.shape[:-1]:
            raise ValueError(f"Expected point_tracks [B,{expected}] and matching point_mask")
        tracks = torch.where(mask[..., None], tracks, torch.zeros_like(tracks))
        tracks = tracks.transpose(1, 2).flatten(2)
        gripper = batch["gripper_history"].transpose(1, 2).repeat_interleave(3, dim=-1)
        x = self.wte(self.point_projector(torch.cat((tracks, gripper), dim=1)))
        if "language_embedding" in batch:
            language = batch["language_embedding"].to(device=x.device, dtype=x.dtype)
        elif self.language_encoder is not None:
            language = self.language_encoder.encode(batch["language"], device=x.device).to(x.dtype)
        else:
            raise ValueError("language_embedding or a language_model_path is required")
        x = torch.cat((x, self.language_projection(language)[:, None]), dim=1)
        token_mask = torch.cat((mask.any(dim=1), torch.ones((x.shape[0], 2), dtype=torch.bool, device=x.device)), dim=1)
        x = self.drop(x + self.wpe.weight[None])
        for block in self.blocks:
            x = checkpoint_module(block, x, token_mask)
        x = self.lm_head(self.ln_f(x))
        # Object tokens condition the prediction, but have no reconstruction loss.
        x = torch.cat((x[:, :self.num_robot_points], x[:, self.num_tracks:self.num_tracks + 1]), dim=1)
        return self.action_head(x).reshape(x.shape[0], self.num_robot_points + 1, self.future_horizon, 3).transpose(1, 2)

    def forward(self, batch):
        prediction = self._predict(batch)
        target = torch.cat((batch["target_points"], batch["target_gripper"].unsqueeze(2).expand(-1, -1, 1, 3)), dim=2)
        if target.shape != prediction.shape:
            raise ValueError("Target point/gripper horizon must match the predicted horizon")
        valid = batch["target_mask"].bool()
        mask = valid[..., None, None]
        error = torch.where(mask, prediction - target, torch.zeros_like(prediction))
        count = (valid.sum() * prediction.shape[2] * 3).clamp_min(1)
        # Fixed-std Gaussian NLL, exactly the deterministic head's training objective.
        nll = error.square() / (2 * self.stddev ** 2) + math.log(self.stddev * math.sqrt(2 * math.pi))
        loss = torch.where(mask, nll, torch.zeros_like(nll)).sum() / count
        return loss, {"loss_nll": loss.detach(), "loss_mse": (error.square().sum() / count).detach()}

    @torch.inference_mode()
    def sample(self, batch):
        prediction = self._predict(batch)
        points = prediction[:, :, :self.num_robot_points]
        return {"point_plan": points, "point_plan_mask": torch.ones_like(points[..., 0], dtype=torch.bool),
                "gripper_plan": (2 * prediction[:, :, -1].mean(-1) - 1).clamp(-1, 1)}
