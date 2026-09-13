# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""LIBERO adaptation of Point Bridge's PointNet, BAKU GPT and deterministic decoder.

Adapted from point_bridge/agent/pb.py and agent/networks/{dp3_encoder,policy_head}.py.
The PointNet originates in YanjieZe/3D-Diffusion-Policy (see LICENSE.dp3).
"""

import math
from pathlib import Path

import torch
from torch import nn

from src.policy.checkpointing import GradientCheckpointingMixin, checkpoint_module
from src.policy.point_bridge.gpt import GPT, GPTConfig
from src.policy.point_bridge.language import FrozenMiniLMEncoder


class PointNet(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        layers = []
        for source, target in ((3, 64), (64, 128), (128, 256)):
            layers.extend((nn.Linear(source, target), nn.LayerNorm(target), nn.ReLU()))
        self.mlp = nn.Sequential(*layers)
        self.final_projection = nn.Linear(256, repr_dim)

    def forward(self, points, mask):
        points = torch.where(mask[..., None], points, 0.0)
        features = self.mlp(points).masked_fill(~mask[..., None], -torch.inf)
        pooled = features.max(dim=-2).values
        present = mask.any(dim=-1, keepdim=True)
        pooled = torch.where(present, pooled, 0.0)
        return torch.where(present, self.final_projection(pooled), 0.0)


class DeterministicHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, horizon, dropout):
        super().__init__()
        self.horizon = horizon
        self.pos_embed = nn.Embedding(100, hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead=4, dim_feedforward=2048,
                                       dropout=dropout, batch_first=True), num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, context):
        queries = self.pos_embed.weight[:self.horizon][None].expand(context.shape[0], -1, -1)
        mask = torch.ones(self.horizon, self.horizon, dtype=torch.bool, device=context.device).triu(1)
        for layer in self.transformer_decoder.layers:
            queries = checkpoint_module(layer, queries, context, tgt_mask=mask)
        return self.output_proj(queries)


class PointBridge(GradientCheckpointingMixin, nn.Module):
    def __init__(self, *, action_mode="pose", num_points=32, max_objects=10,
                 actor_point_indices=(0, 1, 2, 3, 4, 5), history_horizon=0,
                 future_horizon=10, repr_dim=512, hidden_dim=256, num_layers=8,
                 num_heads=4, dropout=0.1, stddev=0.1, language_dim=384,
                 language_model_path: str | Path | None = None):
        super().__init__()
        if action_mode not in ("pose", "points"):
            raise ValueError("action_mode must be 'pose' or 'points'")
        if min(num_points, max_objects, num_layers, num_heads) < 1 or not 1 <= future_horizon <= 100:
            raise ValueError("Invalid point counts, transformer size or action horizon (1..100)")
        if history_horizon < 0 or hidden_dim % num_heads or hidden_dim % 4 or stddev <= 0:
            raise ValueError("Invalid history, attention dimensions or stddev")
        self.action_mode = action_mode
        self.num_robot_points = len(actor_point_indices)
        self.num_points, self.max_objects = num_points, max_objects
        self.history_len = history_horizon + 1
        self.future_horizon = future_horizon
        self.stddev = stddev
        self.encoder = PointNet(repr_dim)
        self.language_projector = nn.Sequential(nn.Linear(language_dim, repr_dim), nn.ReLU(),
                                                nn.Linear(repr_dim, repr_dim))
        self.action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))
        self.gpt = GPT(GPTConfig(block_size=max(65, 1 + 3 * self.history_len), input_dim=repr_dim,
                                 output_dim=hidden_dim, n_layer=num_layers, n_head=num_heads,
                                 n_embd=hidden_dim, dropout=dropout, causal=True))
        action_dim = 10 if action_mode == "pose" else 3 * (self.num_robot_points + 1)
        self.action_head = DeterministicHead(hidden_dim, action_dim, future_horizon, dropout)
        self.language_encoder = (FrozenMiniLMEncoder(language_model_path)
                                 if language_model_path is not None else None)
        # Official BCAgent/Actor reinitialize Linear weights orthogonally.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _predict(self, batch):
        robot, objects = batch["robot_points"], batch["object_points"]
        mask = batch["object_mask"].bool()
        if robot.shape[1:] != (self.history_len, self.num_robot_points, 3):
            raise ValueError("Unexpected robot point/history shape")
        if objects.shape[1:] != (self.history_len, self.max_objects, self.num_points, 3):
            raise ValueError("Unexpected object point/history shape")
        if mask.shape != objects.shape[:-1] or not torch.isfinite(robot).all():
            raise ValueError("Invalid object mask or robot points")
        robot_features = checkpoint_module(self.encoder, robot, torch.ones_like(robot[..., 0], dtype=torch.bool))
        object_features = checkpoint_module(self.encoder, objects.flatten(2, 3), mask.flatten(2, 3))
        tokens = torch.stack((robot_features, object_features), dim=2)
        action_tokens = self.action_token.expand(robot.shape[0], self.history_len, -1, -1)
        tokens = torch.cat((tokens, action_tokens), dim=2).flatten(1, 2)
        if "language_embedding" in batch:
            language = batch["language_embedding"].to(device=tokens.device, dtype=tokens.dtype)
        elif self.language_encoder is not None:
            language = self.language_encoder.encode(batch["language"], device=tokens.device).to(tokens.dtype)
        else:
            raise ValueError("language_embedding or a language_model_path is required")
        tokens = torch.cat((self.language_projector(language)[:, None], tokens), dim=1)
        # Last observation's action token conditions the future chunk.
        return self.action_head(self.gpt(tokens)[:, -1:])

    def forward(self, batch):
        prediction = self._predict(batch)
        target = batch["target_actions"]
        valid = batch["target_mask"].bool()
        if target.shape != prediction.shape or valid.shape != prediction.shape[:2]:
            raise ValueError("Action target/mask shape does not match predictions")
        error = torch.where(valid[..., None], prediction - target, 0.0)
        count = (valid.sum() * prediction.shape[-1]).clamp_min(1)
        # Official deterministic decoder uses fixed-std Gaussian NLL (scaled MSE + constant).
        nll = error.square() / (2 * self.stddev ** 2) + math.log(self.stddev * math.sqrt(2 * math.pi))
        loss = torch.where(valid[..., None], nll, 0.0).sum() / count
        return loss, {"loss_nll": loss.detach(), "loss_mse": (error.square().sum() / count).detach()}

    @torch.inference_mode()
    def sample(self, batch):
        prediction = self._predict(batch)
        if self.action_mode == "pose":
            return {"pose_plan": prediction}
        prediction = prediction.reshape(*prediction.shape[:2], self.num_robot_points + 1, 3)
        points = prediction[:, :, :-1]
        return {"point_plan": points, "point_plan_mask": torch.ones_like(points[..., 0], dtype=torch.bool),
                "gripper_plan": (2 * prediction[:, :, -1].mean(-1) - 1).clamp(-1, 1)}
