"""Entity-centric coupled Flow Matching model for GraphVLA."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTOR_NUM_POINTS
from src.model.encoder import EntityEncoder
from src.model.flow_matching import make_scheduler, sample_time, training_path


def _sinusoidal_time(time: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * -(math.log(10000.0) / max(half - 1, 1))
    )
    embedding = time[:, None] * frequency[None]
    embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


def _decoder_layer(hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> nn.Module:
    return nn.TransformerDecoderLayer(
        hidden_dim,
        num_heads,
        int(hidden_dim * mlp_ratio),
        dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class RelativeTrajectoryFlow(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int, layers: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.input_projection = nn.Linear(3, hidden_dim)
        self.horizon_embedding = nn.Embedding(horizon, hidden_dim)
        self.keypoint_embedding = nn.Embedding(ACTOR_NUM_POINTS, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([
            _decoder_layer(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 3)
        self.gradient_checkpointing = False

    def forward(self, state: torch.Tensor, time: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        token = self.input_projection(state)
        token = token + self.horizon_embedding(torch.arange(self.horizon, device=state.device))[None, :, None]
        token = token + self.keypoint_embedding(torch.arange(ACTOR_NUM_POINTS, device=state.device))[None, None]
        token = token.reshape(batch, self.horizon * ACTOR_NUM_POINTS, -1)
        token = token + self.time_mlp(_sinusoidal_time(time, token.shape[-1]))[:, None]
        for block in self.blocks:
            token = checkpoint(
                block, token, memory, use_reentrant=False, preserve_rng_state=False,
            ) if self.gradient_checkpointing and self.training else block(token, memory)
        hidden = self.norm(token)
        velocity = self.output_projection(hidden).view(batch, self.horizon, ACTOR_NUM_POINTS, 3)
        return velocity, hidden


class RobotMetricFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        history_steps: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_projection = nn.Linear(1, hidden_dim)
        self.metric_history_projection = nn.Linear(3, hidden_dim)
        self.width_history_projection = nn.Linear(1, hidden_dim)
        self.horizon_embedding = nn.Embedding(horizon, hidden_dim)
        self.value_type_embedding = nn.Embedding(ACTOR_NUM_POINTS + 1, hidden_dim)
        self.history_time_embedding = nn.Embedding(history_steps, hidden_dim)
        self.metric_keypoint_embedding = nn.Embedding(ACTOR_NUM_POINTS, hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([
            _decoder_layer(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1)
        self.gradient_checkpointing = False

    def forward(
        self,
        metric_z: torch.Tensor,
        width: torch.Tensor,
        time: torch.Tensor,
        memory: torch.Tensor,
        relative_hidden: torch.Tensor,
        metric_history: torch.Tensor,
        width_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = metric_z.shape[0]
        state = torch.cat([metric_z, width.unsqueeze(2)], dim=2)
        token = self.input_projection(state)
        token = token + self.horizon_embedding(torch.arange(self.horizon, device=state.device))[None, :, None]
        token = token + self.value_type_embedding(
            torch.arange(ACTOR_NUM_POINTS + 1, device=state.device)
        )[None, None]
        token = token.reshape(batch, self.horizon * (ACTOR_NUM_POINTS + 1), -1)
        token = token + self.time_mlp(_sinusoidal_time(time, token.shape[-1]))[:, None]
        history_steps = metric_history.shape[1]
        history_time = self.history_time_embedding(
            torch.arange(history_steps, device=metric_history.device)
        )
        metric_history_token = self.metric_history_projection(metric_history)
        metric_history_token = metric_history_token + history_time[None, :, None]
        metric_history_token = metric_history_token + self.metric_keypoint_embedding(
            torch.arange(ACTOR_NUM_POINTS, device=metric_history.device)
        )[None, None]
        width_history_token = self.width_history_projection(width_history) + history_time[None]
        private_memory = torch.cat([
            memory,
            relative_hidden.detach(),
            metric_history_token.flatten(1, 2),
            width_history_token,
        ], dim=1)
        for block in self.blocks:
            token = checkpoint(
                block, token, private_memory, use_reentrant=False, preserve_rng_state=False,
            ) if self.gradient_checkpointing and self.training else block(token, private_memory)
        velocity = self.output_projection(self.norm(token)).view(
            batch, self.horizon, ACTOR_NUM_POINTS + 1, 1
        )
        return velocity[:, :, :ACTOR_NUM_POINTS], velocity[:, :, ACTOR_NUM_POINTS]


class GraphFlowModel(nn.Module):
    """Joint relative/robot-metric trajectory model with an actor-free complete head."""

    def __init__(
        self,
        num_points: int = 32,
        history_horizon: int = 19,
        future_horizon: int = 10,
        condition_dim: int = 1152,
        hidden_dim: int = 512,
        encoder_layers: int = 8,
        flow_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        sample_steps: int = 10,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if num_points < ACTOR_NUM_POINTS:
            raise ValueError(f"num_points must be at least {ACTOR_NUM_POINTS}, got {num_points}")
        self.num_points = num_points
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.sample_steps = sample_steps
        self.weights = dict(weights or {})
        self.encoder = EntityEncoder(
            hidden_dim, encoder_layers, num_heads, mlp_ratio, condition_dim,
            max_history=history_horizon + 1, dropout=dropout,
        )
        self.relative_flow = RelativeTrajectoryFlow(
            hidden_dim, future_horizon, flow_layers, num_heads, mlp_ratio, dropout
        )
        self.metric_flow = RobotMetricFlow(
            hidden_dim, future_horizon, history_horizon + 1,
            flow_layers, num_heads, mlp_ratio, dropout,
        )
        self.complete_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.gradient_checkpointing = enabled
        self.relative_flow.gradient_checkpointing = enabled
        self.metric_flow.gradient_checkpointing = enabled

    def _encode(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if batch["entity_points"].shape[3] != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} points per entity, got {batch['entity_points'].shape[3]}"
            )
        return self.encoder(
            batch["entity_points"], batch["entity_point_mask"], batch["entity_condition"]
        )

    def _velocities(
        self,
        batch: dict[str, Any],
        memory: torch.Tensor,
        relative_state: torch.Tensor,
        metric_state: torch.Tensor,
        width_state: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relative_velocity, relative_hidden = self.relative_flow(relative_state, time, memory)
        metric_velocity, width_velocity = self.metric_flow(
            metric_state,
            width_state,
            time,
            memory,
            relative_hidden,
            batch["actor_metric_history"],
            batch["gripper_width_history"],
        )
        return relative_velocity, metric_velocity, width_velocity

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = batch["target"]
        memory, relation_local = self._encode(batch)
        complete_logits = self.complete_head(relation_local.flatten(1))
        time = sample_time(batch["entity_points"].shape[0], batch["entity_points"].device)
        relative_state, relative_target_velocity, _ = training_path(target["relative_plan"], time)
        relative_velocity, relative_hidden = self.relative_flow(relative_state, time, memory)
        loss_relative = F.mse_loss(relative_velocity, relative_target_velocity)
        robot_mask_value = batch.get("robot_metric_mask")
        robot_mask = (
            robot_mask_value.reshape(-1).bool()
            if robot_mask_value is not None
            else torch.zeros(relative_state.shape[0], dtype=torch.bool, device=relative_state.device)
        )
        zero = loss_relative.new_zeros(())
        if robot_mask.any():
            required = ("actor_metric_history", "gripper_width_history")
            missing = [name for name in required if name not in batch]
            missing += [name for name in ("metric_z_plan", "gripper_width_plan") if name not in target]
            if missing:
                raise KeyError(f"Robot samples require metric fields: {missing}")
            metric_state, metric_target_velocity, _ = training_path(target["metric_z_plan"], time)
            width_state, width_target_velocity, _ = training_path(target["gripper_width_plan"], time)
            metric_velocity, width_velocity = self.metric_flow(
                metric_state, width_state, time, memory, relative_hidden,
                batch["actor_metric_history"], batch["gripper_width_history"],
            )
            loss_metric = F.mse_loss(metric_velocity[robot_mask], metric_target_velocity[robot_mask])
            loss_width = F.mse_loss(width_velocity[robot_mask], width_target_velocity[robot_mask])
        else:
            # Keep robot-private parameters in the DDP autograd graph even when a
            # rank receives an all-human batch.
            loss_metric = sum(parameter.sum() for parameter in self.metric_flow.parameters()) * 0.0
            loss_width = zero
        loss_complete = F.binary_cross_entropy_with_logits(
            complete_logits, target["is_complete"].to(dtype=complete_logits.dtype)
        )
        metrics = {
            "loss_relative": loss_relative,
            "loss_metric_z": loss_metric,
            "loss_gripper_width": loss_width,
            "loss_complete": loss_complete,
        }
        total = sum(metrics[name] * float(self.weights.get(name, 1.0)) for name in metrics)
        return total, {"loss": total.detach(), **{name: value.detach() for name, value in metrics.items()}}

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, relation_local = self._encode(batch)
        batch_size = batch["entity_points"].shape[0]
        rel_size = self.future_horizon * ACTOR_NUM_POINTS * 3
        metric_size = self.future_horizon * ACTOR_NUM_POINTS
        width_size = self.future_horizon
        flat = noise
        if flat is None:
            flat = torch.randn(
                batch_size, rel_size + metric_size + width_size,
                device=memory.device, dtype=memory.dtype,
            )
        scheduler = make_scheduler(num_steps or self.sample_steps, memory.device)
        for timestep in scheduler.timesteps:
            relative_state = flat[:, :rel_size].view(batch_size, self.future_horizon, ACTOR_NUM_POINTS, 3)
            metric_state = flat[:, rel_size:rel_size + metric_size].view(
                batch_size, self.future_horizon, ACTOR_NUM_POINTS, 1
            )
            width_state = flat[:, -width_size:].view(batch_size, self.future_horizon, 1)
            time = (timestep / scheduler.config.num_train_timesteps).expand(batch_size).to(memory.dtype)
            velocities = self._velocities(
                batch, memory, relative_state, metric_state, width_state, time
            )
            flat_velocity = torch.cat([value.reshape(batch_size, -1) for value in velocities], dim=1)
            flat = scheduler.step(flat_velocity, timestep, flat).prev_sample

        relative_delta = flat[:, :rel_size].view(batch_size, self.future_horizon, ACTOR_NUM_POINTS, 3)
        metric_delta = flat[:, rel_size:rel_size + metric_size].view(
            batch_size, self.future_horizon, ACTOR_NUM_POINTS, 1
        )
        width = flat[:, -width_size:].view(batch_size, self.future_horizon, 1)
        current_relative = batch["entity_points"][:, -1, 0, :ACTOR_NUM_POINTS]
        current_metric = batch["actor_metric_history"][:, -1, :, 2:3]
        return {
            "relative_plan": current_relative[:, None] + relative_delta,
            "metric_z_plan": current_metric[:, None] + metric_delta,
            "gripper_width_plan": width,
            "is_complete": torch.sigmoid(self.complete_head(relation_local.flatten(1))),
        }
