"""Entity-centric coupled Flow Matching model for GraphVLA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTOR_NUM_POINTS
from src.model.encoder import EntityEncoder
from src.model.flow_matching import make_scheduler, sample_time, training_path
from src.model.temporal import AdaRMSNorm, RotaryFlowBlock


def _sinusoidal_time(time: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError(f"Flow time embedding dimension must be even, got {dim}")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=time.device, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angles = time.float()[:, None] / period[None] * (2.0 * torch.pi)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).to(time.dtype)


class RelativeTrajectoryFlow(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int, layers: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.input_projection = nn.Linear(ACTOR_NUM_POINTS * 3, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            RotaryFlowBlock(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.norm = AdaRMSNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, ACTOR_NUM_POINTS * 3)
        self.gradient_checkpointing = False

    def forward(self, state: torch.Tensor, time: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        token = self.input_projection(state.reshape(batch, self.horizon, ACTOR_NUM_POINTS * 3))
        condition = self.time_mlp(_sinusoidal_time(time, token.shape[-1]))
        token_positions = torch.arange(1, self.horizon + 1, device=state.device)
        memory_positions = torch.zeros(memory.shape[1], device=memory.device, dtype=token_positions.dtype)
        for block in self.blocks:
            token = checkpoint(
                block, token, memory, condition, token_positions, memory_positions,
                use_reentrant=False, preserve_rng_state=False,
            ) if self.gradient_checkpointing and self.training else block(
                token, memory, condition, token_positions, memory_positions,
            )
        hidden, _ = self.norm(token, condition)
        velocity = self.output_projection(hidden).view(batch, self.horizon, ACTOR_NUM_POINTS, 3)
        return velocity, hidden


class RobotPrivateFlow(nn.Module):
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
        self.input_projection = nn.Linear(ACTOR_NUM_POINTS + 1, hidden_dim)
        self.history_projection = nn.Linear(ACTOR_NUM_POINTS + 1, hidden_dim)
        self.history_steps = history_steps
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            RotaryFlowBlock(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.norm = AdaRMSNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, ACTOR_NUM_POINTS + 1)
        self.gradient_checkpointing = False

    def forward(
        self,
        private_state: torch.Tensor,
        time: torch.Tensor,
        memory: torch.Tensor,
        relative_hidden: torch.Tensor,
        metric_history: torch.Tensor,
        closedness_history: torch.Tensor,
    ) -> torch.Tensor:
        token = self.input_projection(private_state)
        condition = self.time_mlp(_sinusoidal_time(time, token.shape[-1]))
        token_positions = torch.arange(1, self.horizon + 1, device=private_state.device)
        history_steps = metric_history.shape[1]
        if history_steps > self.history_steps:
            raise ValueError(f"Expected at most {self.history_steps} history steps, got {history_steps}")
        history_state = torch.cat([metric_history.squeeze(-1), closedness_history], dim=-1)
        history_token = self.history_projection(history_state)
        private_memory = torch.cat([
            memory,
            relative_hidden.detach(),
            history_token,
        ], dim=1)
        memory_positions = torch.cat([
            torch.zeros(memory.shape[1], device=memory.device, dtype=token_positions.dtype),
            token_positions,
            torch.arange(1 - history_steps, 1, device=memory.device, dtype=token_positions.dtype),
        ])
        for block in self.blocks:
            token = checkpoint(
                block, token, private_memory, condition, token_positions, memory_positions,
                use_reentrant=False, preserve_rng_state=False,
            ) if self.gradient_checkpointing and self.training else block(
                token, private_memory, condition, token_positions, memory_positions,
            )
        hidden, _ = self.norm(token, condition)
        return self.output_projection(hidden)


class GraphFlowModel(nn.Module):
    """Joint relative/robot-private trajectory model with a parallel completion head."""

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
        self.private_flow = RobotPrivateFlow(
            hidden_dim, future_horizon, history_horizon + 1,
            flow_layers, num_heads, mlp_ratio, dropout,
        )
        self.complete_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.gradient_checkpointing = enabled
        self.relative_flow.gradient_checkpointing = enabled
        self.private_flow.gradient_checkpointing = enabled

    def _encode(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if batch["entity_points"].shape[3] != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} points per entity, got {batch['entity_points'].shape[3]}"
            )
        return self.encoder(
            batch["entity_points"],
            batch["entity_point_mask"],
            batch["scene_condition"],
            batch["entity_role_condition"],
        )

    def _velocities(
        self,
        batch: dict[str, Any],
        memory: torch.Tensor,
        relative_state: torch.Tensor,
        private_state: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_velocity, relative_hidden = self.relative_flow(relative_state, time, memory)
        private_velocity = self.private_flow(
            private_state,
            time,
            memory,
            relative_hidden,
            batch["actor_metric_history"],
            batch["gripper_closedness_history"],
        )
        return relative_velocity, private_velocity

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
            required = ("actor_metric_history", "gripper_closedness_history")
            missing = [name for name in required if name not in batch]
            missing += [name for name in ("metric_z_plan", "gripper_action_plan") if name not in target]
            if missing:
                raise KeyError(f"Robot samples require private fields: {missing}")
            private_target = torch.cat([
                target["metric_z_plan"].squeeze(-1), target["gripper_action_plan"],
            ], dim=-1)
            private_state, private_target_velocity, _ = training_path(private_target, time)
            private_velocity = self.private_flow(
                private_state, time, memory, relative_hidden,
                batch["actor_metric_history"], batch["gripper_closedness_history"],
            )
            private_velocity = private_velocity[robot_mask]
            private_target_velocity = private_target_velocity[robot_mask]
            loss_metric = F.mse_loss(
                private_velocity[..., :ACTOR_NUM_POINTS],
                private_target_velocity[..., :ACTOR_NUM_POINTS],
            )
            loss_action = F.mse_loss(
                private_velocity[..., ACTOR_NUM_POINTS:],
                private_target_velocity[..., ACTOR_NUM_POINTS:],
            )
            loss_private = loss_metric + loss_action
        else:
            # Keep robot-private parameters in the DDP autograd graph even when a
            # rank receives an all-human batch.
            loss_private = sum(parameter.sum() for parameter in self.private_flow.parameters()) * 0.0
            loss_metric = zero
            loss_action = zero
        loss_complete = F.binary_cross_entropy_with_logits(
            complete_logits, target["is_complete"].to(dtype=complete_logits.dtype)
        )
        optimization_losses = {
            "loss_relative": loss_relative,
            "loss_private": loss_private,
            "loss_complete": loss_complete,
        }
        total = sum(
            value * float(self.weights.get(name, 1.0))
            for name, value in optimization_losses.items()
        )
        metrics = {
            **optimization_losses,
            "loss_metric_z": loss_metric,
            "loss_gripper_action": loss_action,
        }
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
        private_size = self.future_horizon * (ACTOR_NUM_POINTS + 1)
        flat = noise
        if flat is None:
            flat = torch.randn(
                batch_size, rel_size + private_size,
                device=memory.device, dtype=memory.dtype,
            )
        scheduler = make_scheduler(num_steps or self.sample_steps, memory.device)
        for timestep in scheduler.timesteps:
            relative_state = flat[:, :rel_size].view(batch_size, self.future_horizon, ACTOR_NUM_POINTS, 3)
            private_state = flat[:, rel_size:].view(
                batch_size, self.future_horizon, ACTOR_NUM_POINTS + 1
            )
            time = (timestep / scheduler.config.num_train_timesteps).expand(batch_size).to(memory.dtype)
            velocities = self._velocities(batch, memory, relative_state, private_state, time)
            flat_velocity = torch.cat([value.reshape(batch_size, -1) for value in velocities], dim=1)
            flat = scheduler.step(flat_velocity, timestep, flat).prev_sample

        relative_delta = flat[:, :rel_size].view(batch_size, self.future_horizon, ACTOR_NUM_POINTS, 3)
        private_plan = flat[:, rel_size:].view(
            batch_size, self.future_horizon, ACTOR_NUM_POINTS + 1
        )
        metric_delta = private_plan[..., :ACTOR_NUM_POINTS].unsqueeze(-1)
        gripper_action = private_plan[..., ACTOR_NUM_POINTS:]
        current_relative = batch["entity_points"][:, -1, 0, :ACTOR_NUM_POINTS]
        current_metric = batch["actor_metric_history"][:, -1]
        return {
            "relative_plan": current_relative[:, None] + relative_delta,
            "metric_z_plan": current_metric[:, None] + metric_delta,
            "gripper_action_plan": gripper_action,
            "is_complete": torch.sigmoid(self.complete_head(relation_local.flatten(1))),
        }
