"""Entity-centric camera-XYZ Flow Matching model for GraphVLA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.schema import NUM_ENTITIES, validate_actor_point_indices
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


class JointTrajectoryFlow(nn.Module):
    """Condition future flow queries on the full actor trajectory history."""

    def __init__(
        self,
        hidden_dim: int,
        trajectory_dim: int,
        horizon: int,
        history_steps: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.history_steps = history_steps
        self.trajectory_dim = trajectory_dim
        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            RotaryFlowBlock(hidden_dim, heads, mlp_ratio, dropout) for _ in range(layers)
        ])
        self.norm = AdaRMSNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, trajectory_dim)
        self.gradient_checkpointing = False

        self_attention_mask = torch.ones(
            history_steps + horizon, history_steps + horizon, dtype=torch.bool
        )
        self_attention_mask[:history_steps, history_steps:] = False
        self.register_buffer("self_attention_mask", self_attention_mask, persistent=False)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        memory: torch.Tensor,
        semantic_memory: torch.Tensor,
        history_state: torch.Tensor,
        memory_positions: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.horizon, self.trajectory_dim):
            raise ValueError(
                f"Expected flow state [B,{self.horizon},{self.trajectory_dim}], got {state.shape}"
            )
        if history_state.ndim != 3 or history_state.shape[1:] != (
            self.history_steps, self.trajectory_dim
        ):
            raise ValueError(
                f"Expected history state [B,{self.history_steps},{self.trajectory_dim}], "
                f"got {history_state.shape}"
            )
        if memory_positions.shape != (memory.shape[1],):
            raise ValueError(
                f"Expected memory positions [{memory.shape[1]}], got {memory_positions.shape}"
            )
        if (
            semantic_memory.ndim != 3
            or semantic_memory.shape[:2] != (state.shape[0], 2)
            or semantic_memory.shape[2] != memory.shape[2]
        ):
            raise ValueError(
                f"Expected semantic memory [{state.shape[0]},2,{memory.shape[2]}], "
                f"got {semantic_memory.shape}"
            )

        token = self.input_projection(torch.cat([history_state, state], dim=1))
        condition = self.time_mlp(_sinusoidal_time(time, token.shape[-1]))
        history_positions = torch.arange(
            1 - self.history_steps, 1, device=state.device
        )
        future_positions = torch.arange(
            1, self.horizon + 1, device=state.device, dtype=history_positions.dtype
        )
        token_positions = torch.cat([history_positions, future_positions])
        memory_positions = memory_positions.to(
            device=memory.device, dtype=token_positions.dtype,
        )
        semantic_positions = torch.zeros(
            semantic_memory.shape[1], device=semantic_memory.device,
            dtype=token_positions.dtype,
        )
        for block in self.blocks:
            token = checkpoint(
                block, token, memory, semantic_memory, condition, token_positions,
                memory_positions, semantic_positions, self.self_attention_mask,
                use_reentrant=False, preserve_rng_state=True,
            ) if self.gradient_checkpointing and self.training else block(
                token, memory, semantic_memory, condition, token_positions,
                memory_positions, semantic_positions, self.self_attention_mask,
            )
        hidden, _ = self.norm(token, condition)
        return self.output_projection(hidden[:, self.history_steps:])


class GraphFlowModel(nn.Module):
    """Single joint trajectory flow with a progress head."""

    def __init__(
        self,
        actor_point_indices: tuple[int, ...],
        num_points: int = 32,
        cls_token_num: int = 1,
        history_horizon: int = 9,
        future_horizon: int = 10,
        condition_dim: int = 384,
        hidden_dim: int = 512,
        encoder_layers: int = 8,
        encoder_output_type: str = "current",
        global_layer_types: tuple[int, ...] | list[int] | None = None,
        node_attention_mode: str = "full",
        flow_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        sample_steps: int = 10,
        gripper_flow_weight: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.actor_point_indices = validate_actor_point_indices(actor_point_indices)
        self.actor_num_points = len(self.actor_point_indices)
        self.trajectory_dim = self.actor_num_points * 3 + 1
        if num_points < self.actor_num_points:
            raise ValueError(f"num_points must be at least {self.actor_num_points}, got {num_points}")
        if encoder_output_type not in ("current", "all"):
            raise ValueError(
                "encoder_output_type must be 'current' or 'all', "
                f"got {encoder_output_type!r}"
            )
        if gripper_flow_weight <= 0:
            raise ValueError(f"gripper_flow_weight must be positive, got {gripper_flow_weight}")
        self.num_points = num_points
        self.cls_token_num = cls_token_num
        self.history_horizon = history_horizon
        self.history_steps = history_horizon + 1
        self.future_horizon = future_horizon
        self.sample_steps = sample_steps
        self.encoder_output_type = encoder_output_type
        self.gripper_flow_weight = float(gripper_flow_weight)
        self.weights = dict(weights or {})
        self.encoder = EntityEncoder(
            hidden_dim, self.actor_num_points, encoder_layers, num_heads, mlp_ratio, condition_dim,
            max_history=self.history_steps, cls_token_num=cls_token_num, dropout=dropout,
            global_layer_types=global_layer_types,
            node_attention_mode=node_attention_mode,
            encoder_output_type=encoder_output_type,
        )
        self.flow = JointTrajectoryFlow(
            hidden_dim, self.trajectory_dim, future_horizon, self.history_steps,
            flow_layers, num_heads, mlp_ratio, dropout,
        )
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim * (2 * cls_token_num + 2), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.gradient_checkpointing = enabled
        self.flow.gradient_checkpointing = enabled

    def _encode(
        self, batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = batch["entity_points"]
        if points.shape[1] != self.history_steps or points.shape[3] != self.num_points:
            raise ValueError(
                f"Expected entity points [B,{self.history_steps},3,{self.num_points},3], got {points.shape}"
            )
        return self.encoder(
            points,
            batch["entity_point_mask"],
            batch["scene_condition"],
        )

    def _actor_history(self, batch: dict[str, Any]) -> torch.Tensor:
        actor_xyz = batch["entity_points"][:, :, 0, :self.actor_num_points].flatten(2)
        closedness = batch["gripper_closedness_history"]
        if closedness.shape != (*actor_xyz.shape[:2], 1):
            raise ValueError(
                f"Expected gripper closedness [B,{self.history_steps},1], got {closedness.shape}"
            )
        return torch.cat([actor_xyz, closedness.to(actor_xyz.dtype)], dim=-1)

    def _memory_positions(self, memory: torch.Tensor) -> torch.Tensor:
        if self.encoder_output_type == "current":
            return torch.zeros(memory.shape[1], device=memory.device, dtype=torch.long)
        history_positions = torch.arange(
            1 - self.history_steps, 1, device=memory.device,
        ).repeat_interleave(NUM_ENTITIES * self.cls_token_num)
        return history_positions

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory, relation_local, semantic_memory = self._encode(batch)
        target = batch["target"]
        trajectory = target["trajectory"]
        if trajectory.shape[1:] != (self.future_horizon, self.trajectory_dim):
            raise ValueError(
                f"Expected target trajectory [B,{self.future_horizon},{self.trajectory_dim}], got {trajectory.shape}"
            )
        time = sample_time(trajectory.shape[0], trajectory.device)
        state, target_velocity, _ = training_path(trajectory, time)
        velocity = self.flow(
            state, time, memory, semantic_memory, self._actor_history(batch),
            self._memory_positions(memory),
        )
        squared_error = (velocity - target_velocity).square()
        point_squared_error = squared_error[..., :-1]
        gripper_squared_error = squared_error[..., -1:]
        loss_flow_points = point_squared_error.mean()
        loss_flow_gripper = gripper_squared_error.mean()
        loss_flow = (
            point_squared_error.sum()
            + self.gripper_flow_weight * gripper_squared_error.sum()
        ) / (
            point_squared_error.numel()
            + self.gripper_flow_weight * gripper_squared_error.numel()
        )

        progress_input = torch.cat([relation_local, semantic_memory], dim=1).flatten(1)
        progress = torch.sigmoid(self.progress_head(progress_input))
        loss_progress = F.smooth_l1_loss(
            progress,
            target["subtask_progress"].to(dtype=progress.dtype),
        )
        losses = {
            "loss_flow": loss_flow,
            "loss_progress": loss_progress,
        }
        total = sum(value * float(self.weights.get(name, 1.0)) for name, value in losses.items())
        return total, {
            "loss": total.detach(),
            **{name: value.detach() for name, value in losses.items()},
            "loss_flow_points": loss_flow_points.detach(),
            "loss_flow_gripper": loss_flow_gripper.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, relation_local, semantic_memory = self._encode(batch)
        batch_size = batch["entity_points"].shape[0]
        state = noise
        if state is None:
            state = torch.randn(
                batch_size, self.future_horizon, self.trajectory_dim,
                device=memory.device, dtype=memory.dtype,
            )
        elif state.shape != (batch_size, self.future_horizon, self.trajectory_dim):
            raise ValueError(
                f"Expected noise [B,{self.future_horizon},{self.trajectory_dim}], got {state.shape}"
            )

        scheduler = make_scheduler(num_steps or self.sample_steps, memory.device)
        for timestep in scheduler.timesteps:
            time = (timestep / scheduler.config.num_train_timesteps).expand(batch_size).to(memory.dtype)
            velocity = self.flow(
                state, time, memory, semantic_memory, self._actor_history(batch),
                self._memory_positions(memory),
            )
            state = scheduler.step(velocity, timestep, state).prev_sample

        point_plan = state[..., : self.actor_num_points * 3].view(
            batch_size, self.future_horizon, self.actor_num_points, 3
        )
        return {
            "point_plan": point_plan,
            "point_plan_mask": torch.ones(
                point_plan.shape[:-1], dtype=torch.bool, device=point_plan.device,
            ),
            "gripper_plan": state[..., -1],
            "subtask_progress": torch.sigmoid(self.progress_head(
                torch.cat([relation_local, semantic_memory], dim=1).flatten(1)
            )),
        }
