"""Entity-centric camera-XYZ Flow Matching model for GraphVLA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.schema import validate_actor_point_indices
from src.model.encoder import EntityEncoder
from src.model.flow_matching import make_scheduler, sample_time, training_path
from src.model.temporal import AdaRMSNorm, RotaryFlowBlock


FLOW_MODES = {"joint", "point_then_action", "action_only", "point_only"}


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
        history_state: torch.Tensor,
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

        token = self.input_projection(torch.cat([history_state, state], dim=1))
        condition = self.time_mlp(_sinusoidal_time(time, token.shape[-1]))
        history_positions = torch.arange(
            1 - self.history_steps, 1, device=state.device
        )
        future_positions = torch.arange(
            1, self.horizon + 1, device=state.device, dtype=history_positions.dtype
        )
        token_positions = torch.cat([history_positions, future_positions])
        memory_positions = torch.zeros(
            memory.shape[1], device=memory.device, dtype=token_positions.dtype
        )
        for block in self.blocks:
            token = checkpoint(
                block, token, memory, condition, token_positions, memory_positions,
                self.self_attention_mask,
                use_reentrant=False, preserve_rng_state=False,
            ) if self.gradient_checkpointing and self.training else block(
                token, memory, condition, token_positions, memory_positions,
                self.self_attention_mask,
            )
        hidden, _ = self.norm(token, condition)
        return self.output_projection(hidden[:, self.history_steps:])


class GraphFlowModel(nn.Module):
    """Single joint trajectory flow with the unchanged completion head."""

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
        global_layer_types: tuple[int, ...] | list[int] | None = None,
        flow_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        sample_steps: int = 10,
        flow_mode: str = "point_only",
        complete_pos_weight: float = 1.0,
        contact_pos_weight: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.actor_point_indices = validate_actor_point_indices(actor_point_indices)
        self.actor_num_points = len(self.actor_point_indices)
        self.trajectory_dim = self.actor_num_points * 3 + 1
        if num_points < self.actor_num_points:
            raise ValueError(f"num_points must be at least {self.actor_num_points}, got {num_points}")
        if flow_mode != "point_only":
            raise ValueError(f"Legacy trajectory model only supports flow_mode='point_only', got {flow_mode!r}")
        if complete_pos_weight <= 0:
            raise ValueError(f"complete_pos_weight must be positive, got {complete_pos_weight}")
        if contact_pos_weight <= 0:
            raise ValueError(f"contact_pos_weight must be positive, got {contact_pos_weight}")
        self.num_points = num_points
        self.cls_token_num = cls_token_num
        self.history_horizon = history_horizon
        self.history_steps = history_horizon + 1
        self.future_horizon = future_horizon
        self.sample_steps = sample_steps
        self.flow_mode = flow_mode
        self.complete_pos_weight = float(complete_pos_weight)
        self.contact_pos_weight = float(contact_pos_weight)
        self.weights = dict(weights or {})
        self.encoder = EntityEncoder(
            hidden_dim, self.actor_num_points, encoder_layers, num_heads, mlp_ratio, condition_dim,
            max_history=self.history_steps, cls_token_num=cls_token_num, dropout=dropout,
            global_layer_types=global_layer_types,
        )
        self.flow = JointTrajectoryFlow(
            hidden_dim, self.trajectory_dim, future_horizon, self.history_steps,
            flow_layers, num_heads, mlp_ratio, dropout,
        )
        self.complete_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 * cls_token_num, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim * cls_token_num, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.gradient_checkpointing = enabled
        self.flow.gradient_checkpointing = enabled

    def _encode(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _contact_logits(self, memory: torch.Tensor) -> torch.Tensor:
        patient_cls = memory[:, self.cls_token_num:2 * self.cls_token_num].flatten(1)
        return self.contact_head(patient_cls)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory, relation_local = self._encode(batch)
        target = batch["target"]
        trajectory = target["trajectory"]
        if trajectory.shape[1:] != (self.future_horizon, self.trajectory_dim):
            raise ValueError(
                f"Expected target trajectory [B,{self.future_horizon},{self.trajectory_dim}], got {trajectory.shape}"
            )
        time = sample_time(trajectory.shape[0], trajectory.device)
        state, target_velocity, _ = training_path(trajectory, time)
        velocity = self.flow(state, time, memory, self._actor_history(batch))
        loss_flow = F.mse_loss(velocity, target_velocity)

        complete_logits = self.complete_head(relation_local.flatten(1))
        loss_complete = F.binary_cross_entropy_with_logits(
            complete_logits,
            target["is_complete"].to(dtype=complete_logits.dtype),
            pos_weight=complete_logits.new_tensor([self.complete_pos_weight]),
        )
        contact_logits = self._contact_logits(memory)
        loss_contact = F.binary_cross_entropy_with_logits(
            contact_logits,
            target["is_contact"].to(dtype=contact_logits.dtype),
            pos_weight=contact_logits.new_tensor([self.contact_pos_weight]),
        )
        losses = {
            "loss_flow": loss_flow,
            "loss_complete": loss_complete,
            "loss_contact": loss_contact,
        }
        total = sum(value * float(self.weights.get(name, 1.0)) for name, value in losses.items())
        return total, {"loss": total.detach(), **{name: value.detach() for name, value in losses.items()}}

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        num_steps: int | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, relation_local = self._encode(batch)
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
            velocity = self.flow(state, time, memory, self._actor_history(batch))
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
            "is_complete": torch.sigmoid(self.complete_head(relation_local.flatten(1))),
            "is_contact": torch.sigmoid(self._contact_logits(memory)),
        }
