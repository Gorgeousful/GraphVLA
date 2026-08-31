"""Entity-centric shape/center Flow Matching model for GraphVLA."""

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


SEMANTIC_INJECTION_MODES = {"encoder_only", "flow_adarms_only"}


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
        condition_dim: int,
        semantic_injection_mode: str,
    ) -> None:
        super().__init__()
        if semantic_injection_mode not in SEMANTIC_INJECTION_MODES:
            raise ValueError(
                f"semantic_injection_mode must be one of {sorted(SEMANTIC_INJECTION_MODES)}, "
                f"got {semantic_injection_mode!r}"
            )
        self.horizon = horizon
        self.history_steps = history_steps
        self.trajectory_dim = trajectory_dim
        self.condition_dim = condition_dim
        self.semantic_injection_mode = semantic_injection_mode
        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        semantic_injection = semantic_injection_mode == "flow_adarms_only"
        self.action_projection = (
            nn.Linear(condition_dim, hidden_dim) if semantic_injection else None
        )
        self.degree_projection = (
            nn.Linear(condition_dim, hidden_dim) if semantic_injection else None
        )
        self.null_degree_condition = (
            nn.Parameter(torch.zeros(1, hidden_dim)) if semantic_injection else None
        )
        self.blocks = nn.ModuleList([
            RotaryFlowBlock(
                hidden_dim, heads, mlp_ratio, dropout,
                semantic_injection=semantic_injection,
            )
            for _ in range(layers)
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
        shape_memory: torch.Tensor,
        center_memory: torch.Tensor,
        history_state: torch.Tensor,
        shape_positions: torch.Tensor,
        center_positions: torch.Tensor,
        scene_condition: torch.Tensor | None = None,
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
        if shape_positions.shape != (shape_memory.shape[1],):
            raise ValueError(
                f"Expected shape positions [{shape_memory.shape[1]}], got {shape_positions.shape}"
            )
        if center_positions.shape != (center_memory.shape[1],):
            raise ValueError(
                f"Expected center positions [{center_memory.shape[1]}], got {center_positions.shape}"
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
        shape_positions = shape_positions.to(
            device=shape_memory.device, dtype=token_positions.dtype,
        )
        center_positions = center_positions.to(
            device=center_memory.device, dtype=token_positions.dtype,
        )
        action_condition, degree_condition = self._semantic_conditions(
            scene_condition, batch_size=state.shape[0],
        )
        for block in self.blocks:
            token = checkpoint(
                block, token, center_memory, shape_memory, condition,
                token_positions, center_positions, shape_positions,
                self.self_attention_mask, action_condition, degree_condition,
                use_reentrant=False, preserve_rng_state=True,
            ) if self.gradient_checkpointing and self.training else block(
                token, center_memory, shape_memory, condition,
                token_positions, center_positions, shape_positions,
                self.self_attention_mask, action_condition, degree_condition,
            )
        hidden, _ = self.norm(token, condition)
        return self.output_projection(hidden[:, self.history_steps:])

    def _semantic_conditions(
        self,
        scene_condition: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.semantic_injection_mode == "encoder_only":
            return None, None
        if scene_condition is None:
            raise ValueError("flow_adarms_only semantic injection requires scene_condition")
        if scene_condition.shape != (batch_size, 2, self.condition_dim):
            raise ValueError(
                "Expected scene_condition "
                f"[{batch_size},2,{self.condition_dim}], got "
                f"{tuple(scene_condition.shape)}"
            )
        assert self.action_projection is not None
        assert self.degree_projection is not None
        assert self.null_degree_condition is not None
        action_condition = self.action_projection(scene_condition[:, 0])
        projected_degree = self.degree_projection(scene_condition[:, 1])
        has_degree = scene_condition[:, 1].abs().sum(dim=-1, keepdim=True) > 0
        degree_condition = torch.where(
            has_degree,
            projected_degree,
            self.null_degree_condition.expand(batch_size, -1),
        )
        return action_condition, degree_condition


class GraphFlowModel(nn.Module):
    """Joint trajectory flow over separate entity shape and center memories."""

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
        semantic_injection_mode: str = "encoder_only",
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
        self.semantic_injection_mode = semantic_injection_mode
        self.encoder_output_type = encoder_output_type
        self.gripper_flow_weight = float(gripper_flow_weight)
        self.weights = dict(weights or {})
        self.encoder = EntityEncoder(
            hidden_dim, self.actor_num_points, encoder_layers, num_heads, mlp_ratio, condition_dim,
            max_history=self.history_steps, cls_token_num=cls_token_num, dropout=dropout,
            global_layer_types=global_layer_types,
            node_attention_mode=node_attention_mode,
            encoder_output_type=encoder_output_type,
            include_scene_condition=semantic_injection_mode == "encoder_only",
        )
        self.flow = JointTrajectoryFlow(
            hidden_dim, self.trajectory_dim, future_horizon, self.history_steps,
            flow_layers, num_heads, mlp_ratio, dropout,
            condition_dim, semantic_injection_mode,
        )
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim * (2 * cls_token_num + 4), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.progress_action_projection = nn.Linear(condition_dim, hidden_dim)
        self.progress_degree_projection = nn.Linear(condition_dim, hidden_dim)
        self.progress_null_degree = nn.Parameter(torch.zeros(1, hidden_dim))

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.gradient_checkpointing = enabled
        self.flow.gradient_checkpointing = enabled

    def _encode(
        self, batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = batch["entity_points"]
        if points.shape[1] != self.history_steps or points.shape[3] != self.num_points:
            raise ValueError(
                f"Expected entity points [B,{self.history_steps},3,{self.num_points},3], got {points.shape}"
            )
        return self.encoder(
            points,
            batch["entity_point_mask"],
            (
                batch["scene_condition"]
                if self.semantic_injection_mode == "encoder_only"
                else None
            ),
        )

    def _actor_history(self, batch: dict[str, Any]) -> torch.Tensor:
        actor_xyz = batch["entity_points"][:, :, 0, :self.actor_num_points].flatten(2)
        closedness = batch["gripper_closedness_history"]
        if closedness.shape != (*actor_xyz.shape[:2], 1):
            raise ValueError(
                f"Expected gripper closedness [B,{self.history_steps},1], got {closedness.shape}"
            )
        return torch.cat([actor_xyz, closedness.to(actor_xyz.dtype)], dim=-1)

    def _shape_memory_positions(self, memory: torch.Tensor) -> torch.Tensor:
        if self.encoder_output_type == "current":
            return torch.zeros(memory.shape[1], device=memory.device, dtype=torch.long)
        history_positions = torch.arange(
            1 - self.history_steps, 1, device=memory.device,
        ).repeat_interleave(NUM_ENTITIES * self.cls_token_num)
        return torch.cat([
            history_positions,
            history_positions.new_zeros(self.encoder.scene_token_count),
        ])

    def _center_memory_positions(self, memory: torch.Tensor) -> torch.Tensor:
        if self.encoder_output_type == "current":
            return torch.zeros(memory.shape[1], device=memory.device, dtype=torch.long)
        return torch.arange(
            1 - self.history_steps, 1, device=memory.device,
        ).repeat_interleave(NUM_ENTITIES)

    def _progress(
        self,
        relation_shape: torch.Tensor,
        relation_center: torch.Tensor,
        scene_condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = relation_shape.shape[0]
        if scene_condition.shape != (batch_size, 2, self.flow.condition_dim):
            raise ValueError(
                "Expected progress scene_condition "
                f"[{batch_size},2,{self.flow.condition_dim}], got "
                f"{tuple(scene_condition.shape)}"
            )
        action = self.progress_action_projection(scene_condition[:, 0])
        projected_degree = self.progress_degree_projection(scene_condition[:, 1])
        has_degree = scene_condition[:, 1].abs().sum(dim=-1, keepdim=True) > 0
        degree = torch.where(
            has_degree,
            projected_degree,
            self.progress_null_degree.expand(batch_size, -1),
        )
        features = torch.cat([
            relation_shape.flatten(1),
            relation_center.flatten(1),
            action,
            degree,
        ], dim=-1)
        return torch.sigmoid(self.progress_head(features))

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shape_memory, center_memory, relation_shape, relation_center = self._encode(batch)
        target = batch["target"]
        trajectory = target["trajectory"]
        if trajectory.shape[1:] != (self.future_horizon, self.trajectory_dim):
            raise ValueError(
                f"Expected target trajectory [B,{self.future_horizon},{self.trajectory_dim}], got {trajectory.shape}"
            )
        time = sample_time(trajectory.shape[0], trajectory.device)
        state, target_velocity, _ = training_path(trajectory, time)
        velocity = self.flow(
            state, time, shape_memory, center_memory, self._actor_history(batch),
            self._shape_memory_positions(shape_memory),
            self._center_memory_positions(center_memory), batch["scene_condition"],
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

        progress = self._progress(
            relation_shape, relation_center, batch["scene_condition"],
        )
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
        shape_memory, center_memory, relation_shape, relation_center = self._encode(batch)
        batch_size = batch["entity_points"].shape[0]
        state = noise
        if state is None:
            state = torch.randn(
                batch_size, self.future_horizon, self.trajectory_dim,
                device=shape_memory.device, dtype=shape_memory.dtype,
            )
        elif state.shape != (batch_size, self.future_horizon, self.trajectory_dim):
            raise ValueError(
                f"Expected noise [B,{self.future_horizon},{self.trajectory_dim}], got {state.shape}"
            )

        scheduler = make_scheduler(num_steps or self.sample_steps, shape_memory.device)
        for timestep in scheduler.timesteps:
            time = (timestep / scheduler.config.num_train_timesteps).expand(batch_size).to(
                shape_memory.dtype
            )
            velocity = self.flow(
                state, time, shape_memory, center_memory, self._actor_history(batch),
                self._shape_memory_positions(shape_memory),
                self._center_memory_positions(center_memory), batch["scene_condition"],
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
            "subtask_progress": self._progress(
                relation_shape, relation_center, batch["scene_condition"],
            ),
        }
