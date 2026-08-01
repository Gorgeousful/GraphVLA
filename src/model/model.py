"""Joint future-point and action Flow Matching model for GraphVLA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTION_DIM, ACTOR_NUM_POINTS, NUM_ENTITIES, POINT_FEATURE_DIM
from src.model.encoder import EntityEncoder
from src.model.flow_matching import FlowMatchScheduler, make_scheduler
from src.model.temporal import AdaRMSNorm, RotaryAttention, RotaryFlowBlock


FLOW_MODES = {"joint", "point_then_action"}


def _sinusoidal_time(time: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError(f"Flow time embedding dimension must be even, got {dim}")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=time.device, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angles = time.float()[..., None] / period * (2.0 * torch.pi)
    return torch.cat([angles.sin(), angles.cos()], dim=-1).to(time.dtype)


class JointFlowLayer(nn.Module):
    """One paired Point/Action expert layer with concatenated self-attention K/V."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        self_axis_dims: tuple[int, int, int],
        history_axis_dims: tuple[int, int],
    ) -> None:
        super().__init__()
        self.point_block = RotaryFlowBlock(
            hidden_dim, num_heads, mlp_ratio, dropout, self_axis_dims, history_axis_dims,
        )
        self.action_block = RotaryFlowBlock(
            hidden_dim, num_heads, mlp_ratio, dropout, self_axis_dims, history_axis_dims,
        )

    def forward(
        self,
        point_token: torch.Tensor,
        action_token: torch.Tensor,
        point_condition: torch.Tensor,
        action_condition: torch.Tensor,
        point_positions: torch.Tensor,
        action_positions: torch.Tensor,
        point_history_positions: torch.Tensor,
        action_history_positions: torch.Tensor,
        history_memory: torch.Tensor,
        history_positions: torch.Tensor,
        scene_memory: torch.Tensor,
        point_attention_mask: torch.Tensor,
        action_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_q, point_k, point_v, point_gate = self.point_block.prepare_self_attention(
            point_token, point_condition, point_positions,
        )
        action_q, action_k, action_v, action_gate = self.action_block.prepare_self_attention(
            action_token, action_condition, action_positions,
        )
        joint_key = torch.cat([point_k, action_k], dim=2)
        joint_value = torch.cat([point_v, action_v], dim=2)
        point_update = self.point_block.self_attention.attend(
            point_q, joint_key, joint_value, point_attention_mask,
        )
        action_update = self.action_block.self_attention.attend(
            action_q, joint_key, joint_value, action_attention_mask,
        )
        point_token = self.point_block.finish(
            point_token, point_update, point_gate,
            history_memory, scene_memory, point_condition,
            point_history_positions, history_positions,
        )
        action_token = self.action_block.finish(
            action_token, action_update, action_gate,
            history_memory, scene_memory, action_condition,
            action_history_positions, history_positions,
        )
        return point_token, action_token


class PointActionFlow(nn.Module):
    """OpenWAM-style paired experts for point/action joint or staged denoising."""

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        num_points: int,
        include_future_object_point: bool,
        cls_token_num: int,
        condition_dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        head_dim = hidden_dim // heads
        if hidden_dim % heads or head_dim % 6:
            raise ValueError(
                f"Three-axis RoPE requires head_dim divisible by 6, got {hidden_dim}/{heads}"
            )
        self_axis_dims = (head_dim // 3,) * 3
        history_axis_dims = (head_dim // 2, head_dim - head_dim // 2)
        self.horizon = horizon
        self.num_points = num_points
        self.cls_token_num = cls_token_num
        self.include_future_object_point = include_future_object_point
        self.points_per_step = ACTOR_NUM_POINTS + (
            2 * num_points if include_future_object_point else 0
        )
        role_ids = torch.zeros(self.points_per_step, dtype=torch.long)
        slot_ids = torch.zeros(self.points_per_step, dtype=torch.float32)
        slot_ids[:ACTOR_NUM_POINTS] = torch.arange(ACTOR_NUM_POINTS, dtype=torch.float32)
        if include_future_object_point:
            role_ids[ACTOR_NUM_POINTS:ACTOR_NUM_POINTS + num_points] = 1
            role_ids[ACTOR_NUM_POINTS + num_points:] = 2
        self.register_buffer("point_role_ids", role_ids, persistent=False)
        self.register_buffer("point_slot_ids", slot_ids, persistent=False)
        self.point_projection = nn.Linear(POINT_FEATURE_DIM, hidden_dim)
        self.action_projection = nn.Linear(ACTION_DIM, hidden_dim)
        self.action_scene_projection = nn.Linear(condition_dim, hidden_dim)
        self.degree_scene_projection = nn.Linear(condition_dim, hidden_dim)
        self.null_degree_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.point_time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.action_time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            JointFlowLayer(
                hidden_dim, heads, mlp_ratio, dropout, self_axis_dims, history_axis_dims,
            )
            for _ in range(layers)
        ])
        self.point_norm = AdaRMSNorm(hidden_dim)
        self.action_norm = AdaRMSNorm(hidden_dim)
        self.point_output = nn.Linear(hidden_dim, POINT_FEATURE_DIM)
        self.action_output = nn.Linear(hidden_dim, ACTION_DIM)
        self.gradient_checkpointing = False

    def encode_scene(self, scene_condition: torch.Tensor) -> torch.Tensor:
        if scene_condition.ndim != 3 or scene_condition.shape[1] != 2:
            raise ValueError(f"Expected scene_condition [B,2,C], got {scene_condition.shape}")
        action_token = self.action_scene_projection(scene_condition[:, 0])
        projected_degree = self.degree_scene_projection(scene_condition[:, 1])
        has_degree = scene_condition[:, 1].abs().sum(dim=-1, keepdim=True) > 0
        degree_token = torch.where(
            has_degree,
            projected_degree,
            self.null_degree_token.expand(scene_condition.shape[0], -1),
        )
        return self.scene_norm(torch.stack([action_token, degree_token], dim=1))

    def _point_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32)
        return torch.stack([
            time.repeat_interleave(self.points_per_step),
            self.point_role_ids.to(device=device, dtype=torch.float32).repeat(self.horizon),
            self.point_slot_ids.to(device=device).repeat(self.horizon),
        ], dim=-1)

    def _action_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32) + 0.5
        return torch.stack([time, torch.full_like(time, -1.0), torch.full_like(time, -1.0)], dim=-1)

    def _point_history_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32)
        slots = self.point_role_ids.to(device=device, dtype=torch.float32) * self.cls_token_num
        return torch.stack([
            time.repeat_interleave(self.points_per_step), slots.repeat(self.horizon),
        ], dim=-1)

    def _action_history_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32) + 0.5
        return torch.stack([time, torch.full_like(time, -1.0)], dim=-1)

    def _point_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma[:, :, None].expand(-1, -1, self.points_per_step).flatten(1)
        return self.point_time_mlp(_sinusoidal_time(sigma, self.point_projection.out_features))

    def _action_condition(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.action_time_mlp(_sinusoidal_time(sigma, self.action_projection.out_features))

    @staticmethod
    def _attention_masks(
        mode: str,
        point_valid: torch.Tensor,
        action_valid: torch.Tensor,
        point_single_length: int,
        action_single_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = point_valid.shape[0]
        point_length, action_length = point_valid.shape[1], action_valid.shape[1]
        total = point_length + action_length
        allow = torch.zeros(total, total, dtype=torch.bool, device=point_valid.device)
        if mode == "joint":
            allow[:] = True
        elif mode == "point_then_action":
            p, a = point_single_length, action_single_length
            pn, pc = slice(0, p), slice(p, 2 * p)
            an = slice(2 * p, 2 * p + a)
            allow[pn, pn] = True
            allow[pc, pc] = True
            allow[an, pc] = True
            allow[an, an] = True
        else:
            raise ValueError(f"Unsupported flow mode: {mode}")
        key_valid = torch.cat([point_valid, action_valid], dim=1)
        mask = allow[None].expand(batch, -1, -1) & key_valid[:, None, :]
        return mask[:, :point_length], mask[:, point_length:]

    def _run(
        self,
        point_state: torch.Tensor,
        point_sigma: torch.Tensor,
        point_valid: torch.Tensor,
        action_state: torch.Tensor,
        action_sigma: torch.Tensor,
        history_memory: torch.Tensor,
        history_positions: torch.Tensor,
        scene_memory: torch.Tensor,
        *,
        mode: str,
        point_clean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = point_state.shape[0]
        point_single_length = self.horizon * self.points_per_step
        action_single_length = self.horizon
        point_flat = point_state.reshape(batch, point_single_length, POINT_FEATURE_DIM)
        action_flat = action_state.reshape(batch, action_single_length, ACTION_DIM)
        point_condition = self._point_condition(point_sigma)
        action_condition = self._action_condition(action_sigma)
        point_positions = self._point_positions(point_state.device)
        action_positions = self._action_positions(action_state.device)
        point_history_positions = self._point_history_positions(point_state.device)
        action_history_positions = self._action_history_positions(action_state.device)
        point_valid_flat = point_valid.flatten(1)
        action_valid = torch.ones(batch, action_single_length, dtype=torch.bool, device=action_state.device)

        if mode == "point_then_action":
            if point_clean is None:
                raise ValueError("point_then_action requires a clean point stream")
            point_flat = torch.cat([
                point_flat,
                point_clean.reshape(batch, point_single_length, POINT_FEATURE_DIM),
            ], dim=1)
            point_condition = torch.cat([point_condition, torch.zeros_like(point_condition)], dim=1)
            point_positions = torch.cat([point_positions, point_positions], dim=0)
            point_history_positions = torch.cat([
                point_history_positions, point_history_positions,
            ], dim=0)
            point_valid_flat = torch.cat([point_valid_flat, point_valid_flat], dim=1)

        point_token = self.point_projection(point_flat)
        action_token = self.action_projection(action_flat)
        point_mask, action_mask = self._attention_masks(
            mode, point_valid_flat, action_valid, point_single_length, action_single_length,
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                point_token, action_token = checkpoint(
                    layer,
                    point_token,
                    action_token,
                    point_condition,
                    action_condition,
                    point_positions,
                    action_positions,
                    point_history_positions,
                    action_history_positions,
                    history_memory,
                    history_positions,
                    scene_memory,
                    point_mask,
                    action_mask,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                point_token, action_token = layer(
                    point_token,
                    action_token,
                    point_condition,
                    action_condition,
                    point_positions,
                    action_positions,
                    point_history_positions,
                    action_history_positions,
                    history_memory,
                    history_positions,
                    scene_memory,
                    point_mask,
                    action_mask,
                )
        point_hidden, _ = self.point_norm(point_token, point_condition)
        action_hidden, _ = self.action_norm(action_token, action_condition)
        point_velocity = self.point_output(point_hidden[:, :point_single_length]).view(
            batch, self.horizon, self.points_per_step, POINT_FEATURE_DIM,
        )
        action_velocity = self.action_output(action_hidden[:, :action_single_length])
        return point_velocity, action_velocity

    def forward_joint(
        self,
        point_state: torch.Tensor,
        point_sigma: torch.Tensor,
        point_valid: torch.Tensor,
        action_state: torch.Tensor,
        action_sigma: torch.Tensor,
        history_memory: torch.Tensor,
        history_positions: torch.Tensor,
        scene_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run(
            point_state, point_sigma, point_valid,
            action_state, action_sigma,
            history_memory, history_positions, scene_memory,
            mode="joint",
        )

    def forward_point_then_action(
        self,
        point_state: torch.Tensor,
        point_sigma: torch.Tensor,
        point_valid: torch.Tensor,
        point_clean: torch.Tensor,
        action_state: torch.Tensor,
        action_sigma: torch.Tensor,
        history_memory: torch.Tensor,
        history_positions: torch.Tensor,
        scene_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run(
            point_state,
            point_sigma,
            point_valid,
            action_state,
            action_sigma,
            history_memory,
            history_positions,
            scene_memory,
            mode="point_then_action",
            point_clean=point_clean,
        )


class GraphFlowModel(nn.Module):
    """Entity encoder plus joint future-point/action flow experts."""

    def __init__(
        self,
        num_points: int = 32,
        cls_token_num: int = 1,
        history_horizon: int = 9,
        future_horizon: int = 10,
        condition_dim: int = 384,
        hidden_dim: int = 768,
        encoder_layers: int = 8,
        flow_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        flow_mode: str = "joint",
        include_future_object_point: bool = True,
        point_num_train_timesteps: int = 1000,
        action_num_train_timesteps: int = 1000,
        point_sigma_shift: float = 5.0,
        action_sigma_shift: float = 1.0,
        point_sample_steps: int = 10,
        action_sample_steps: int = 10,
        complete_pos_weight: float = 1.0,
        contact_pos_weight: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if num_points < ACTOR_NUM_POINTS:
            raise ValueError(f"num_points must be at least {ACTOR_NUM_POINTS}, got {num_points}")
        if flow_mode not in FLOW_MODES:
            raise ValueError(f"flow_mode must be one of {sorted(FLOW_MODES)}, got {flow_mode!r}")
        if flow_mode == "joint" and point_sample_steps != action_sample_steps:
            raise ValueError("joint flow requires matching point/action sample steps")
        if min(point_num_train_timesteps, action_num_train_timesteps, point_sample_steps, action_sample_steps) <= 0:
            raise ValueError("Flow timestep counts must be positive")
        if complete_pos_weight <= 0 or contact_pos_weight <= 0:
            raise ValueError("Completion/contact positive weights must be positive")
        self.num_points = num_points
        self.cls_token_num = cls_token_num
        self.history_horizon = history_horizon
        self.history_steps = history_horizon + 1
        self.future_horizon = future_horizon
        self.flow_mode = flow_mode
        self.include_future_object_point = include_future_object_point
        self.point_num_train_timesteps = point_num_train_timesteps
        self.action_num_train_timesteps = action_num_train_timesteps
        self.point_sigma_shift = point_sigma_shift
        self.action_sigma_shift = action_sigma_shift
        self.point_sample_steps = point_sample_steps
        self.action_sample_steps = action_sample_steps
        self.complete_pos_weight = float(complete_pos_weight)
        self.contact_pos_weight = float(contact_pos_weight)
        self.weights = dict(weights or {})
        self.encoder = EntityEncoder(
            hidden_dim, encoder_layers, num_heads, mlp_ratio,
            max_history=self.history_steps, cls_token_num=cls_token_num, dropout=dropout,
        )
        self.flow = PointActionFlow(
            hidden_dim,
            future_horizon,
            num_points,
            include_future_object_point,
            cls_token_num,
            condition_dim,
            flow_layers,
            num_heads,
            mlp_ratio,
            dropout,
        )
        self.complete_head = nn.Sequential(
            nn.Linear(hidden_dim * (2 * cls_token_num + 2), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contact_query = nn.Parameter(torch.zeros(1, future_horizon, hidden_dim))
        self.contact_pool_attn = RotaryAttention(hidden_dim, num_heads=num_heads, dropout=0.0)
        self.contact_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.contact_query, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.gradient_checkpointing = enabled
        self.flow.gradient_checkpointing = enabled

    def _encode(
        self, batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = batch["entity_points"]
        expected = (self.history_steps, NUM_ENTITIES, self.num_points, POINT_FEATURE_DIM)
        if points.ndim != 5 or tuple(points.shape[1:]) != expected:
            raise ValueError(f"Expected entity points [B,{','.join(map(str, expected))}], got {points.shape}")
        history_memory, history_positions, relation_local = self.encoder(
            points, batch["entity_point_mask"],
        )
        scene_memory = self.flow.encode_scene(batch["scene_condition"])
        return history_memory, history_positions, scene_memory, relation_local

    def _compact_points(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.future_horizon, NUM_ENTITIES, self.num_points, POINT_FEATURE_DIM)
        if points.ndim != 5 or tuple(points.shape[1:]) != expected:
            raise ValueError(f"Expected target points [B,{','.join(map(str, expected))}], got {points.shape}")
        if tuple(mask.shape) != tuple(points.shape[:-1]):
            raise ValueError(f"Point mask {mask.shape} does not match points {points.shape}")
        compact_points = points[:, :, 0, :ACTOR_NUM_POINTS]
        compact_mask = mask[:, :, 0, :ACTOR_NUM_POINTS]
        if self.include_future_object_point:
            batch = points.shape[0]
            object_points = points[:, :, 1:].reshape(
                batch, self.future_horizon, 2 * self.num_points, POINT_FEATURE_DIM,
            )
            object_mask = mask[:, :, 1:].reshape(
                batch, self.future_horizon, 2 * self.num_points,
            )
            compact_points = torch.cat([compact_points, object_points], dim=2)
            compact_mask = torch.cat([compact_mask, object_mask], dim=2)
        return compact_points, compact_mask.bool()

    def _inference_point_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        current = batch["entity_point_mask"][:, -1]
        actor = current[:, 0, :ACTOR_NUM_POINTS]
        if self.include_future_object_point:
            actor = torch.cat([actor, current[:, 1:].flatten(1)], dim=1)
        return actor[:, None].expand(-1, self.future_horizon, -1).bool()

    def _contact_logits(self, history_memory: torch.Tensor) -> torch.Tensor:
        """Predict the future contact profile from the patient CLS history.

        One learnable contact query per future step cross-attends over the
        patient CLS trajectory (history frames at positions -history_steps+1
        ... 0) from its own future-time position (1 ... future_horizon),
        producing per-step tokens that the contact head reads out into
        [B, future_horizon] logits.
        """
        batch = history_memory.shape[0]
        hidden_dim = history_memory.shape[-1]
        patient_history = history_memory.view(
            batch, self.history_steps, NUM_ENTITIES, self.cls_token_num, hidden_dim,
        )[:, :, 1]                                        # [B, T, cls, D]
        patient_history = patient_history.reshape(
            batch, self.history_steps * self.cls_token_num, hidden_dim,
        )
        positions = torch.arange(
            1 - self.history_steps, 1, device=history_memory.device, dtype=torch.float32,
        )
        if self.cls_token_num > 1:
            positions = positions.repeat_interleave(self.cls_token_num)
        query = self.contact_query.expand(batch, -1, -1)
        query_positions = torch.arange(
            1, self.future_horizon + 1, device=history_memory.device, dtype=torch.float32,
        )
        contact_token = self.contact_pool_attn(
            query, patient_history, query_positions, positions,
        )
        return self.contact_head(contact_token).squeeze(-1)

    def _point_flow_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        timestep_weight: torch.Tensor,
    ) -> torch.Tensor:
        squared = (prediction.float() - target.float().detach()).square()
        squared = squared * timestep_weight.float()[:, :, None, None]
        role_losses, role_active = [], []
        for role in self.flow.point_role_ids.unique(sorted=True):
            role_slots = self.flow.point_role_ids == role
            valid = mask[:, :, role_slots]
            numerator = (squared[:, :, role_slots] * valid[..., None]).sum(dim=(1, 2, 3))
            denominator = (valid.sum(dim=(1, 2)) * POINT_FEATURE_DIM).clamp_min(1)
            role_losses.append(numerator / denominator)
            role_active.append(valid.any(dim=(1, 2)))
        losses = torch.stack(role_losses, dim=1)
        active = torch.stack(role_active, dim=1)
        return ((losses * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)).mean()

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_memory, history_positions, scene_memory, relation_local = self._encode(batch)
        target = batch["target"]
        action = target["action"]
        if tuple(action.shape[1:]) != (self.future_horizon, ACTION_DIM):
            raise ValueError(f"Expected target action [B,{self.future_horizon},{ACTION_DIM}], got {action.shape}")
        points, point_mask = self._compact_points(target["points"], target["point_mask"])
        point_scheduler = FlowMatchScheduler(self.point_num_train_timesteps, self.point_sigma_shift)
        action_scheduler = FlowMatchScheduler(self.action_num_train_timesteps, self.action_sigma_shift)
        point_state, point_target, point_sigma, point_weight = point_scheduler.sample_training(points)
        action_state, action_target, action_sigma, action_weight = action_scheduler.sample_training(action)
        point_state = point_state * point_mask[..., None]
        point_target = point_target * point_mask[..., None]

        if self.flow_mode == "joint":
            point_velocity, action_velocity = self.flow.forward_joint(
                point_state, point_sigma, point_mask,
                action_state, action_sigma,
                history_memory, history_positions, scene_memory,
            )
        else:
            point_velocity, action_velocity = self.flow.forward_point_then_action(
                point_state,
                point_sigma,
                point_mask,
                points * point_mask[..., None],
                action_state,
                action_sigma,
                history_memory,
                history_positions,
                scene_memory,
            )
        loss_point_flow = self._point_flow_loss(
            point_velocity, point_target, point_mask, point_weight,
        )
        loss_action_flow = (
            (action_velocity.float() - action_target.float().detach()).square()
            * action_weight.float()[:, :, None]
        ).mean()
        complete_logits = self.complete_head(torch.cat([
            relation_local.flatten(1), scene_memory.flatten(1),
        ], dim=-1))
        loss_complete = F.binary_cross_entropy_with_logits(
            complete_logits,
            target["is_complete"].to(dtype=complete_logits.dtype),
            pos_weight=complete_logits.new_tensor([self.complete_pos_weight]),
        )
        contact_logits = self._contact_logits(history_memory)
        target_contact = target["is_contact_future"].to(dtype=contact_logits.dtype)
        if tuple(contact_logits.shape) != tuple(target_contact.shape):
            raise ValueError(
                f"Expected contact logits {tuple(contact_logits.shape)} to match "
                f"target is_contact_future {tuple(target_contact.shape)}"
            )
        loss_contact = F.binary_cross_entropy_with_logits(
            contact_logits,
            target_contact,
            pos_weight=contact_logits.new_tensor([self.contact_pos_weight]),
        )
        losses = {
            "loss_point_flow": loss_point_flow,
            "loss_action_flow": loss_action_flow,
            "loss_complete": loss_complete,
            "loss_contact": loss_contact,
        }
        total = sum(value * float(self.weights.get(name, 1.0)) for name, value in losses.items())
        return total, {"loss": total.detach(), **{name: value.detach() for name, value in losses.items()}}

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, Any],
        *,
        point_num_steps: int | None = None,
        action_num_steps: int | None = None,
        point_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        history_memory, history_positions, scene_memory, relation_local = self._encode(batch)
        batch_size = history_memory.shape[0]
        point_mask = self._inference_point_mask(batch)
        point_shape = (batch_size, self.future_horizon, self.flow.points_per_step, POINT_FEATURE_DIM)
        action_shape = (batch_size, self.future_horizon, ACTION_DIM)
        point_state = torch.randn(point_shape, device=history_memory.device, dtype=history_memory.dtype) if point_noise is None else point_noise
        action_state = torch.randn(action_shape, device=history_memory.device, dtype=history_memory.dtype) if action_noise is None else action_noise
        if tuple(point_state.shape) != point_shape or tuple(action_state.shape) != action_shape:
            raise ValueError(f"Noise shapes must be {point_shape} and {action_shape}")
        point_state = point_state * point_mask[..., None]
        point_steps = point_num_steps or self.point_sample_steps
        action_steps = action_num_steps or self.action_sample_steps
        if self.flow_mode == "joint" and point_steps != action_steps:
            raise ValueError("joint sampling requires matching point/action steps")
        point_scheduler = make_scheduler(
            point_steps, self.point_num_train_timesteps, self.point_sigma_shift, history_memory.device,
        )
        action_scheduler = make_scheduler(
            action_steps, self.action_num_train_timesteps, self.action_sigma_shift, history_memory.device,
        )

        if self.flow_mode == "joint":
            for step_index in range(point_steps):
                point_sigma = point_scheduler.sigmas[step_index].expand(batch_size, self.future_horizon)
                action_sigma = action_scheduler.sigmas[step_index].expand(batch_size, self.future_horizon)
                point_velocity, action_velocity = self.flow.forward_joint(
                    point_state, point_sigma, point_mask,
                    action_state, action_sigma,
                    history_memory, history_positions, scene_memory,
                )
                point_state = point_scheduler.step(point_velocity, step_index, point_state)
                action_state = action_scheduler.step(action_velocity, step_index, action_state)
                point_state = point_state * point_mask[..., None]
        else:
            zero_point = torch.zeros_like(point_state)
            zero_action = torch.zeros_like(action_state)
            zero_point_sigma = torch.zeros(
                batch_size,
                self.future_horizon,
                device=history_memory.device,
                dtype=history_memory.dtype,
            )
            zero_action_sigma = torch.zeros_like(zero_point_sigma)
            for step_index in range(point_steps):
                point_sigma = point_scheduler.sigmas[step_index].expand(batch_size, self.future_horizon)
                point_velocity, _ = self.flow.forward_point_then_action(
                    point_state,
                    point_sigma,
                    point_mask,
                    zero_point,
                    zero_action,
                    zero_action_sigma,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                point_state = point_scheduler.step(point_velocity, step_index, point_state)
                point_state = point_state * point_mask[..., None]
            for step_index in range(action_steps):
                action_sigma = action_scheduler.sigmas[step_index].expand(batch_size, self.future_horizon)
                _, action_velocity = self.flow.forward_point_then_action(
                    point_state,
                    zero_point_sigma,
                    point_mask,
                    point_state,
                    action_state,
                    action_sigma,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                action_state = action_scheduler.step(action_velocity, step_index, action_state)

        return {
            "action_plan": action_state,
            "point_plan": point_state,
            "point_plan_mask": point_mask,
            "is_complete": torch.sigmoid(self.complete_head(torch.cat([
                relation_local.flatten(1), scene_memory.flatten(1),
            ], dim=-1))),
            "contact_profile": torch.sigmoid(self._contact_logits(history_memory)),
        }
