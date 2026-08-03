"""Joint future-point and action Flow Matching model for GraphVLA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.common.schema import ACTION_DIM, ACTOR_POINT_INDICES, NUM_ENTITIES, POINT_FEATURE_DIM
from src.model.encoder import EntityEncoder
from src.model.flow_matching import FlowMatchScheduler, make_scheduler
from src.model.temporal import AdaRMSNorm, RotaryAttention, RotaryFlowBlock


FLOW_MODES = {"joint", "point_then_action", "action_only", "point_only"}


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
        self.point_feature_dim = len(ACTOR_POINT_INDICES) * POINT_FEATURE_DIM + 1
        self.point_projection = nn.Linear(self.point_feature_dim, hidden_dim)
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
        self.point_output = nn.Linear(hidden_dim, self.point_feature_dim)
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
            time,
            torch.zeros_like(time),
            torch.full_like(time, -1.0),
        ], dim=-1)

    def _action_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32) + 0.5
        return torch.stack([time, torch.full_like(time, -1.0), torch.full_like(time, -1.0)], dim=-1)

    def _point_history_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32)
        return torch.stack([
            time, torch.zeros_like(time),
        ], dim=-1)

    def _action_history_positions(self, device: torch.device) -> torch.Tensor:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32) + 0.5
        return torch.stack([time, torch.full_like(time, -1.0)], dim=-1)

    def _point_condition(self, sigma: torch.Tensor) -> torch.Tensor:
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
        point_single_length = self.horizon
        action_single_length = self.horizon
        expected_point_shape = (self.horizon, self.point_feature_dim)
        if tuple(point_state.shape[1:]) != expected_point_shape:
            raise ValueError(
                f"Expected actor point state [B,{self.horizon},{self.point_feature_dim}], "
                f"got {point_state.shape}"
            )
        point_flat = point_state
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
                point_clean,
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
        point_velocity = self.point_output(point_hidden[:, :point_single_length])
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


class SingleFlowLayer(RotaryFlowBlock):
    """One standalone point or action flow layer."""

    def forward(
        self,
        token: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        query_history_positions: torch.Tensor,
        history_memory: torch.Tensor,
        history_positions: torch.Tensor,
        scene_memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value, gate = self.prepare_self_attention(token, condition, positions)
        update = self.self_attention.attend(query, key, value, attention_mask)
        return self.finish(
            token,
            update,
            gate,
            history_memory,
            scene_memory,
            condition,
            query_history_positions,
            history_positions,
        )


class SingleStreamFlow(nn.Module):
    """Standalone action flow or flattened actor-point/gripper flow."""

    def __init__(
        self,
        stream: str,
        hidden_dim: int,
        horizon: int,
        condition_dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if stream not in {"point", "action"}:
            raise ValueError(f"Unsupported single flow stream: {stream!r}")
        head_dim = hidden_dim // heads
        if hidden_dim % heads or head_dim % 6:
            raise ValueError(
                f"Three-axis RoPE requires head_dim divisible by 6, got {hidden_dim}/{heads}"
            )
        self.stream = stream
        self.horizon = horizon
        feature_dim = (
            len(ACTOR_POINT_INDICES) * POINT_FEATURE_DIM + 1
            if stream == "point" else ACTION_DIM
        )
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.action_scene_projection = nn.Linear(condition_dim, hidden_dim)
        self.degree_scene_projection = nn.Linear(condition_dim, hidden_dim)
        self.null_degree_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            SingleFlowLayer(
                hidden_dim,
                heads,
                mlp_ratio,
                dropout,
                (head_dim // 3,) * 3,
                (head_dim // 2, head_dim - head_dim // 2),
            )
            for _ in range(layers)
        ])
        self.norm = AdaRMSNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)
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

    def _positions(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        time = torch.arange(1, self.horizon + 1, device=device, dtype=torch.float32)
        if self.stream == "action":
            time = time + 0.5
        positions = torch.stack([
            time,
            torch.full_like(time, -1.0 if self.stream == "action" else 0.0),
            torch.full_like(time, -1.0),
        ], dim=-1)
        history_positions = torch.stack([
            time, torch.full_like(time, -1.0 if self.stream == "action" else 0.0),
        ], dim=-1)
        return positions, history_positions

    def forward(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        valid: torch.Tensor | None,
        history_memory: torch.Tensor,
        history_positions: torch.Tensor,
        scene_memory: torch.Tensor,
    ) -> torch.Tensor:
        batch = state.shape[0]
        feature_dim = self.projection.in_features
        expected = (self.horizon, feature_dim)
        if tuple(state.shape[1:]) != expected:
            raise ValueError(f"Expected {self.stream} state [B,{','.join(map(str, expected))}], got {state.shape}")
        token = self.projection(state)
        condition = self.time_mlp(_sinusoidal_time(sigma, token.shape[-1]))
        positions, query_history_positions = self._positions(state.device)
        if valid is None:
            valid = torch.ones(
                batch, self.horizon, dtype=torch.bool, device=state.device,
            )
        attention_mask = valid[:, None, :].expand(-1, valid.shape[1], -1)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                token = checkpoint(
                    layer,
                    token,
                    condition,
                    positions,
                    query_history_positions,
                    history_memory,
                    history_positions,
                    scene_memory,
                    attention_mask,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                token = layer(
                    token,
                    condition,
                    positions,
                    query_history_positions,
                    history_memory,
                    history_positions,
                    scene_memory,
                    attention_mask,
                )
        hidden, _ = self.norm(token, condition)
        velocity = self.output(hidden)
        return velocity.view(batch, self.horizon, feature_dim)


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
        action_delta: bool = True,
        point_num_train_timesteps: int = 1000,
        action_num_train_timesteps: int = 1000,
        point_sigma_shift: float = 5.0,
        action_sigma_shift: float = 1.0,
        correlated_sigma_sampling: bool = False,
        shared_horizon_sigma_sampling: bool = False,
        point_sample_steps: int = 10,
        action_sample_steps: int = 10,
        complete_pos_weight: float = 1.0,
        contact_pos_weight: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if num_points < len(ACTOR_POINT_INDICES):
            raise ValueError(f"num_points must be at least {len(ACTOR_POINT_INDICES)}, got {num_points}")
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
        self.action_delta = bool(action_delta)
        self.point_num_train_timesteps = point_num_train_timesteps
        self.action_num_train_timesteps = action_num_train_timesteps
        self.point_sigma_shift = point_sigma_shift
        self.action_sigma_shift = action_sigma_shift
        self.correlated_sigma_sampling = bool(correlated_sigma_sampling)
        self.shared_horizon_sigma_sampling = bool(shared_horizon_sigma_sampling)
        self.point_sample_steps = point_sample_steps
        self.action_sample_steps = action_sample_steps
        self.complete_pos_weight = float(complete_pos_weight)
        self.contact_pos_weight = float(contact_pos_weight)
        self.weights = dict(weights or {})
        self.encoder = EntityEncoder(
            hidden_dim, encoder_layers, num_heads, mlp_ratio,
            max_history=self.history_steps, cls_token_num=cls_token_num, dropout=dropout,
        )
        if flow_mode in {"joint", "point_then_action"}:
            self.flow = PointActionFlow(
                hidden_dim, future_horizon, condition_dim, flow_layers,
                num_heads, mlp_ratio, dropout,
            )
        else:
            self.flow = SingleStreamFlow(
                flow_mode.removesuffix("_only"), hidden_dim, future_horizon,
                condition_dim, flow_layers, num_heads, mlp_ratio, dropout,
            )
        self.complete_head = nn.Sequential(
            nn.Linear(hidden_dim * (2 * cls_token_num + 2), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contact_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.contact_pool_attn = RotaryAttention(hidden_dim, num_heads=num_heads, dropout=0.0)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_horizon),
        )
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

    def _actor_points(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.future_horizon, NUM_ENTITIES, self.num_points, POINT_FEATURE_DIM)
        if points.ndim != 5 or tuple(points.shape[1:]) != expected:
            raise ValueError(f"Expected target points [B,{','.join(map(str, expected))}], got {points.shape}")
        if tuple(mask.shape) != tuple(points.shape[:-1]):
            raise ValueError(f"Point mask {mask.shape} does not match points {points.shape}")
        return (
            points[:, :, 0, :len(ACTOR_POINT_INDICES)],
            mask[:, :, 0, :len(ACTOR_POINT_INDICES)].bool(),
        )

    def _inference_point_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        current = batch["entity_point_mask"][:, -1, 0, :len(ACTOR_POINT_INDICES)]
        return current[:, None].expand(-1, self.future_horizon, -1).bool()

    def _contact_logits(self, history_memory: torch.Tensor) -> torch.Tensor:
        """Predict the future contact profile from the patient CLS history.

        One learnable query pools the patient CLS trajectory across any history
        length. An MLP decodes the pooled token into all future contact logits.
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
        contact_token = self.contact_pool_attn(
            query, patient_history, query_positions=None, key_positions=positions,
        )
        return self.contact_head(contact_token[:, 0])


    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_memory, history_positions, scene_memory, relation_local = self._encode(batch)
        target = batch["target"]
        use_points = self.flow_mode != "action_only"
        use_actions = self.flow_mode != "point_only"
        batch_size = history_memory.shape[0]
        point_timestep_ids = action_timestep_ids = None
        if self.correlated_sigma_sampling and use_points and use_actions:
            timestep_shape = (
                (batch_size, 1)
                if self.shared_horizon_sigma_sampling
                else (batch_size, self.future_horizon)
            )
            shared_timestep_ids = torch.randint(
                min(self.point_num_train_timesteps, self.action_num_train_timesteps),
                timestep_shape,
                device=history_memory.device,
            )
            if self.shared_horizon_sigma_sampling:
                shared_timestep_ids = shared_timestep_ids.expand(-1, self.future_horizon)
            point_timestep_ids = action_timestep_ids = shared_timestep_ids
        elif self.shared_horizon_sigma_sampling:
            if use_points:
                point_timestep_ids = torch.randint(
                    self.point_num_train_timesteps,
                    (batch_size, 1),
                    device=history_memory.device,
                ).expand(-1, self.future_horizon)
            if use_actions:
                action_timestep_ids = torch.randint(
                    self.action_num_train_timesteps,
                    (batch_size, 1),
                    device=history_memory.device,
                ).expand(-1, self.future_horizon)

        loss_point_flow = history_memory.new_zeros((), dtype=torch.float32)
        loss_action_flow = history_memory.new_zeros((), dtype=torch.float32)
        if use_points:
            points, point_mask = self._actor_points(target["points"], target["point_mask"])
            point_flow_target = points.flatten(2)
            point_feature_mask = point_mask[..., None].expand(
                -1, -1, -1, POINT_FEATURE_DIM,
            ).flatten(2)
            point_group_valid = point_mask.all(dim=-1)
            action = target["action"]
            if tuple(action.shape[1:]) != (self.future_horizon, ACTION_DIM):
                raise ValueError(
                    f"Expected target action [B,{self.future_horizon},{ACTION_DIM}], got {action.shape}"
                )
            point_flow_target = torch.cat([
                point_flow_target, action[..., -1:],
            ], dim=-1)
            point_feature_mask = torch.cat([
                point_feature_mask,
                torch.ones_like(action[..., -1:], dtype=torch.bool),
            ], dim=-1)
            point_scheduler = FlowMatchScheduler(
                self.point_num_train_timesteps, self.point_sigma_shift,
            )
            point_state, point_target, point_sigma, point_weight = point_scheduler.sample_training(
                point_flow_target, timestep_ids=point_timestep_ids,
            )
            point_state = point_state * point_feature_mask
            point_target = point_target * point_feature_mask
        if use_actions:
            action = target["action"]
            if tuple(action.shape[1:]) != (self.future_horizon, ACTION_DIM):
                raise ValueError(
                    f"Expected target action [B,{self.future_horizon},{ACTION_DIM}], got {action.shape}"
                )
            action_scheduler = FlowMatchScheduler(
                self.action_num_train_timesteps, self.action_sigma_shift,
            )
            action_state, action_target, action_sigma, action_weight = action_scheduler.sample_training(
                action, timestep_ids=action_timestep_ids,
            )

        if self.flow_mode in {"joint", "point_then_action"}:
            if self.flow_mode == "joint":
                point_velocity, action_velocity = self.flow.forward_joint(
                    point_state, point_sigma, point_group_valid,
                    action_state, action_sigma,
                    history_memory, history_positions, scene_memory,
                )
            else:
                point_velocity, action_velocity = self.flow.forward_point_then_action(
                    point_state,
                    point_sigma,
                    point_group_valid,
                    point_flow_target * point_feature_mask,
                    action_state,
                    action_sigma,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
            loss_action_flow = (
                (action_velocity.float() - action_target.float().detach()).square()
                * action_weight.float()[:, :, None]
            ).mean()
        elif self.flow_mode == "point_only":
            point_velocity = self.flow(
                point_state,
                point_sigma,
                None,
                history_memory,
                history_positions,
                scene_memory,
            )
        else:
            action_velocity = self.flow(
                action_state,
                action_sigma,
                None,
                history_memory,
                history_positions,
                scene_memory,
            )
            loss_action_flow = (
                (action_velocity.float() - action_target.float().detach()).square()
                * action_weight.float()[:, :, None]
            ).mean()

        if use_points:
            point_squared = (
                (point_velocity.float() - point_target.float().detach()).square()
                * point_weight.float()[:, :, None]
                * point_feature_mask
            )
            loss_point_flow = (
                point_squared.sum(dim=(1, 2))
                / point_feature_mask.sum(dim=(1, 2)).clamp_min(1)
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
        use_points = self.flow_mode != "action_only"
        use_actions = self.flow_mode != "point_only"
        outputs = {
            "is_complete": torch.sigmoid(self.complete_head(torch.cat([
                relation_local.flatten(1), scene_memory.flatten(1),
            ], dim=-1))),
            "contact_profile": torch.sigmoid(self._contact_logits(history_memory)),
        }

        if use_points:
            point_mask = self._inference_point_mask(batch)
            point_feature_mask = point_mask[..., None].expand(
                -1, -1, -1, POINT_FEATURE_DIM,
            ).flatten(2)
            point_group_valid = point_mask.all(dim=-1)
            point_feature_mask = torch.cat([
                point_feature_mask,
                torch.ones(
                    batch_size, self.future_horizon, 1,
                    dtype=torch.bool, device=point_mask.device,
                ),
            ], dim=-1)
            point_shape = (
                batch_size,
                self.future_horizon,
                point_feature_mask.shape[-1],
            )
            point_state = (
                torch.randn(
                    point_shape,
                    device=history_memory.device,
                    dtype=history_memory.dtype,
                )
                if point_noise is None else point_noise
            )
            if tuple(point_state.shape) != point_shape:
                raise ValueError(f"Point noise shape must be {point_shape}, got {point_state.shape}")
            point_state = point_state * point_feature_mask
            point_steps = point_num_steps or self.point_sample_steps
            point_scheduler = make_scheduler(
                point_steps,
                self.point_num_train_timesteps,
                self.point_sigma_shift,
                history_memory.device,
            )
        if use_actions:
            action_shape = (batch_size, self.future_horizon, ACTION_DIM)
            action_state = (
                torch.randn(
                    action_shape,
                    device=history_memory.device,
                    dtype=history_memory.dtype,
                )
                if action_noise is None else action_noise
            )
            if tuple(action_state.shape) != action_shape:
                raise ValueError(f"Action noise shape must be {action_shape}, got {action_state.shape}")
            action_steps = action_num_steps or self.action_sample_steps
            action_scheduler = make_scheduler(
                action_steps,
                self.action_num_train_timesteps,
                self.action_sigma_shift,
                history_memory.device,
            )

        if self.flow_mode == "joint":
            if point_steps != action_steps:
                raise ValueError("joint sampling requires matching point/action steps")
            for step_index in range(point_steps):
                point_sigma = point_scheduler.sigmas[step_index].expand(
                    batch_size, self.future_horizon,
                )
                action_sigma = action_scheduler.sigmas[step_index].expand(
                    batch_size, self.future_horizon,
                )
                point_velocity, action_velocity = self.flow.forward_joint(
                    point_state,
                    point_sigma,
                    point_group_valid,
                    action_state,
                    action_sigma,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                point_state = point_scheduler.step(point_velocity, step_index, point_state)
                action_state = action_scheduler.step(action_velocity, step_index, action_state)
                point_state = point_state * point_feature_mask
        elif self.flow_mode == "point_then_action":
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
                point_sigma = point_scheduler.sigmas[step_index].expand(
                    batch_size, self.future_horizon,
                )
                point_velocity, _ = self.flow.forward_point_then_action(
                    point_state,
                    point_sigma,
                    point_group_valid,
                    zero_point,
                    zero_action,
                    zero_action_sigma,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                point_state = point_scheduler.step(point_velocity, step_index, point_state)
                point_state = point_state * point_feature_mask
            for step_index in range(action_steps):
                action_sigma = action_scheduler.sigmas[step_index].expand(
                    batch_size, self.future_horizon,
                )
                _, action_velocity = self.flow.forward_point_then_action(
                    point_state,
                    zero_point_sigma,
                    point_group_valid,
                    point_state,
                    action_state,
                    action_sigma,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                action_state = action_scheduler.step(
                    action_velocity, step_index, action_state,
                )
        elif self.flow_mode == "point_only":
            for step_index in range(point_steps):
                point_sigma = point_scheduler.sigmas[step_index].expand(
                    batch_size, self.future_horizon,
                )
                point_velocity = self.flow(
                    point_state,
                    point_sigma,
                    None,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                point_state = point_scheduler.step(point_velocity, step_index, point_state)
                point_state = point_state * point_feature_mask
        else:
            for step_index in range(action_steps):
                action_sigma = action_scheduler.sigmas[step_index].expand(
                    batch_size, self.future_horizon,
                )
                action_velocity = self.flow(
                    action_state,
                    action_sigma,
                    None,
                    history_memory,
                    history_positions,
                    scene_memory,
                )
                action_state = action_scheduler.step(
                    action_velocity, step_index, action_state,
                )

        if use_points:
            point_coordinates = point_state[..., :len(ACTOR_POINT_INDICES) * POINT_FEATURE_DIM]
            outputs["point_plan"] = point_coordinates.view(
                batch_size, self.future_horizon, len(ACTOR_POINT_INDICES), POINT_FEATURE_DIM,
            )
            outputs["gripper_plan"] = point_state[..., -1]
            outputs["point_plan_mask"] = point_mask
        if use_actions:
            outputs["action_plan"] = action_state
        return outputs
