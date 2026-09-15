"""LIBERO adaptation of the architecture in the released Compose by Focus code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from src.policy.checkpointing import GradientCheckpointingMixin, checkpoint_module
from src.policy.dp.blocks import ConditionalUnet1d
from src.policy.cbf_code.clip_text import FrozenClipTextEncoder


class _FixedChainGAT(nn.Module):
    """PyG GATConv equivalent for CbF's fixed 1->2, 0->1 graph plus self-loops."""

    def __init__(self, input_dim: int, output_dim: int, heads: int, *, concat: bool) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.linear = nn.Linear(input_dim, heads * output_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(1, heads, output_dim))
        self.attention_target = nn.Parameter(torch.empty(1, heads, output_dim))
        self.bias = nn.Parameter(torch.empty(heads * output_dim if concat else output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.zeros_(self.bias)

    def forward(self, nodes: Tensor) -> Tensor:
        if nodes.ndim != 3 or nodes.shape[1] != 3:
            raise ValueError(f"Expected three graph nodes [B,3,D], got {tuple(nodes.shape)}")
        features = self.linear(nodes).reshape(nodes.shape[0], 3, self.heads, self.output_dim)
        # edge_index=[[1,0],[2,1]] in the release, followed by PyG self-loops.
        sources = (1, 0, 0, 1, 2)
        targets = (2, 1, 0, 1, 2)
        edge_logits = torch.stack([
            (features[:, source] * self.attention_source).sum(-1)
            + (features[:, target] * self.attention_target).sum(-1)
            for source, target in zip(sources, targets, strict=True)
        ], dim=1)
        edge_logits = F.leaky_relu(edge_logits, negative_slope=0.2)
        outputs = []
        for target in range(3):
            indices = [index for index, item in enumerate(targets) if item == target]
            weights = edge_logits[:, indices].softmax(dim=1)
            messages = torch.stack([features[:, sources[index]] for index in indices], dim=1)
            outputs.append((weights[..., None] * messages).sum(dim=1))
        output = torch.stack(outputs, dim=1)
        output = output.flatten(2) if self.concat else output.mean(dim=2)
        return output + self.bias


class _GraphEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gat1 = _FixedChainGAT(128, 128, 4, concat=True)
        self.gat2 = _FixedChainGAT(512, 512, 1, concat=False)

    def forward(self, nodes: Tensor) -> Tensor:
        return self.gat2(F.elu(self.gat1(nodes))).mean(dim=1)


class CbFCodePolicy(GradientCheckpointingMixin, nn.Module):
    """CbF release architecture with 32-point entities and 7D LIBERO deltas."""

    def __init__(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        num_points: int = 32,
        obs_steps: int = 2,
        horizon: int = 16,
        action_steps: int = 8,
        clip_model_path: str | Path | None = None,
        clip_bpe_path: str | Path | None = None,
        diffusion_steps: int = 100,
        inference_steps: int = 100,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        num_groups: int = 8,
        semantic_edge_injection: bool = False,
    ) -> None:
        super().__init__()
        if semantic_edge_injection:
            raise ValueError("The released CbF code does not configure GAT edge_dim")
        if obs_steps != 2 or horizon != 16 or action_steps != 8:
            raise ValueError("Code-faithful CbF requires obs_steps=2, horizon=16, action_steps=8")
        if num_points < 1 or state_dim < 1 or action_dim < 1:
            raise ValueError("num_points, state_dim, and action_dim must be positive")
        if not down_dims or horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must be divisible by the U-Net downsampling factor")
        if not 1 <= inference_steps <= diffusion_steps:
            raise ValueError("inference_steps must be in [1, diffusion_steps]")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_points = num_points
        self.obs_steps = obs_steps
        self.horizon = horizon
        self.action_steps = action_steps
        self.inference_steps = inference_steps
        point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.nets = nn.ModuleDict({
            "point_mlp": point_mlp,
            "point_projection": nn.Linear(256, 64),
            "state_mlp": nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64)),
            "graph_convolution": _GraphEncoder(),
            "mlp_language": nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512),
            ),
            "invariant": ConditionalUnet1d(
                action_dim,
                (512 + state_dim + 512) * obs_steps,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                groups=num_groups,
                predict_scale=True,
            ),
        })
        object.__setattr__(self, "_language_encoder", (
            FrozenClipTextEncoder(clip_model_path, clip_bpe_path)
            if clip_model_path is not None and clip_bpe_path is not None else None
        ))
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _language(self, batch: dict[str, Any], device: torch.device) -> Tensor:
        if "language_embedding" in batch:
            embedding = torch.as_tensor(batch["language_embedding"], device=device).float()
        else:
            texts = batch.get("language")
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, (list, tuple)) or not all(isinstance(text, str) for text in texts):
                raise TypeError("CbF-Code requires language strings or language_embedding")
            if self._language_encoder is None:
                raise RuntimeError("clip_model_path and clip_bpe_path are required for language strings")
            embedding = self._language_encoder.encode(texts, device=device).clone()
        if embedding.ndim == 3 and embedding.shape[1] == self.obs_steps:
            embedding = embedding[:, 0]
        if embedding.ndim != 2 or embedding.shape[-1] != 512:
            raise ValueError(f"Expected CLIP language embedding [B,512], got {tuple(embedding.shape)}")
        return embedding

    def _condition(self, batch: dict[str, Any], nets: nn.ModuleDict) -> Tensor:
        points = batch["node_point_clouds"]
        state = batch["state"]
        expected_points = (points.shape[0], self.obs_steps, 3, self.num_points, 3)
        if tuple(points.shape) != expected_points:
            raise ValueError(f"Expected node_point_clouds [B,{self.obs_steps},3,{self.num_points},3], got {tuple(points.shape)}")
        if tuple(state.shape) != (points.shape[0], self.obs_steps, self.state_dim):
            raise ValueError(f"Expected state [B,{self.obs_steps},{self.state_dim}], got {tuple(state.shape)}")
        point_features = nets["point_mlp"](points).amax(dim=3)
        point_features = nets["point_projection"](point_features)
        state_features = nets["state_mlp"](state).unsqueeze(2).expand(-1, -1, 3, -1)
        nodes = torch.cat((point_features, state_features), dim=-1).flatten(0, 1)
        graph = checkpoint_module(nets["graph_convolution"], nodes).reshape(points.shape[0], self.obs_steps, 512)
        language = nets["mlp_language"](self._language(batch, state.device))
        language = language[:, None].expand(-1, self.obs_steps, -1)
        return torch.cat((graph, state, language), dim=-1).flatten(1)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        actions = batch["actions"]
        if tuple(actions.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(f"Expected actions [B,{self.horizon},{self.action_dim}], got {tuple(actions.shape)}")
        condition = self._condition(batch, self.nets)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],), device=actions.device, dtype=torch.long,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.nets["invariant"](noisy_actions, timesteps, condition)
        loss = F.mse_loss(prediction, noise)
        return loss, {"loss_mse": loss.detach()}

    @torch.inference_mode()
    def predict_action(self, batch: dict[str, Any]) -> Tensor:
        condition = self._condition(batch, self.nets)
        sample = torch.randn(
            (condition.shape[0], self.horizon, self.action_dim),
            device=condition.device, dtype=condition.dtype,
        )
        self.noise_scheduler.set_timesteps(self.inference_steps, device=sample.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.nets["invariant"](sample, timestep, condition)
            sample = self.noise_scheduler.step(prediction, timestep, sample).prev_sample
        start = self.obs_steps - 1
        return sample[:, start : start + self.action_steps]

    def sample(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        return {"action_plan": self.predict_action(batch)}
