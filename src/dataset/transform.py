from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import json
import numpy as np
import torch
from rich.console import Console
from src.common.geom_utils import uv_to_normalized_ray_torch
from src.common.schema import (
    ACTOR_POINT_INDICES,
    ACTOR_NUM_POINTS,
    ENTITY_ROLES,
    POINT_FEATURE_DIM,
)
cs = Console()
DataDict = dict[str, Any]

class TransformFn:
    def __call__(self, data: DataDict) -> DataDict:
        raise NotImplementedError

@dataclass
class Compose(TransformFn):
    transforms: Sequence[TransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class FlattenTransform(TransformFn):
    fields: Sequence[str]

    def __call__(self, data: DataDict) -> DataDict:
        for field in self.fields:
            data[field] = self._flatten(data[field])
        return data

    def _flatten(self, value: Any) -> Any:
        ndim = getattr(value, "ndim", None)
        if ndim is None:
            ndim = len(getattr(value, "shape"))
        if ndim < 2:
            raise ValueError(f"Cannot flatten value with ndim={ndim}; expected at least 2 dimensions")
        if hasattr(value, "flatten"):
            return value.flatten(start_dim=1)
        shape = value.shape
        return value.reshape(shape[0], -1)


@dataclass
class SubtaskBoundryPadding(TransformFn):
    fields: Sequence[str] = (
        "subtask_id",
        "is_complete",
        "node_points_track",
        "node_points_mask",
        "depths.depth_rel",
        "gripper_uv",
        "gripper_d",
        "gripper_openness",
        "gripper_openness_mask",
        "action",
    )

    def __call__(self, data: DataDict) -> DataDict:
        if "subtask_id" in data:
            subtask = data["subtask_id"]
        elif "subtask" in data:
            subtask = data["subtask"]
        else:
            raise KeyError("SubtaskBoundryPadding requires subtask_id or subtask")

        history_horizon = int(data["history_horizon"])
        device = subtask.device if isinstance(subtask, torch.Tensor) else None
        time_indices = self._nearest_same_subtask_indices(
            subtask,
            current_index=history_horizon,
            device=device,
        )
        for field in self.fields:
            if field in data:
                data[field] = self._index_first_dim(data[field], time_indices)
        return data

    def _nearest_same_subtask_indices(
        self,
        subtask: Any,
        *,
        current_index: int,
        device: torch.device | None,
    ) -> torch.Tensor:
        if self._is_string_sequence(subtask):
            values = self._string_values(subtask)
            frame_indices = torch.arange(len(values), dtype=torch.long)
            current_subtask = values[current_index]
            same_indices = torch.tensor(
                [index for index, value in enumerate(values) if value == current_subtask],
                dtype=torch.long,
            )
        else:
            subtask = torch.as_tensor(subtask, device=device).reshape(-1)
            frame_indices = torch.arange(subtask.numel(), device=subtask.device)
            current_subtask = subtask[current_index]
            same_indices = torch.nonzero(subtask == current_subtask, as_tuple=False).flatten()

        if same_indices.numel() == 0:
            return frame_indices
        distances = (frame_indices[:, None] - same_indices[None]).abs()
        return same_indices[distances.argmin(dim=1)].long()

    def _is_string_sequence(self, value: Any) -> bool:
        if isinstance(value, (str, bytes)):
            return True
        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"}:
            return True
        if isinstance(value, Sequence) and not isinstance(value, torch.Tensor):
            return any(isinstance(item, (str, bytes)) for item in value)
        return False

    def _string_values(self, value: Any) -> list[str]:
        if isinstance(value, (str, bytes)):
            return [str(value).strip().lower()]
        if isinstance(value, np.ndarray):
            return [str(item).strip().lower() for item in value.reshape(-1).tolist()]
        return [str(item).strip().lower() for item in value]

    def _index_first_dim(self, value: Any, indices: torch.Tensor) -> Any:
        if not hasattr(value, "shape") or len(value.shape) == 0 or value.shape[0] != indices.numel():
            return value
        if isinstance(value, torch.Tensor):
            return value.index_select(0, indices.to(device=value.device))
        return value[indices.cpu().numpy()]


@dataclass
class RepackTransform(TransformFn):
    structure: Mapping[str, Any]

    def __call__(self, data: DataDict) -> DataDict:
        return self._repack(self.structure, data)

    def _repack(self, structure: Mapping[str, Any], data: DataDict) -> DataDict:
        result: DataDict = {}
        for key, value in structure.items():
            if isinstance(value, str):
                result[key] = data[value]
            elif isinstance(value, Mapping):
                result[key] = self._repack(value, data)
            else:
                raise TypeError(f"Unsupported repack spec for key={key!r}: {value!r}")
        return result


@dataclass
class PromptFromTask(TransformFn):
    tasks: Mapping[int, str]
    task_index_path: tuple[str, ...] = ("metadata", "task_index")
    output_key: str = "prompt"

    def __call__(self, data: DataDict) -> DataDict:
        value = self._get_nested(data, self.task_index_path)
        task_index = int(value.item() if hasattr(value, "item") else value)
        if task_index not in self.tasks:
            raise KeyError(f"task_index={task_index} not found in task table")

        return {**data, self.output_key: self.tasks[task_index]}

    def _get_nested(self, data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = data
        for key in path:
            current = current[key]
        return current


@dataclass
class ResizeImages(TransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        if "images" in data:
            data["images"] = self._resize_mapping(data["images"], mode="bilinear")
        if "depths" in data:
            data["depths"] = self._resize_mapping(data["depths"], mode="nearest")
        return data

    def _resize_mapping(self, values: Mapping[str, Any], mode: str) -> dict[str, Any]:
        return {key: self._resize_value(value, mode=mode) for key, value in values.items()}

    def _resize_value(self, value: Any, *, mode: str) -> Any:
        is_numpy = isinstance(value, np.ndarray)
        tensor = torch.as_tensor(value) if is_numpy else value
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Cannot resize value of type {type(value)!r}")

        original_shape = tuple(tensor.shape)
        channel_last = self._is_channel_last(tensor)
        tensor = self._to_nchw(tensor, channel_last=channel_last)
        leading_shape = tensor.shape[:-3]
        channels, old_height, old_width = tensor.shape[-3:]
        tensor = tensor.reshape(-1, channels, old_height, old_width).float()

        kwargs = {"size": (self.height, self.width), "mode": mode}
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        tensor = torch.nn.functional.interpolate(tensor, **kwargs)
        tensor = tensor.reshape(*leading_shape, channels, self.height, self.width)
        tensor = self._from_nchw(tensor, channel_last=channel_last, original_ndim=len(original_shape))

        if is_numpy:
            return tensor.cpu().numpy().astype(value.dtype, copy=False)
        return tensor.to(dtype=value.dtype, device=value.device)

    def _is_channel_last(self, value: Any) -> bool:
        if value.ndim < 3:
            return False
        return value.shape[-1] in {1, 3, 4}

    def _to_nchw(self, value: Any, *, channel_last: bool) -> Any:
        if value.ndim == 2:
            return value.unsqueeze(0)
        if channel_last:
            return value.movedim(-1, -3)
        return value

    def _from_nchw(self, value: Any, *, channel_last: bool, original_ndim: int) -> Any:
        if original_ndim == 2:
            return value.squeeze(0)
        if channel_last:
            return value.movedim(-3, -1)
        return value


@dataclass
class Normalize(TransformFn):
    norm_stats: Mapping[str, Any] | None
    use_quantiles: bool = True
    quantile_to_neg_one_one: bool = True
    eps: float = 1e-6
    task_index_path: tuple[str, ...] = ("task_index",)
    episode_index_path: tuple[str, ...] = ("episode_index",)

    def __call__(self, data: DataDict) -> DataDict:
        level = self._stats_level(self.norm_stats)
        stats = self._unwrap_stats(self.norm_stats)
        if stats is None:
            return data

        for key, field_stats in stats.items():
            if key not in data:
                cs.print(f"[Normalize] skip missing field: {key}")
                continue
            value = data[key]
            selected_stats = self._select_field_stats(field_stats, level, data)
            value = (
                self._normalize_quantile(value, selected_stats)
                if self.use_quantiles
                else self._normalize(value, selected_stats)
            )
            data[key] = value
        return data

    def _normalize(self, value: Any, stats: Mapping[str, Any]) -> Any:
        mean = self._stats_like(stats["mean"], value)
        std = self._stats_like(stats["std"], value)
        mean = self._match_last_dim(mean, value.shape[-1])
        std = self._match_last_dim(std, value.shape[-1])
        return (value - mean) / (std + self.eps)

    def _normalize_quantile(self, value: Any, stats: Mapping[str, Any]) -> Any:
        q01 = self._stats_like(stats["q01"], value)
        q99 = self._stats_like(stats["q99"], value)
        q01 = self._match_last_dim(q01, value.shape[-1])
        q99 = self._match_last_dim(q99, value.shape[-1])
        normalized = (value - q01) / (q99 - q01 + self.eps)
        if self.quantile_to_neg_one_one:
            return normalized * 2.0 - 1.0
        return normalized

    def _stats_level(self, norm_stats: Mapping[str, Any] | None) -> str:
        if norm_stats is None:
            return "suite"
        return str(norm_stats.get("level", "suite"))

    def _unwrap_stats(self, norm_stats: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if norm_stats is None:
            return None
        return norm_stats.get("norm_stats", norm_stats)

    def _select_field_stats(self, field_stats: Mapping[str, Any], level: str, data: DataDict) -> Mapping[str, Any]:
        if level == "suite":
            return field_stats
        if level == "task":
            group_index = self._get_index(data, self.task_index_path)
        elif level == "episode":
            group_index = self._get_index(data, self.episode_index_path)
        else:
            raise ValueError(f"Unsupported norm stats level: {level}")

        group_key = str(group_index)
        if group_key not in field_stats:
            raise KeyError(f"Missing {level} norm stats for index {group_key}")
        return field_stats[group_key]

    def _get_index(self, data: Mapping[str, Any], path: tuple[str, ...]) -> int:
        try:
            value = self._get_nested(data, path)
        except KeyError:
            value = self._get_nested(data, ("metadata", path[-1]))
        return self._to_int(value)

    def _get_nested(self, data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = data
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                raise KeyError(key)
            current = current[key]
        return current

    def _to_int(self, value: Any) -> int:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).reshape(-1)
        if array.size == 0:
            raise ValueError("Cannot read index from an empty value")
        return int(array[0].item() if hasattr(array[0], "item") else array[0])

    def _stats_like(self, stats: Any, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return torch.as_tensor(stats, dtype=value.dtype, device=value.device)
        if isinstance(value, np.ndarray):
            return np.asarray(stats, dtype=value.dtype)
        raise TypeError(f"Cannot normalize value of type {type(value)!r}")

    def _match_last_dim(self, stats: Any, dim: int) -> Any:
        if stats.shape[-1] == 1:
            return stats
        if stats.shape[-1] < dim:
            raise ValueError(f"Norm stats dim={stats.shape[-1]} is smaller than value dim={dim}")
        return stats[..., :dim]


@dataclass
class Unnormalize(Normalize):

    def _normalize(self, value: Any, stats: Mapping[str, Any]) -> Any:
        mean = self._stats_like(stats["mean"], value)
        std = self._stats_like(stats["std"], value)
        mean = self._pad_last_dim(mean, value.shape[-1], value=0.0)
        std = self._pad_last_dim(std, value.shape[-1], value=1.0)
        return value * (std + self.eps) + mean

    def _normalize_quantile(self, value: Any, stats: Mapping[str, Any]) -> Any:
        q01 = self._stats_like(stats["q01"], value)
        q99 = self._stats_like(stats["q99"], value)
        dim = q01.shape[-1]
        normalized = (value[..., :dim] + 1.0) / 2.0 if self.quantile_to_neg_one_one else value[..., :dim]
        if dim < value.shape[-1]:
            restored = normalized * (q99 - q01 + self.eps) + q01
            return self._concat_last_dim(restored, value[..., dim:])
        q01 = q01[..., : value.shape[-1]]
        q99 = q99[..., : value.shape[-1]]
        normalized = (value + 1.0) / 2.0 if self.quantile_to_neg_one_one else value
        return normalized * (q99 - q01 + self.eps) + q01

    def _pad_last_dim(self, stats: Any, dim: int, *, value: float) -> Any:
        if stats.shape[-1] >= dim:
            return stats[..., :dim]
        if isinstance(stats, torch.Tensor):
            return torch.nn.functional.pad(stats, (0, dim - stats.shape[-1]), value=value)
        pad_width = [(0, 0)] * stats.ndim
        pad_width[-1] = (0, dim - stats.shape[-1])
        return np.pad(stats, pad_width, constant_values=value)

    def _concat_last_dim(self, left: Any, right: Any) -> Any:
        if isinstance(left, torch.Tensor):
            return torch.cat([left, right], dim=-1)
        return np.concatenate([left, right], axis=-1)


@dataclass
class AddHorizon(TransformFn):
    history_horizon: int
    future_horizon: int

    def __call__(self, data: DataDict) -> DataDict:
        data["history_horizon"] = self.history_horizon
        data["future_horizon"] = self.future_horizon
        return data


@dataclass
class CustomTransform(TransformFn):
    mode: str
    dataset_dir: str | Path | None = None
    extra: dict | None = None
    

    def __call__(self, data: DataDict) -> DataDict:
        if self.mode == "add_subtaskstructure":
            return self.add_subtaskstructure(data)
        if self.mode == "split_gripper_uvd":
            return self.split_gripper_uvd(data)
        if self.mode in {"build_model_input", "build_final_input"}:
            return self.build_model_input(data)
        if self.mode == "build_model_output":
            return self.build_model_output(data)
        else:
            raise KeyError(f"Not support mode: {self.mode}")

    def split_gripper_uvd(self, data: DataDict) -> DataDict:
        gripper_uvd = data["gripper_uvd"]
        data["gripper_uv"] = gripper_uvd[..., :2]
        data["gripper_d"] = gripper_uvd[..., 2:3]
        return data

    def build_model_output(self, data: DataDict) -> DataDict:
        outputs = data.get("outputs", data)
        if not isinstance(outputs, Mapping):
            raise TypeError("build_model_output expects data or data['outputs'] to be a mapping")

        if "relative_plan" in outputs:
            relative = outputs["relative_plan"].clone()
            ray_scale = float(self._extra_value("ray_scale", 1.0))
            if ray_scale <= 0.0:
                raise ValueError(f"ray_scale must be positive, got {ray_scale}")
            relative[..., :2] /= ray_scale
            relative[..., 2:3] = self._unnormalize_output_field(
                relative[..., 2:3],
                field="depths.depth_rel",
                context=data,
            )
            outputs["relative_plan"] = relative
        if "metric_z_plan" in outputs:
            outputs["metric_z_plan"] = self._unnormalize_output_field(
                outputs["metric_z_plan"].clone(),
                field="gripper_d",
                context=data,
            )
        return data

    def _extra_value(self, key: str, default: Any) -> Any:
        if self.extra is None:
            return default
        return self.extra.get(key, default)

    def _unnormalize_output_field(self, value: Any, *, field: str, context: DataDict) -> Any:
        if self.extra is None or "norm_stats" not in self.extra:
            raise ValueError("build_model_output requires extra['norm_stats']")
        transform = Unnormalize(
            norm_stats=self.extra["norm_stats"],
            use_quantiles=bool(self._extra_value("use_quantiles", True)),
            quantile_to_neg_one_one=bool(self._extra_value("quantile_to_neg_one_one", True)),
        )
        stats = transform._unwrap_stats(transform.norm_stats)
        if stats is None or field not in stats:
            raise KeyError(f"Missing norm stats for output field: {field}")
        field_stats = transform._select_field_stats(stats[field], transform._stats_level(transform.norm_stats), context)
        return (
            transform._normalize_quantile(value, field_stats)
            if transform.use_quantiles
            else transform._normalize(value, field_stats)
        )

    def build_model_input(self, data: DataDict) -> DataDict:
        height = 256
        width = 256
        node_points_track = data["node_points_track"]
        depth_rel = data["depths.depth_rel"]
        gripper_uv = data["gripper_uv"]
        gripper_d = data["gripper_d"]
        if gripper_uv.ndim != 3 or gripper_uv.shape[1:] != (6, 2):
            raise ValueError(f"Expected gripper_uv [T, 6, 2], got {tuple(gripper_uv.shape)}")
        if gripper_d.shape != gripper_uv.shape[:2]:
            raise ValueError(f"Expected flattened gripper_d [T, 6], got {tuple(gripper_d.shape)}")
        gripper_openness = torch.as_tensor(data["gripper_openness"], device=gripper_uv.device, dtype=gripper_uv.dtype)
        history_horizon = int(data["history_horizon"])
        future_horizon = int(data["future_horizon"])
        num_frames = gripper_uv.shape[0]
        gripper_openness = gripper_openness.reshape(num_frames, 1)
        intrinsic = self._camera_intrinsic(data, device=gripper_uv.device, dtype=gripper_uv.dtype)
        ray_scale = float(self._extra_value("ray_scale", 1.0))
        if ray_scale <= 0.0:
            raise ValueError(f"ray_scale must be positive, got {ray_scale}")

        node_points_mask = None
        if "node_points_mask" in data:
            node_points_mask = data["node_points_mask"].to(dtype=torch.bool, device=node_points_track.device)
            if node_points_mask.ndim == 2:
                node_points_mask = node_points_mask[history_horizon]

        if node_points_mask is not None:
            selected_node_indices = torch.nonzero(
                node_points_mask[:node_points_track.shape[1]], as_tuple=False
            ).flatten()
        else:
            object_start, object_end = data.get("subtask_object_slice", (0, node_points_track.shape[1]))
            selected_node_indices = torch.arange(
                int(object_start),
                min(int(object_end), node_points_track.shape[1]),
                device=node_points_track.device,
            )

        object_roles = self._object_roles(data["subtaskstructure"], None)
        node_xyv = self._select_object_slots(
            node_points_track,
            selected_node_indices,
            object_roles=object_roles,
        )
        node_uv = node_xyv[..., :2]
        node_depth, _ = self._sample_flat_depth(depth_rel, node_uv, height=height, width=width)
        object_ray = uv_to_normalized_ray_torch(node_uv, intrinsic) * ray_scale
        object_points = torch.cat([object_ray, node_depth], dim=-1)

        actor_depth, _ = self._sample_flat_depth(depth_rel, gripper_uv, height=height, width=width)
        actor_depth[:, 5] = actor_depth[:, 3:5].mean(dim=1)
        actor_indices = torch.as_tensor(ACTOR_POINT_INDICES, device=gripper_uv.device)
        actor_ray = (
            uv_to_normalized_ray_torch(gripper_uv, intrinsic) * ray_scale
        ).index_select(1, actor_indices)
        actor_relative = torch.cat([actor_ray, actor_depth.index_select(1, actor_indices)], dim=-1)
        actor_metric_z = gripper_d.index_select(1, actor_indices).unsqueeze(-1)

        points_per_entity = object_points.shape[2]
        entity_points = object_points.new_zeros((num_frames, 3, points_per_entity, POINT_FEATURE_DIM))
        entity_mask = torch.zeros((num_frames, 3, points_per_entity), dtype=torch.bool, device=object_points.device)
        entity_points[:, 0, :ACTOR_NUM_POINTS] = actor_relative
        entity_mask[:, 0, :ACTOR_NUM_POINTS] = True
        entity_points[:, 1:3] = object_points
        for role_index, role in enumerate(object_roles[:selected_node_indices.numel()]):
            slot_index = {"patient": 0, "target": 1}.get(role, role_index)
            if slot_index < 2:
                entity_mask[:, slot_index + 1] = True

        input_horizon = history_horizon + 1
        frame_offsets = torch.arange(
            -history_horizon,
            future_horizon + 1,
            dtype=torch.long,
            device=entity_points.device,
        )
        if frame_offsets.numel() != entity_points.shape[0]:
            raise ValueError(
                f"Expected {frame_offsets.numel()} frames from horizon, got {entity_points.shape[0]}"
            )
        scene_condition = self._build_scene_condition(
            data["subtaskstructure"], device=entity_points.device,
        )
        entity_role_condition = self._build_entity_role_condition(device=entity_points.device)
        future_relative = actor_relative[input_horizon:]
        future_metric_z = actor_metric_z[input_horizon:]
        if self._extra_value("use_delta", True):
            relative_plan = future_relative - actor_relative[history_horizon][None]
            metric_z_plan = future_metric_z - actor_metric_z[history_horizon][None]
        else:
            relative_plan = future_relative
            metric_z_plan = future_metric_z
        normalized_closedness = 1.0 - 2.0 * gripper_openness
        action = torch.as_tensor(data["action"], device=entity_points.device, dtype=entity_points.dtype)
        gripper_action = 1.0 - 2.0 * action[..., -1:].clamp(0.0, 1.0)
        result = {
            "entity_points": entity_points[:input_horizon],
            "entity_point_mask": entity_mask[:input_horizon],
            "scene_condition": scene_condition,
            "entity_role_condition": entity_role_condition,
            "actor_metric_history": actor_metric_z[:input_horizon],
            "gripper_closedness_history": normalized_closedness[:input_horizon],
            "robot_metric_mask": torch.ones(1, dtype=torch.bool, device=entity_points.device),
            "target": {
                "relative_plan": relative_plan,
                "metric_z_plan": metric_z_plan,
                "gripper_action_plan": gripper_action[
                    history_horizon:history_horizon + future_horizon
                ],
                "is_complete": torch.as_tensor(
                    data["is_complete"], device=entity_points.device, dtype=entity_points.dtype
                )[history_horizon].reshape(1),
            },
        }
        for key in ("images", "state", "metadata"):
            if key in data:
                result[key] = data[key]
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("images."):
                result[key] = value
        return result

    def _camera_intrinsic(self, data: DataDict, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.dataset_dir is None:
            raise ValueError("dataset_dir is required to construct normalized camera rays")
        cache = getattr(self, "_camera_intrinsic_cache", None)
        if cache is None:
            rows = json.loads((Path(self.dataset_dir) / "meta" / "cameras.json").read_text())
            cache = {
                int(row["task_index"]): row["cameras"]["agentview"]["intrinsic"]
                for row in rows
            }
            self._camera_intrinsic_cache = cache
        task_index = self._to_int(data["metadata"]["task_index"])
        return torch.as_tensor(cache[task_index], device=device, dtype=dtype)

    def _select_object_slots(
        self,
        node_points_track: torch.Tensor,
        selected_node_indices: torch.Tensor,
        *,
        object_roles: Sequence[str],
    ) -> torch.Tensor:
        object_slots = node_points_track.new_zeros(
            (node_points_track.shape[0], 2, node_points_track.shape[2], node_points_track.shape[3])
        )
        for role_index, role in enumerate(object_roles[:selected_node_indices.numel()]):
            if role == "patient":
                slot_index = 0
            elif role == "target":
                slot_index = 1
            else:
                slot_index = role_index
            if slot_index >= object_slots.shape[1]:
                continue
            object_slots[:, slot_index] = node_points_track[:, selected_node_indices[role_index]]
        return object_slots

    def _build_entity_role_condition(self, *, device: torch.device) -> torch.Tensor:
        return torch.stack([self._embed_text(role, device=device) for role in ENTITY_ROLES])

    def _build_scene_condition(
        self,
        subtaskstructure: Mapping[str, Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        action_type = str(subtaskstructure.get("action_type", ""))
        action_degree = subtaskstructure.get("action_degree")
        return torch.stack([
            self._embed_text(action_type, device=device),
            self._embed_text(None if action_degree is None else str(action_degree), device=device),
        ])

    def _embed_text(self, text: str | None, *, device: torch.device) -> torch.Tensor:
        self._ensure_bge()
        assert self.extra is not None
        dim = int(self.extra["bge_dim"])
        if text is None or text == "":
            return torch.zeros(dim, dtype=torch.float32, device=device)

        cache = self.extra["embedding_cache"]
        if text not in cache:
            tokenizer = self.extra["bge_tokenizer"]
            model = self.extra["bge_model"]
            batch = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**batch)
                embedding = outputs.last_hidden_state[:, 0]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)[0]
            cache[text] = embedding.detach().cpu().float()
        return cache[text].to(device=device)

    def _ensure_bge(self) -> None:
        if self.extra is not None and "bge_model" in self.extra:
            return
        from transformers import AutoModel, AutoTokenizer

        model_path = Path("/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        if self.extra is None:
            self.extra = {}
        self.extra.update({
            "bge_tokenizer": tokenizer,
            "bge_model": model,
            "embedding_cache": {},
            "bge_dim": int(model.config.hidden_size),
        })

    def _object_roles(self, subtaskstructure: Mapping[str, Any], expected_count: int | None) -> list[str]:
        roles = []
        for node in subtaskstructure.get("nodes", []):
            role = str(node.get("role", ""))
            if role != "actor":
                roles.append(role)
        if expected_count is not None:
            if len(roles) < expected_count:
                fallback = ["patient", "target"]
                roles.extend(fallback[len(roles):expected_count])
            roles = roles[:expected_count]
        return roles

    def _sample_flat_depth(self, depth: torch.Tensor, uv: torch.Tensor, *, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.round(uv[..., 0]).long()
        v = torch.round(uv[..., 1]).long()
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        flat_index = (v.clamp(0, height - 1) * width + u.clamp(0, width - 1)).long()
        batch_index = torch.arange(depth.shape[0], device=depth.device)
        for _ in range(flat_index.ndim - 1):
            batch_index = batch_index.unsqueeze(-1)
        sampled = depth[batch_index.expand_as(flat_index), flat_index]
        sampled = sampled.masked_fill(~in_bounds, 0.0)
        return sampled.unsqueeze(-1), in_bounds.unsqueeze(-1)

    def add_subtaskstructure(self, data):
        if self.dataset_dir is None:
            raise ValueError("dataset_dir is required for add_subtaskstructure")
        if self.extra is None:
            extra_file = Path(self.dataset_dir) / "meta" / "taskstructures.jsonl"
            with extra_file.open("r", encoding="utf-8") as f:
                if extra_file.suffix == ".jsonl":
                    rows = [json.loads(line) for line in f if line.strip()]
                elif extra_file.suffix == ".json":
                    rows = json.load(f)
                else:
                    raise ValueError(f"Unsupported extra file extension: {extra_file.suffix}")

            self.extra = {}
            for row in rows:
                task = str(row["task"])
                self.extra[task] = list(row.get("subtasks", []))

        task = self._to_str(data.get("prompt", data.get("task")))
        subtask_id_value = data.get("subtask_id")
        if subtask_id_value is None and isinstance(data.get("metadata"), Mapping):
            subtask_id_value = data["metadata"].get("subtask_id")
        if subtask_id_value is None:
            raise KeyError("subtask_id")

        subtask_ids = torch.as_tensor(subtask_id_value).reshape(-1)
        current_index = min(int(data.get("history_horizon", 0)), subtask_ids.numel() - 1)
        subtask_index = self._to_int(subtask_ids[current_index]) - 1
        subtasks = self.extra[task]
        data["subtaskstructure"] = subtasks[subtask_index]

        object_start = sum(len(self._object_roles(subtask, None)) for subtask in subtasks[:subtask_index])
        object_end = object_start + len(self._object_roles(data["subtaskstructure"], None))
        data["subtask_object_slice"] = (object_start, object_end)
        return data

    def _to_str(self, value: Any) -> str:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.shape == ():
            return str(array.item())
        if array.size == 1:
            return str(array.reshape(-1)[0].item() if hasattr(array.reshape(-1)[0], "item") else array.reshape(-1)[0])
        return str(value)

    def _to_int(self, value: Any) -> int:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).reshape(-1)
        if array.size == 0:
            raise ValueError("Cannot read index from an empty value")
        return int(array[0].item() if hasattr(array[0], "item") else array[0])
        
