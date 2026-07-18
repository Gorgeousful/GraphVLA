from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import json
import numpy as np
import torch
from rich.console import Console
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
class FlipTransform(TransformFn):
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in {"horizontal", "vertical", "all"}:
            raise ValueError(f"Unsupported flip mode: {self.mode}")

    def __call__(self, data: DataDict) -> DataDict:
        for key in ("images", "depths"):
            if key in data:
                data[key] = self._flip_mapping(data[key])
        return data

    def _flip_mapping(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return {key: self._flip_value(value) for key, value in values.items()}

    def _flip_value(self, value: Any) -> Any:
        dims = self._flip_dims(value)
        if hasattr(value, "flip"):
            return value.flip(dims)
        if hasattr(value, "__getitem__"):
            return self._flip_sequence(value)
        raise TypeError(f"Cannot flip value of type {type(value)!r}")

    def _flip_dims(self, value: Any) -> tuple[int, ...]:
        ndim = getattr(value, "ndim", None)
        if ndim is None:
            ndim = len(getattr(value, "shape"))
        if ndim < 2:
            raise ValueError(f"Cannot flip value with ndim={ndim}")

        horizontal_dim = ndim - 1
        vertical_dim = ndim - 2
        if self.mode == "horizontal":
            return (horizontal_dim,)
        if self.mode == "vertical":
            return (vertical_dim,)
        return (vertical_dim, horizontal_dim)

    def _flip_sequence(self, value: Any) -> Any:
        if self.mode == "horizontal":
            return value[..., ::-1]
        if self.mode == "vertical":
            return value[..., ::-1, :]
        return value[..., ::-1, ::-1]


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
        if self.mode == "random_object_permutation":
            return self.random_object_permutation(data)
        if self.mode == "build_model_output":
            return self.build_model_output(data)
        else:
            raise KeyError(f"Not support mode: {self.mode}")

    def split_gripper_uvd(self, data: DataDict) -> DataDict:
        gripper_uvd = data["gripper_uvd"]
        data["gripper_uv"] = gripper_uvd[..., :2]
        data["gripper_d"] = gripper_uvd[..., 2:3]
        return data

    def random_object_permutation(self, data: DataDict) -> DataDict:
        point_feats = data["point_feats"]
        target = data["target"]
        object_id = data["object_id"]

        if point_feats.ndim != 4:
            raise ValueError(f"Expected point_feats [T, N, P, F], got {tuple(point_feats.shape)}")

        object_count = point_feats.shape[1]
        points_per_object = point_feats.shape[2]
        permuted_feats = point_feats.clone()
        target_fields = tuple(
            key
            for key in ("point", "metric_depth", "metric_depth_mask", "point_mask")
            if key in target
        )
        permuted_target = {**target, **{key: target[key].clone() for key in target_fields}}

        for object_index in range(object_count):
            permutation = torch.randperm(points_per_object, device=point_feats.device)
            permuted_feats[:, object_index] = point_feats[:, object_index].index_select(1, permutation)

            query_mask = object_id == object_index + 1
            for key in target_fields:
                selected = target[key][query_mask]
                if selected.shape[0] % points_per_object != 0:
                    raise ValueError(
                        f"Object {object_index + 1} has {selected.shape[0]} {key} rows; "
                        f"expected a multiple of {points_per_object}"
                    )
                grouped = selected.reshape(-1, points_per_object, *selected.shape[1:])
                permuted_target[key][query_mask] = (
                    grouped.index_select(1, permutation.to(selected.device)).reshape_as(selected)
                )

        return {**data, "point_feats": permuted_feats, "target": permuted_target}

    def build_model_output(self, data: DataDict) -> DataDict:
        height = int(self._extra_value("height", 256))
        width = int(self._extra_value("width", 256))
        outputs = data.get("outputs", data)
        if not isinstance(outputs, Mapping):
            raise TypeError("build_model_output expects data or data['outputs'] to be a mapping")

        if "point" in outputs:
            point = outputs["point"].clone()
            point[..., 0] = (point[..., 0] + 1.0) * (width / 2.0)
            point[..., 1] = (point[..., 1] + 1.0) * (height / 2.0)
            point[..., 2:3] = self._unnormalize_output_field(
                point[..., 2:3],
                field="depths.depth_rel",
                context=data,
            )
            point[..., 3] = torch.sigmoid(point[..., 3])

            outputs["point"] = point

        if "metric_depth" in outputs:
            outputs["metric_depth"] = self._unnormalize_output_field(
                outputs["metric_depth"].clone(),
                field="gripper_d",
                context=data,
            )

        if self._extra_value("sigmoid_is_complete", True) and "is_complete" in outputs:
            outputs["is_complete"] = torch.sigmoid(outputs["is_complete"])
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
        is_complete = data["is_complete"]
        history_horizon = int(data["history_horizon"])
        future_horizon = int(data["future_horizon"])

        node_points_mask = None
        if "node_points_mask" in data:
            node_points_mask = data["node_points_mask"].to(dtype=torch.bool, device=node_points_track.device)
            if node_points_mask.ndim == 2:
                node_points_mask = node_points_mask[history_horizon]

        node_valid = node_points_track.abs().sum(dim=(0, 2, 3)) > 0
        if node_points_mask is not None:
            node_valid = node_valid & node_points_mask[:node_valid.shape[0]]
            selected_node_indices = torch.nonzero(node_valid, as_tuple=False).flatten()
        else:
            object_start, object_end = data.get("subtask_object_slice", (0, node_points_track.shape[1]))
            selected_node_indices = torch.arange(
                int(object_start),
                min(int(object_end), node_points_track.shape[1]),
                device=node_points_track.device,
            )
            selected_node_indices = selected_node_indices[node_valid[selected_node_indices]]

        node_xyv = self._select_object_slots(
            node_points_track,
            selected_node_indices,
            object_roles=self._object_roles(data["subtaskstructure"], None),
        )
        object_valid_mask = node_xyv.abs().sum(dim=(0, 2, 3)) > 0
        node_uv = node_xyv[..., :2]
        node_vis = node_xyv[..., 2:3]
        node_depth, node_in_bounds = self._sample_flat_depth(depth_rel, node_uv, height=height, width=width)
        node_uv_norm = self._normalize_uv(node_uv, height=height, width=width)
        node_vis = node_vis * node_in_bounds.to(dtype=node_vis.dtype)
        node_metric = torch.zeros_like(node_depth)
        node_metric_mask = torch.zeros_like(node_depth)
        object_points = torch.cat(
            [node_uv_norm, node_depth, node_vis, node_metric, node_metric_mask],
            dim=-1,
        )
        object_roles = ["patient", "target"]

        actor_uv = gripper_uv
        actor_depth, actor_in_bounds = self._sample_flat_depth(depth_rel, actor_uv, height=height, width=width)
        actor_uv_norm = self._normalize_uv(actor_uv, height=height, width=width)
        actor_vis = actor_in_bounds.to(dtype=actor_depth.dtype)
        actor_metric = gripper_d.unsqueeze(-1) if gripper_d.ndim == 2 else gripper_d
        actor_metric_mask = torch.ones_like(actor_metric)
        actor_points = torch.cat(
            [actor_uv_norm, actor_depth, actor_vis, actor_metric, actor_metric_mask],
            dim=-1,
        )
        actor_points = actor_points.unsqueeze(1)

        input_horizon = history_horizon + 1
        frame_offsets = torch.arange(
            -history_horizon,
            future_horizon + 1,
            dtype=torch.long,
            device=object_points.device,
        )
        if frame_offsets.numel() != object_points.shape[0]:
            raise ValueError(
                f"Expected {frame_offsets.numel()} frames from horizon, got {object_points.shape[0]}"
            )

        target_point, target_point_mask, object_id, point_id, frame_id = self._build_point_targets(
            object_points=object_points,
            actor_points=actor_points,
            frame_offsets=frame_offsets,
            object_valid_mask=object_valid_mask,
            object_roles=object_roles,
        )
        object_condition, actor_condition = self._build_conditions(
            data["subtaskstructure"],
            object_roles=object_roles,
            device=object_points.device,
        )

        result = {
            "point_feats": object_points[:input_horizon],
            "actor_feats": actor_points[:input_horizon],
            "object_condition": object_condition,
            "actor_condition": actor_condition,
            "object_id": object_id,
            "point_id": point_id,
            "frame_id": frame_id,
            "frame_query_frame_id": frame_offsets,
            "target": {
                "point": target_point[..., :4],
                "metric_depth": target_point[..., 4:5],
                "metric_depth_mask": target_point[..., 5] > 0.5,
                "point_mask": target_point_mask,
                "is_complete": is_complete,
            },
        }
        for key in ("images", "state", "metadata"):
            if key in data:
                result[key] = data[key]
        for key, value in data.items():
            if isinstance(key, str) and key.startswith("images."):
                result[key] = value
        return result

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

    def _build_point_targets(
        self,
        *,
        object_points: torch.Tensor,
        actor_points: torch.Tensor,
        frame_offsets: torch.Tensor,
        object_valid_mask: torch.Tensor,
        object_roles: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_chunks = []
        target_masks = []
        object_ids = []
        point_ids = []
        frame_ids = []
        actor_local_ids = torch.arange(actor_points.shape[2], dtype=torch.long, device=object_points.device)
        object_local_ids = torch.arange(object_points.shape[2], dtype=torch.long, device=object_points.device)

        for frame_index, frame_offset in enumerate(frame_offsets):
            actor_frame = actor_points[frame_index, 0]
            target_chunks.append(actor_frame)
            target_masks.append(torch.ones(actor_frame.shape[0], dtype=torch.bool, device=object_points.device))
            object_ids.append(torch.zeros(actor_frame.shape[0], dtype=torch.long, device=object_points.device))
            point_ids.append(actor_local_ids)
            frame_ids.append(torch.full((actor_frame.shape[0],), int(frame_offset.item()), dtype=torch.long, device=object_points.device))

            for node_index, role in enumerate(object_roles):
                object_frame = object_points[frame_index, node_index]
                object_id = 1 if role == "patient" else 2 if role == "target" else node_index + 1
                target_chunks.append(object_frame)
                target_masks.append(torch.full((object_frame.shape[0],), bool(object_valid_mask[node_index].item()), dtype=torch.bool, device=object_points.device))
                object_ids.append(torch.full((object_frame.shape[0],), object_id, dtype=torch.long, device=object_points.device))
                point_ids.append(object_local_ids)
                frame_ids.append(torch.full((object_frame.shape[0],), int(frame_offset.item()), dtype=torch.long, device=object_points.device))

        return (
            torch.cat(target_chunks, dim=0),
            torch.cat(target_masks, dim=0),
            torch.cat(object_ids, dim=0),
            torch.cat(point_ids, dim=0),
            torch.cat(frame_ids, dim=0),
        )

    def _build_conditions(
        self,
        subtaskstructure: Mapping[str, Any],
        *,
        object_roles: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_type = str(subtaskstructure.get("action_type", ""))
        action_degree = subtaskstructure.get("action_degree")
        actor_condition = self._condition_for("actor", action_type, action_degree, device=device).unsqueeze(0)
        object_conditions = [
            self._condition_for(role, action_type, action_degree, device=device)
            for role in object_roles
        ]
        if object_conditions:
            object_condition = torch.stack(object_conditions, dim=0)
        else:
            object_condition = actor_condition.new_zeros((0, actor_condition.shape[-1]))
        return object_condition, actor_condition

    def _condition_for(self, role: str, action_type: str, action_degree: Any, *, device: torch.device) -> torch.Tensor:
        return torch.cat(
            [
                self._embed_text(role, device=device),
                self._embed_text(action_type, device=device),
                self._embed_text(None if action_degree is None else str(action_degree), device=device),
            ],
            dim=0,
        )

    def _embed_text(self, text: str | None, *, device: torch.device) -> torch.Tensor:
        self._ensure_bge()
        assert self.extra is not None
        dim = int(self.extra["bge_dim"])
        if text is None:
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
        self.extra = {
            "bge_tokenizer": tokenizer,
            "bge_model": model,
            "embedding_cache": {},
            "bge_dim": int(model.config.hidden_size),
        }

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

    def _normalize_uv(self, uv: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        uv_norm = uv.clone()
        uv_norm[..., 0] = (uv[..., 0] - width / 2.0) / (width / 2.0)
        uv_norm[..., 1] = (uv[..., 1] - height / 2.0) / (height / 2.0)
        return uv_norm

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
        
