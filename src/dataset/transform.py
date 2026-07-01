from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
    transforms: Sequence[Transform]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RepackTransform(TransformFn):
    structure: Mapping[str, Any]
    optional_structure: Mapping[str, Any] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        result = self._repack_required(self.structure, data)
        if self.optional_structure:
            optional = self._repack_optional(self.optional_structure, data)
            result.update(optional)
        return result

    def _repack_required(self, structure: Mapping[str, Any], data: DataDict) -> DataDict:
        result: DataDict = {}
        for key, value in structure.items():
            if isinstance(value, str):
                result[key] = data[value]
            elif isinstance(value, Mapping):
                result[key] = self._repack_required(value, data)
            else:
                raise TypeError(f"Unsupported repack spec for key={key!r}: {value!r}")
        return result

    def _repack_optional(self, structure: Mapping[str, Any], data: DataDict) -> DataDict:
        result: DataDict = {}
        for key, value in structure.items():
            if isinstance(value, str):
                if value in data:
                    result[key] = data[value]
            elif isinstance(value, Mapping):
                nested = self._repack_optional(value, data)
                if nested:
                    result[key] = nested
            else:
                raise TypeError(f"Unsupported optional repack spec for key={key!r}: {value!r}")
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
    use_quantiles: bool = False
    eps: float = 1e-6

    def __call__(self, data: DataDict) -> DataDict:
        stats = self._unwrap_stats(self.norm_stats)
        if stats is None:
            return data

        for key, field_stats in stats.items():
            if key not in data:
                raise KeyError(f"Cannot normalize missing field: {key}")
            value = data[key]
            value = self._normalize_quantile(value, field_stats) if self.use_quantiles else self._normalize(value, field_stats)
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
        return (value - q01) / (q99 - q01 + self.eps) * 2.0 - 1.0

    def _unwrap_stats(self, norm_stats: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if norm_stats is None:
            return None
        return norm_stats.get("norm_stats", norm_stats)

    def _stats_like(self, stats: Any, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return torch.as_tensor(stats, dtype=value.dtype, device=value.device)
        if isinstance(value, np.ndarray):
            return np.asarray(stats, dtype=value.dtype)
        raise TypeError(f"Cannot normalize value of type {type(value)!r}")

    def _match_last_dim(self, stats: Any, dim: int) -> Any:
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
        if dim < value.shape[-1]:
            restored = (value[..., :dim] + 1.0) / 2.0 * (q99 - q01 + self.eps) + q01
            return self._concat_last_dim(restored, value[..., dim:])
        q01 = q01[..., : value.shape[-1]]
        q99 = q99[..., : value.shape[-1]]
        return (value + 1.0) / 2.0 * (q99 - q01 + self.eps) + q01

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
