"""Construct policies from experiment configurations."""

from __future__ import annotations

from typing import Any

from torch import nn

SUPPORTED_POLICIES = ("graphpoint", "act")


def resolve_policy_name(model_config: Any) -> str:
    """Resolve the policy name, treating configs without one as legacy GraphPoint."""
    policy_name = getattr(model_config, "policy_name", None)
    return "graphpoint" if policy_name is None else str(policy_name).lower()


def build_policy(model_config: Any) -> nn.Module:
    policy_name = resolve_policy_name(model_config)
    if policy_name == "graphpoint":
        from src.policy.graphpoint.model import GraphFlowModel

        kwargs = model_config.to_kwargs()
        kwargs.pop("include_future_object_point", None)
        return GraphFlowModel(**kwargs)
    if policy_name == "act":
        from src.policy.act.model import ACTPolicy

        return ACTPolicy(**model_config.to_kwargs())
    raise ValueError(f"Unsupported policy: {policy_name!r}")
