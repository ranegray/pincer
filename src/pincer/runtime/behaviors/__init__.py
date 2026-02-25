"""Behavior registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pincer.runtime.behaviors.base import Behavior

_REGISTRY: dict[str, type[Behavior]] = {}


def _load_registry() -> dict[str, type[Behavior]]:
    if not _REGISTRY:
        from pincer.runtime.behaviors.detect_and_grasp import DetectAndGrasp

        _REGISTRY["detect_and_grasp"] = DetectAndGrasp
    return _REGISTRY


def get_behavior(name: str) -> type[Behavior]:
    registry = _load_registry()
    if name not in registry:
        raise ValueError(f"Unknown behavior {name!r}. Available: {', '.join(registry)}")
    return registry[name]
