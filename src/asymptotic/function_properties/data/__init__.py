"""Reviewed built-in function-property data."""

from __future__ import annotations

from ..registry import FunctionPropertyRegistry
from . import elementary, special


def register_defaults(registry: FunctionPropertyRegistry) -> None:
    elementary.register(registry)
    special.register(registry)


__all__ = ["register_defaults"]
