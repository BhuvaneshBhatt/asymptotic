"""Registry primitives for mathematical function properties."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import sympy as sp

from .model import FunctionProperties

PropertyBuilder = Callable[[sp.Expr], FunctionProperties]


class FunctionPropertyRegistry:
    """Mutable mapping from expression heads to property builders.

    The class is intentionally small and independent of asymptotic algorithms,
    keeping registration separate from expansion and certification logic.
    """

    def __init__(self) -> None:
        self._builders: dict[object, PropertyBuilder] = {}

    def register(self, function: object, builder: PropertyBuilder, *, replace: bool = True) -> None:
        if not replace and function in self._builders:
            raise KeyError(f"properties already registered for {function!r}")
        self._builders[function] = builder

    def unregister(self, function: object) -> None:
        self._builders.pop(function, None)

    def builder_for(self, function: object) -> PropertyBuilder | None:
        return self._builders.get(function)

    def __contains__(self, function: object) -> bool:
        return function in self._builders

    def __iter__(self) -> Iterator[object]:
        return iter(self._builders)

    def copy(self) -> FunctionPropertyRegistry:
        other = FunctionPropertyRegistry()
        other._builders.update(self._builders)
        return other


DEFAULT_REGISTRY = FunctionPropertyRegistry()
