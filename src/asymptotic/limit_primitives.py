"""Dependency-light normalization primitives for limit solvers."""

from __future__ import annotations

from collections.abc import Mapping

import sympy as sp


def _normalize_variables(variables) -> tuple[sp.Symbol, ...]:
    if isinstance(variables, sp.Symbol):
        result = (variables,)
    else:
        try:
            result = tuple(sp.sympify(v) for v in variables)
        except TypeError as exc:
            raise TypeError(
                "variables must be a Symbol or a sequence of Symbols"
            ) from exc
    if not result or any(not isinstance(v, sp.Symbol) for v in result):
        raise TypeError("variables must contain one or more SymPy Symbols")
    if len(set(result)) != len(result):
        raise ValueError("variables must be distinct")
    return result


def normalize_limit_target(variables, target) -> tuple[sp.Expr, ...]:
    """Normalize scalar, sequence, or mapping limit targets to variable order."""
    variables = _normalize_variables(variables)
    if isinstance(target, Mapping):
        missing = [v for v in variables if v not in target]
        extra = [v for v in target if v not in variables]
        if missing or extra:
            raise ValueError("target mapping must contain exactly the limit variables")
        values = tuple(sp.sympify(target[v]) for v in variables)
    elif len(variables) == 1 and not isinstance(target, (tuple, list, sp.Tuple)):
        values = (sp.sympify(target),)
    else:
        if isinstance(target, (str, bytes)):
            raise TypeError("target must be a scalar, sequence, or variable mapping")
        try:
            values = tuple(sp.sympify(v) for v in target)
        except TypeError as exc:
            raise TypeError(
                "target must be a scalar, sequence, or variable mapping"
            ) from exc
        if len(values) != len(variables):
            raise ValueError("target length must match variables")
    if any(v in (sp.oo, -sp.oo, sp.zoo, sp.nan) for v in values):
        raise NotImplementedError(
            "simultaneous limits currently support finite real target points only"
        )
    return values
