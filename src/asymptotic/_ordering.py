"""Shared deterministic ordering helpers for sparse asymptotic exponents."""

from __future__ import annotations

import sympy as sp


def exponent_sort_key(exponent: sp.Expr) -> tuple[object, ...]:
    """Order numeric exponents by value and symbolic exponents canonically."""

    exponent = sp.sympify(exponent)
    if exponent.is_number:
        try:
            return (0, float(exponent))
        except (TypeError, ValueError):
            pass
    return (1, sp.default_sort_key(exponent))
