"""Shared deterministic ordering helpers for sparse asymptotic exponents."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import sympy as sp


@lru_cache(maxsize=512)
def exponent_sort_key(exponent: sp.Expr) -> tuple[object, ...]:
    """Order numeric exponents by value and symbolic exponents canonically."""

    exponent = sp.sympify(exponent)
    if exponent.is_Rational:
        value = (
            int(exponent.p)
            if exponent.q == 1
            else Fraction(int(exponent.p), int(exponent.q))
        )
        return (0, value)
    if exponent.is_Float:
        value = sp.Rational(exponent)
        return (0, Fraction(int(value.p), int(value.q)))
    if exponent.is_number and exponent.is_extended_real is True:
        # Real symbolic numbers retain exact comparisons; float keys can
        # collapse distinct nearby exponents and incorrectly combine terms.
        return (0, exponent)
    return (1, sp.default_sort_key(exponent))
