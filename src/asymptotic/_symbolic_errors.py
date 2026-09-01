"""Recoverable exceptions from bounded symbolic backend operations."""

from __future__ import annotations

import sympy as sp

SYMBOLIC_ERRORS = (
    ArithmeticError,
    NotImplementedError,
    RecursionError,
    TypeError,
    ValueError,
    sp.PoleError,
    sp.PolynomialError,
    sp.SympifyError,
)
