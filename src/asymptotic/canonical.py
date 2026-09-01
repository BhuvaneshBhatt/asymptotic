"""Stable canonicalization helpers for symbolic asymptotic certificates.

The helpers deliberately avoid relying on incidental ``args`` order in unevaluated
SymPy objects.  They are conservative: ``canonical_equal`` only returns ``True``
when structural normalization or an exact algebraic simplification proves equality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass

import sympy as sp

from ._symbolic_errors import SYMBOLIC_ERRORS


def _canonical_expr(expr: sp.Basic) -> sp.Basic:
    if not expr.args:
        return expr
    args = tuple(_canonical_expr(arg) if isinstance(arg, sp.Basic) else arg for arg in expr.args)
    if expr.is_Add:
        return sp.Add(*sorted(args, key=sp.default_sort_key), evaluate=False)
    if expr.is_Mul and expr.is_commutative:
        return sp.Mul(*sorted(args, key=sp.default_sort_key), evaluate=False)
    if isinstance(expr, sp.Equality):
        return sp.Eq(*args, evaluate=False)
    if isinstance(expr, sp.Unequality):
        return sp.Ne(*args, evaluate=False)
    if expr.func in (sp.And, sp.Or):
        return expr.func(*sorted(args, key=sp.default_sort_key), evaluate=False)
    try:
        return expr.func(*args, evaluate=False)
    except (TypeError, ValueError):
        try:
            return expr.func(*args)
        except SYMBOLIC_ERRORS:
            return expr


def canonical_expr(expr: sp.Expr | bool) -> sp.Expr:
    """Return a stable expression tree independent of commutative argument order."""
    return sp.sympify(_canonical_expr(sp.sympify(expr)))


def canonical_key(value: object) -> object:
    """Hashable stable key for SymPy-rich dataclasses, tuples, mappings, and scalars."""
    if isinstance(value, (sp.Equality, sp.Unequality)):
        left, right = value.args
        sides = tuple(sorted((canonical_key(left), canonical_key(right)), key=repr))
        return ("sympy_rel", value.func.__name__, sides)
    if isinstance(value, sp.Basic):
        return ("sympy", sp.srepr(canonical_expr(value)))
    if is_dataclass(value):
        return (
            value.__class__.__module__,
            value.__class__.__qualname__,
            tuple((f.name, canonical_key(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, Mapping):
        items = ((canonical_key(k), canonical_key(v)) for k, v in value.items())
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, tuple):
        return ("tuple", tuple(canonical_key(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(canonical_key(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((canonical_key(v) for v in value), key=repr)))
    return value


def canonical_equal(left: object, right: object) -> bool:
    """Conservatively prove equality without depending on SymPy argument ordering."""
    if canonical_key(left) == canonical_key(right):
        return True
    if isinstance(left, sp.Basic) and isinstance(right, sp.Basic):
        try:
            difference = sp.cancel(sp.together(sp.sympify(left) - sp.sympify(right)))
            if difference == 0 or difference.is_zero is True:
                return True
        except SYMBOLIC_ERRORS:
            # Algebraic normalization is opportunistic; canonical-key equality
            # and the subsequent simplify fallback remain available.
            pass
        try:
            return sp.simplify(sp.sympify(left) - sp.sympify(right)) == 0
        except SYMBOLIC_ERRORS:
            return False
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(canonical_equal(a, b) for a, b in zip(left, right))
    return left == right
