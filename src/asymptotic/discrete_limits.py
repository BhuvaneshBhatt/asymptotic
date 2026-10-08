"""Limits of integer-indexed sequences with certified discrete reductions."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask
from .limits import limit


@dataclass(frozen=True)
class DiscreteLimitResult:
    """Certified result of an integer-sequence limit proof."""

    value: sp.Expr | None
    method: str
    certified: bool


def _finite_difference(expr: sp.Expr, n: sp.Symbol) -> sp.Expr:
    return sp.cancel(expr.xreplace({n: n + 1}) - expr)


def _stolz_quotient(expr: sp.Expr, n: sp.Symbol):
    num, den = sp.fraction(sp.cancel(expr))
    if den == 1 or not num.has(n) or not den.has(n):
        return None
    dnum = _finite_difference(num, n)
    dden = _finite_difference(den, n)
    if dden == 0:
        return None
    return sp.cancel(dnum / dden), num, den


def discrete_limit(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = sp.oo,
    *,
    assumptions=sp.S.true,
    return_result: bool = False,
):
    """Compute a certified limit along integer values of ``variable``.

    The direct route is used when the ordinary real limit is proved.  For
    quotients at positive infinity, a Stolz--Cesaro reduction is attempted
    when the denominator is eventually increasing and unbounded.  Failure to
    prove those hypotheses leaves the result unknown.
    """
    expr = sp.sympify(expr)
    variable = sp.sympify(variable)
    point = sp.sympify(point)
    direct = limit(expr, variable, point, return_result=True, assumptions=assumptions)
    if direct.status.name in {"EXISTS", "PROVED"}:
        out = DiscreteLimitResult(direct.value, "continuous-restriction", True)
        return out if return_result else out.value
    if point != sp.oo:
        out = DiscreteLimitResult(None, "unknown", False)
        return out if return_result else None

    reduced = _stolz_quotient(expr, variable)
    if reduced is None:
        out = DiscreteLimitResult(None, "unknown", False)
        return out if return_result else None
    quotient, _, den = reduced

    delta_den = _finite_difference(den, variable)
    positive = bounded_ask(
        sp.Q.positive(delta_den),
        sp.Q.integer(variable) & sp.Q.positive(variable) & sp.sympify(assumptions),
    )
    den_limit = limit(den, variable, sp.oo, return_result=True, assumptions=assumptions)
    if positive is not True or den_limit.value != sp.oo:
        out = DiscreteLimitResult(None, "unknown", False)
        return out if return_result else None

    qlimit = limit(
        quotient, variable, sp.oo, return_result=True, assumptions=assumptions
    )
    if qlimit.status.name not in {"EXISTS", "PROVED"}:
        out = DiscreteLimitResult(None, "unknown", False)
        return out if return_result else None
    out = DiscreteLimitResult(qlimit.value, "stolz-cesaro", True)
    return out if return_result else out.value
