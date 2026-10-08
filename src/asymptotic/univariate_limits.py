"""Certified helpers for univariate limit and cluster-set analysis."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_limit
from .oscillatory_normal_form import oscillatory_cluster_interval


@dataclass(frozen=True)
class ExtremalLimit:
    lower: sp.Expr
    upper: sp.Expr
    cluster_set: sp.Set
    provider: str = "univariate_extremal_limit"


def extremal_limit(expr, variable, *, point=sp.oo):
    """Return exact liminf/limsup when an exact periodic cluster image is known."""
    if point not in (sp.oo, -sp.oo):
        return None
    r = oscillatory_cluster_interval(expr, variable)
    if r is None:
        return None
    if isinstance(r.set, sp.Interval):
        return ExtremalLimit(r.set.inf, r.set.sup, r.set)
    # Keep the exact cluster image when symbolic extrema are unavailable.
    return ExtremalLimit(r.lower, r.upper, r.set)


def complex_direction_limit(expr, variable, *, point=0):
    """Certified complex limit from polar/ray disagreement or SymPy's exact result."""
    z = sp.sympify(variable)
    expr = sp.sympify(expr)
    point = sp.sympify(point)
    t = sp.Dummy("complex_t", positive=True)
    vals = []
    for phase in (1, sp.I, -1, -sp.I):
        try:
            vals.append(
                bounded_limit(
                    expr.xreplace({z: point + phase * t}),
                    t,
                    0,
                    direction="+",
                    allow_general=True,
                )
            )
        except (TypeError, ValueError, NotImplementedError):
            return None
    resolved = [v for v in vals if not isinstance(v, sp.Limit)]
    if len(resolved) == 4 and any(
        sp.simplify(v - resolved[0]) != 0 for v in resolved[1:]
    ):
        return sp.S.NaN
    try:
        v = bounded_limit(expr, z, point, allow_general=True)
    except (TypeError, ValueError, NotImplementedError):
        return None
    return None if isinstance(v, sp.Limit) else v


def periodic_composition_cluster(outer, phase, variable):
    """Exact cluster image when a continuous periodic outer function is driven through infinity."""
    x = sp.sympify(variable)
    h = sp.sympify(phase)
    try:
        hv = bounded_limit(h, x, sp.oo, allow_general=True)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if hv not in (sp.oo, -sp.oo):
        return None
    t = sp.Dummy("periodic_t", real=True)
    e = sp.sympify(outer(t))
    return oscillatory_cluster_interval(e, t)
