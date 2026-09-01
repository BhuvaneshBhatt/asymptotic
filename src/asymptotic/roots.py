"""Asymptotic root finding."""

from __future__ import annotations

import sympy as sp

from .solve import AsymptoticSolveResult, asymptotic_solve


def asymptotic_root(
    expression: sp.Expr,
    variable: sp.Symbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    domain: sp.Set = sp.S.Complexes,
    assumptions: sp.Expr | bool = sp.S.true,
    limit: sp.Expr | None = None,
    branch: int | None = None,
) -> AsymptoticSolveResult | sp.Expr:
    """Find roots of ``expression == 0`` asymptotically in ``parameter``.

    With ``branch=None`` the complete :class:`AsymptoticSolveResult` is
    returned.  Otherwise the requested branch expression is returned directly.
    ``limit`` may select roots tending to a prescribed value.
    """
    limits = None if limit is None else {variable: sp.sympify(limit)}
    result = asymptotic_solve(
        expression,
        variable,
        parameter=parameter,
        point=point,
        terms=terms,
        domain=domain,
        assumptions=assumptions,
        limits=limits,
    )
    if branch is None:
        return result
    if not isinstance(branch, int):
        raise TypeError("branch must be an integer or None")
    try:
        selected = result.branches[branch]
    except IndexError as exc:
        raise IndexError("asymptotic root branch index out of range") from exc
    root = selected.as_dict().get(variable)
    if root is None:
        raise ValueError("selected solution branch does not determine the requested variable")
    return root
