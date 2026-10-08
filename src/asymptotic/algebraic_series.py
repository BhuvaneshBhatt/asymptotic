"""Local algebraic/implicit branch series built on the package Puiseux engine."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import (
    bounded_ask,
    bounded_limit,
    bounded_solve_one,
    bounded_solve_system,
)
from .implicit import implicit, implicit_singularity_profile


@dataclass(frozen=True)
class AlgebraicSeriesResult:
    equation: sp.Expr
    dependent: sp.Symbol
    variable: sp.Symbol
    expansion: object
    singular: bool | None
    provider: str = "algebraic_series"


def algebraic_series(
    equation,
    dependent,
    variable,
    *,
    point=0,
    dependent_limit=0,
    order=6,
    assumptions=True,
):
    profile = implicit_singularity_profile(
        equation,
        dependent,
        variable,
        point=point,
        dependent_limit=dependent_limit,
        assumptions=assumptions,
        taylor_degree=order,
    )
    try:
        expansion = implicit(
            equation,
            dependent,
            variable,
            point=point,
            dependent_limit=dependent_limit,
            terms=order,
            assumptions=assumptions,
        )
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        expansion = None
    return AlgebraicSeriesResult(
        sp.sympify(equation), dependent, variable, expansion, profile.singular
    )


@dataclass(frozen=True)
class ImplicitMapSeriesResult:
    equations: tuple[sp.Expr, ...]
    dependent: tuple[sp.Symbol, ...]
    variable: sp.Symbol
    series: tuple[sp.Expr, ...]
    order: int
    provider: str = "implicit_map_series"


def implicit_map_series(
    equations,
    dependent,
    variable,
    *,
    point=0,
    dependent_limit=None,
    order=6,
    assumptions=sp.S.true,
):
    """Taylor series for a regular vector implicit map F(x,y)=0.

    Requires a square system and an invertible dependent-variable Jacobian at
    the base point. Coefficients are solved order-by-order exactly.
    """
    eqs = tuple(map(sp.sympify, equations))
    ys = tuple(dependent)
    x = sp.sympify(variable)
    if len(eqs) != len(ys) or not ys:
        return None
    y0 = (
        tuple(sp.S.Zero for _ in ys)
        if dependent_limit is None
        else tuple(map(sp.sympify, dependent_limit))
    )
    subs0 = {x: sp.sympify(point), **dict(zip(ys, y0))}
    if any(sp.simplify(f.subs(subs0)) != 0 for f in eqs):
        return None
    jac = sp.Matrix(eqs).jacobian(ys).subs(subs0)
    det = sp.simplify(jac.det())
    assumptions = sp.sympify(assumptions)
    nonzero = (
        det.is_nonzero is True or bounded_ask(sp.Q.nonzero(det), assumptions) is True
    )
    if not nonzero:
        return None
    z = sp.Dummy("implicit_z")
    coeff = [
        [sp.Symbol(f"_c_{j}_{k}") for k in range(1, order + 1)] for j in range(len(ys))
    ]
    ser = [
        y0[j] + sum(coeff[j][k - 1] * z**k for k in range(1, order + 1))
        for j in range(len(ys))
    ]
    solved = {}
    for k in range(1, order + 1):
        es = []
        for f in eqs:
            local = sp.expand(
                f.subs({x: sp.sympify(point) + z, **dict(zip(ys, ser))}).subs(solved)
            )
            es.append(sp.expand(local).coeff(z, k))
        unk = [coeff[j][k - 1] for j in range(len(ys))]
        sol = bounded_solve_system(es, unk, allow_general=True)
        if len(sol) != 1:
            return None
        solved.update(sol[0])
    return ImplicitMapSeriesResult(
        eqs,
        ys,
        x,
        tuple(sp.expand(s.subs(solved)).subs(z, x - sp.sympify(point)) for s in ser),
        order,
    )


def algebraic_series_branches(
    equation, dependent, variable, *, point=0, dependent_limit=0, order=6
):
    """Return all explicitly solvable local algebraic branches through a point.

    This is exact and conservative: only branches SymPy solves
    algebraically and whose value tends to the requested dependent limit are
    returned. Unresolved algebraic roots are not guessed.
    """
    f = sp.sympify(equation)
    y = sp.sympify(dependent)
    x = sp.sympify(variable)
    try:
        sols = bounded_solve_one(f, y, allow_general=True)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return ()
    out = []
    for sol in sols:
        try:
            lv = bounded_limit(sol, x, sp.sympify(point), allow_general=True)
        except (TypeError, ValueError, NotImplementedError):
            continue
        if sp.simplify(lv - sp.sympify(dependent_limit)) != 0:
            continue
        z = sp.Dummy("branch_z", positive=True)
        local = sol.xreplace({x: sp.sympify(point) + z})
        try:
            s = (
                sp.series(local, z, 0, order)
                .removeO()
                .xreplace({z: x - sp.sympify(point)})
            )
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            continue
        out.append(sp.expand(s))
    return tuple(out)
