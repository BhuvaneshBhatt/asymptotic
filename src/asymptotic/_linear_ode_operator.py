"""Shared extraction of homogeneous scalar linear differential operators."""

from __future__ import annotations

import sympy as sp


def linear_operator_coefficients(
    operator: sp.Expr,
    delta: sp.Expr,
    variable: sp.Symbol,
) -> tuple[tuple[sp.Expr, ...], int] | None:
    """Return ``(a_0, ..., a_r), r`` for ``sum a_k delta^(k)``.

    The routine rejects non-polynomial, nonlinear, and inhomogeneous jet
    dependence.  It performs one algebraic replacement/``Poly`` conversion and
    does not invoke generic solving or simplification of the full operator.
    """

    derivatives = [d for d in operator.atoms(sp.Derivative) if d.expr == delta]
    order = max((d.derivative_count for d in derivatives), default=0)
    jets = tuple(sp.Dummy(f"linear_ode_D{k}") for k in range(order + 1))
    replacements: dict[sp.Expr, sp.Expr] = {delta: jets[0]}
    replacements.update({sp.diff(delta, variable, k): jets[k] for k in range(1, order + 1)})
    algebraic = sp.expand(operator.xreplace(replacements))
    try:
        poly = sp.Poly(algebraic, *jets)
    except sp.PolynomialError:
        return None
    if poly.total_degree() > 1:
        return None
    zero_monomial = (0,) * len(jets)
    if sp.simplify(poly.coeff_monomial(zero_monomial)) != 0:
        return None
    coefficients = tuple(sp.simplify(poly.coeff_monomial(jet)) for jet in jets)
    return coefficients, order
