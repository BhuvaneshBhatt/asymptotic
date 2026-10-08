"""Automatic geometry analysis for coalescing saddle-point integrals."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_limit, bounded_polynomial_roots
from .regime_selection import RegimeGeometry


@dataclass(frozen=True)
class SaddleGeometryCertificate:
    phase: sp.Expr
    integration_variable: sp.Symbol
    large_parameter: sp.Symbol
    saddles: tuple[sp.Expr, ...]
    coalescing_pair: tuple[sp.Expr, sp.Expr] | None
    separation: sp.Expr | None
    separation_exponent: sp.Expr | None
    geometry: RegimeGeometry
    hypotheses_verified: bool


def _power_scale(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr | None:
    """Prove expr=O(variable**power) from its logarithmic order."""
    try:
        power = bounded_limit(sp.log(sp.Abs(expr)) / sp.log(variable), variable, sp.oo)
    except (TypeError, ValueError, NotImplementedError, RecursionError, sp.PoleError):
        return None
    if power is None or isinstance(power, sp.Limit) or power.has(sp.nan, sp.zoo):
        return None
    return sp.simplify(power)


def analyze_saddle_geometry(
    phase: sp.Expr,
    integration_variable: sp.Symbol,
    large_parameter: sp.Symbol,
) -> SaddleGeometryCertificate:
    """Find polynomial saddles and prove whether a pair coalesces."""
    phase = sp.sympify(phase)
    t = sp.sympify(integration_variable)
    n = sp.sympify(large_parameter)
    derivative = sp.factor(sp.diff(phase, t))
    if not derivative.is_polynomial(t):
        return SaddleGeometryCertificate(
            phase, t, n, (), None, None, None, RegimeGeometry(), False
        )
    polynomial = sp.Poly(derivative, t)
    if polynomial.degree() > 4:
        return SaddleGeometryCertificate(
            phase, t, n, (), None, None, None, RegimeGeometry(), False
        )
    roots = bounded_polynomial_roots(polynomial.as_expr(), t) or ()
    best = None
    for i, first in enumerate(roots):
        for second in roots[i + 1 :]:
            separation = sp.simplify(second - first)
            power = _power_scale(separation, n)
            if (
                power is not None
                and power.is_negative is True
                and (best is None or sp.N(power) < sp.N(best[3]))
            ):
                best = (first, second, separation, power)
    if best is None:
        return SaddleGeometryCertificate(
            phase, t, n, roots, None, None, None, RegimeGeometry(), False
        )
    first, second, separation, power = best
    geometry = RegimeGeometry(
        coalescing_saddles=(first, second),
        saddle_separation_scale=separation,
        hypotheses_verified=True,
    )
    return SaddleGeometryCertificate(
        phase, t, n, roots, (first, second), separation, power, geometry, True
    )


def analyze_exponential_integral(
    integral: sp.Integral,
    large_parameter: sp.Symbol,
) -> SaddleGeometryCertificate | None:
    """Extract exp(-n*phase) from a one-dimensional integral and analyze it."""
    if not isinstance(integral, sp.Integral) or len(integral.limits) != 1:
        return None
    t = integral.limits[0][0]
    exponentials = tuple(integral.function.atoms(sp.exp))
    candidates = []
    for exponential in exponentials:
        exponent = sp.expand(exponential.args[0])
        phase = sp.simplify(-exponent / large_parameter)
        if sp.simplify(exponent + large_parameter * phase) != 0:
            continue
        candidates.append(phase)
    if len(candidates) != 1:
        return None
    return analyze_saddle_geometry(candidates[0], t, large_parameter)
