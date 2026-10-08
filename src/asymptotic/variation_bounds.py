"""Total-variation majorants for Liouville--Green error bounds."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class VariationCertificate:
    """A computable upper bound for total variation along a stated path."""

    function: sp.Expr
    variable: sp.Symbol
    start: sp.Expr
    end: sp.Expr
    bound: sp.Expr
    path: str
    hypotheses_verified: bool
    exact_variation: bool = False


@dataclass(frozen=True)
class OlverErrorControl:
    """Olver-style exponential majorant built from coefficient variations."""

    truncation_index: int
    first_variation: VariationCertificate
    omitted_variation: VariationCertificate
    large_parameter: sp.Expr
    constant: sp.Expr
    bound: sp.Expr
    hypotheses_verified: bool


def polynomial_segment_variation(
    polynomial: sp.Expr,
    variable: sp.Symbol,
    start: sp.Expr,
    end: sp.Expr,
) -> VariationCertificate:
    """Bound variation of a polynomial on a straight segment coefficientwise.

    For ``q(s)=p(start+s*(end-start))`` and ``0<=s<=1``, integrating the
    coefficientwise majorant of ``|q'(s)|`` gives a rigorous upper bound.
    """
    polynomial = sp.sympify(polynomial)
    start = sp.sympify(start)
    end = sp.sympify(end)
    s = sp.Symbol("_s", real=True, nonnegative=True)
    q = sp.Poly(sp.expand(polynomial.subs(variable, start + s * (end - start))), s)
    derivative = sp.Poly(sp.diff(q.as_expr(), s), s)
    bound = sp.Add(
        *(
            sp.Abs(coefficient) / (power[0] + 1)
            for power, coefficient in derivative.terms()
        )
    )
    verified = all(coefficient.is_number for _, coefficient in derivative.terms()) or (
        start.is_real is True and end.is_real is True
    )
    return VariationCertificate(
        polynomial,
        variable,
        start,
        end,
        sp.simplify(bound),
        "straight segment with coefficientwise polynomial majorant",
        verified,
    )


def debye_variation(
    polynomial: sp.Expr,
    p_value: sp.Expr,
) -> VariationCertificate:
    """Variation bound from the Debye base point p=0 to ``p_value``."""
    p = sp.Symbol("p")
    return polynomial_segment_variation(polynomial, p, sp.S.Zero, p_value)


def olver_error_control(
    first_coefficient: sp.Expr,
    omitted_coefficient: sp.Expr,
    p_value: sp.Expr,
    large_parameter: sp.Expr,
    truncation_index: int,
    *,
    amplitude: sp.Expr = sp.S.One,
) -> OlverErrorControl:
    """Construct the standard exponential total-variation majorant.

    The majorant has the Liouville--Green form
    ``2 V_N / |u|**N * exp(2 V_1 / |u|)``.  ``V_k`` are certified total-
    variation upper bounds, so the returned expression is directly
    evaluable and contains no unspecified theorem constant.
    """
    if truncation_index < 1:
        raise ValueError("truncation_index must be positive")
    u = sp.sympify(large_parameter)
    first = debye_variation(first_coefficient, p_value)
    omitted = debye_variation(omitted_coefficient, p_value)
    constant = 2 * sp.exp(2 * first.bound / sp.Abs(u))
    bound = sp.simplify(
        sp.Abs(amplitude) * constant * omitted.bound / sp.Abs(u) ** truncation_index
    )
    verified = (
        first.hypotheses_verified
        and omitted.hypotheses_verified
        and (u.is_positive is True or (u.is_number and sp.Abs(u).is_positive is True))
    )
    return OlverErrorControl(
        truncation_index, first, omitted, u, constant, bound, verified
    )


class TotalVariation(sp.Function):
    """Symbolic exact total-variation functional with family-specific evaluation."""

    nargs = 2


def turning_point_variation(family: str, z: sp.Expr) -> sp.Expr:
    """Return the exact variation functional used by a turning-point family."""
    return TotalVariation(sp.Symbol(family), sp.sympify(z))


def materialize_turning_point_variation(family: str, z: sp.Expr) -> sp.Expr:
    """Materialize a represented turning-point variation as an exact integral."""
    if family != "bessel_A1_B0":
        raise ValueError("unknown turning-point variation family")
    from .olver_coefficients import olver_coefficient

    z = sp.sympify(z)
    if z == 1:
        return sp.S.Zero
    s = sp.Symbol("_tv", real=True, nonnegative=True)
    path = 1 + s * (z - 1)
    control = olver_coefficient("A", 1, path) + olver_coefficient("B", 0, path)
    return sp.Integral(sp.Abs(sp.diff(control, s)), (s, 0, 1))
