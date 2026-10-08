"""Optimization and exact extremal certificates for multivariate limits."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._multivariate_germ import (
    ParityCertificate,
    _cheap_lojasiewicz_exponent_bound,
    _coercive_polynomial_model,
    _nonnegative_rational_axis_minimum,
    _poly_order,
    _shift,
    isolated_zero_certificate,
)


def _cheap_lagrange_critical_path_certificate(expr, variables, target):
    """Certify a zero rational limit from radial critical-order domination.

    This fast path recognizes homogeneous numerator/positive-definite denominator
    germs; general Lagrange systems are left to exact algebraic
    backends rather than solved heuristically.
    """
    u, f = _shift(expr, variables, target)
    num, den = sp.fraction(sp.cancel(f))
    on, od = _poly_order(num, u), _poly_order(den, u)
    loj = _cheap_lojasiewicz_exponent_bound(den, u, (0,) * len(u))
    if on not in (None, sp.oo) and loj.certified and on > int(loj.value):
        return ParityCertificate(
            "lagrange_critical",
            True,
            sp.S.Zero,
            "radial critical values vanish by certified order domination",
        )
    return ParityCertificate("lagrange_critical", False, data=(on, od))


def extremal_limit(expr, variables, target, *, kind="min", domain=True):
    """First-class local extremal-limit semantics via cluster liminf/limsup."""
    from .multivariate_limits_advanced import multivariate_liminf, multivariate_limsup

    if kind == "min":
        cheap_min = _nonnegative_rational_axis_minimum(expr, variables, target)
        if cheap_min is not None:
            return ParityCertificate(
                "min_limit",
                True,
                cheap_min,
                "nonnegativity plus an exact zero approach certifies the local infimum",
            )
    fn = multivariate_liminf if kind == "min" else multivariate_limsup
    try:
        bound = fn(expr, variables, target, domain=domain, return_result=True)
    except (TypeError, ValueError, NotImplementedError):
        return ParityCertificate(f"{kind}_limit", False)
    if (
        bound.certified
        and bound.value is not None
        and bound.value not in (sp.nan, sp.zoo)
    ):
        return ParityCertificate(
            f"{kind}_limit",
            True,
            bound.value,
            f"certified local {kind} extremal limit via {bound.provider}",
            (bound,),
        )
    return ParityCertificate(f"{kind}_limit", False, data=(bound,))


@dataclass(frozen=True)
class LojasiewiczExponentResult:
    """Exact local radial Łojasiewicz exponent search result."""

    exponent: sp.Integer | None
    certified: bool
    lower_bound_formula: sp.Expr | None = None
    provider: str = "none"
    searched: tuple[int, ...] = ()


def general_lojasiewicz_exponent(expr, variables, target, *, max_exponent=24):
    """Find the least integer m certified by QE with |f| >= c ||x-a||**m.

    For polynomial germs this is a complete search up to ``max_exponent``.
    The returned exponent is an exact *integer radial bound*, not a numerical
    fit.  If the configured search ceiling is reached the result is explicitly
    uncertified.
    """
    cheap = _cheap_lojasiewicz_exponent_bound(expr, variables, target)
    if cheap is not None and cheap.certified:
        return LojasiewiczExponentResult(
            sp.Integer(cheap.value),
            True,
            provider="pure_power_exact",
            searched=(int(cheap.value),),
        )
    u, f = _shift(expr, variables, target)
    try:
        p = sp.Poly(sp.expand(f), *u)
    except (sp.PolynomialError, TypeError, ValueError):
        return LojasiewiczExponentResult(None, False, provider="nonpolynomial")
    if p.is_zero:
        return LojasiewiczExponentResult(None, False, provider="zero_germ")
    order = int(_poly_order(p.as_expr(), u))
    r2 = sp.Add(*(x * x for x in u))
    from .domain_witnesses import attained_ray

    if (
        attained_ray(sp.And(sp.Eq(p.as_expr(), 0), r2 > 0), u, (0,) * len(u))
        is not None
    ):
        return LojasiewiczExponentResult(None, False, provider="nonisolated_zero_ray")
    # Quantified variables range over all reals. Encode positivity in the
    # formula: symbol assumptions would simplify these guards away before QE.
    c = sp.Dummy("_loj_c", real=True)
    delta = sp.Dummy("_loj_delta", real=True)
    searched = []
    try:
        from semialg import quantifier_eliminate
    except (ImportError, ModuleNotFoundError):
        return LojasiewiczExponentResult(None, False, provider="semialg_unavailable")
    for m in range(max(1, order), max_exponent + 1):
        searched.append(m)
        implication = sp.Or(
            sp.Not(sp.And(sp.Gt(r2, 0), sp.Lt(r2, delta**2))),
            sp.Ge(p.as_expr() ** 2, c**2 * r2**m),
        )
        formula = sp.And(sp.Gt(c, 0), sp.Gt(delta, 0), implication)
        quantifiers = [("exists", c), ("exists", delta), *(("forall", x) for x in u)]
        try:
            q = quantifier_eliminate(
                formula, quantifiers=quantifiers, variables=(c, delta, *u)
            )
        except (
            ArithmeticError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
        ):
            continue
        if q is sp.S.true or sp.simplify(q) is sp.S.true:
            return LojasiewiczExponentResult(
                sp.Integer(m), True, formula, "semialg_qe", tuple(searched)
            )
    return LojasiewiczExponentResult(
        None, False, provider="search_exhausted", searched=tuple(searched)
    )


def general_isolated_zero_certificate(expr, variables, target):
    """Certify an isolated real zero by coercivity, then exact elimination."""
    cheap = isolated_zero_certificate(expr, variables, target)
    if cheap.certified:
        return ParityCertificate(
            "isolated_zero_qe", True, cheap.value, cheap.statement, (cheap,)
        )
    u, f = _shift(expr, variables, target)
    try:
        p = sp.Poly(sp.expand(f), *u)
    except (sp.PolynomialError, TypeError, ValueError):
        return ParityCertificate("isolated_zero_qe", False)
    model = _coercive_polynomial_model(p.as_expr(), u)
    if p.TC() == 0 and model is not None:
        return ParityCertificate(
            "isolated_zero_coercive",
            True,
            sp.S.true,
            "a definite weighted principal form dominates the higher-order remainder",
            (model,),
        )
    r2 = sp.Add(*(x * x for x in u))
    from .domain_witnesses import attained_ray

    direction = attained_ray(sp.And(sp.Eq(p.as_expr(), 0), r2 > 0), u, (0,) * len(u))
    if direction is not None:
        return ParityCertificate(
            "nonisolated_zero_ray",
            True,
            sp.S.false,
            "the polynomial vanishes along an attained punctured ray",
            (direction,),
        )
    # Keep delta > 0 as a formula atom so elimination cannot choose delta=0.
    delta = sp.Dummy("_iso_delta", real=True)
    body = sp.Or(
        sp.Not(sp.And(sp.Gt(r2, 0), sp.Lt(r2, delta**2))),
        sp.Gt(p.as_expr() ** 2, 0),
    )
    try:
        from semialg import quantifier_eliminate

        q = quantifier_eliminate(
            sp.And(sp.Gt(delta, 0), body),
            quantifiers=[("exists", delta), *(("forall", x) for x in u)],
            variables=(delta, *u),
        )
    except (
        ImportError,
        ModuleNotFoundError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return ParityCertificate("isolated_zero_qe", False)
    if q is sp.S.true or sp.simplify(q) is sp.S.true:
        return ParityCertificate(
            "isolated_zero_qe",
            True,
            sp.S.true,
            "punctured-neighborhood nonvanishing certified by QE",
        )
    if q is sp.S.false or sp.simplify(q) is sp.S.false:
        return ParityCertificate(
            "isolated_zero_qe", True, sp.S.false, "non-isolated zero certified by QE"
        )
    return ParityCertificate("isolated_zero_qe", False, data=(q,))


def complete_lagrange_critical_certificate(
    expr, variables, target, *, domain=sp.S.true
):
    """Certify a scalar limit from the complete radial critical locus.

    Extrema of a smooth scalar germ on sufficiently small spheres occur where
    ``x_i df/dx_j-x_j df/dx_i = 0``.  Once poles are excluded in a punctured
    neighborhood, agreement of the exact limit on this complete critical locus
    certifies the ambient limit by the min/max squeeze on each sphere.
    """
    u, f = _shift(expr, variables, target)
    num, den = sp.fraction(sp.cancel(f))
    if not (num.is_polynomial(*u) and den.is_polynomial(*u)):
        return ParityCertificate("lagrange_critical_complete", False)
    iso = (
        general_isolated_zero_certificate(den, u, (0,) * len(u))
        if den.subs(dict.fromkeys(u, 0)) == 0
        else None
    )
    if iso is not None and not (iso.certified and iso.value is sp.S.true):
        return ParityCertificate(
            "lagrange_critical_complete",
            False,
            statement="denominator pole set not excluded",
        )
    dd = [sp.expand(sp.diff(num, x) * den - sp.diff(den, x) * num) for x in u]
    equations = []
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            equations.append(sp.Eq(sp.expand(u[i] * dd[j] - u[j] * dd[i]), 0))
    critical = sp.And(*equations) if equations else sp.S.true
    shifted_domain = sp.sympify(domain).subs(
        dict(
            zip(variables, (a + x for a, x in zip(target, u, strict=True)), strict=True)
        )
    )
    critical_domain = sp.And(shifted_domain, critical)
    try:
        from .limits import limit

        res = limit(f, u, (0,) * len(u), domain=critical_domain, return_result=True)
    except (TypeError, ValueError, NotImplementedError, RuntimeError, ArithmeticError):
        return ParityCertificate("lagrange_critical_complete", False, data=(critical,))
    status = getattr(res, "status", None)
    value = getattr(res, "value", None)
    if (
        status is not None
        and getattr(status, "name", "") == "EXISTS"
        and value is not None
    ):
        return ParityCertificate(
            "lagrange_critical_complete",
            True,
            value,
            "all radial critical values have one certified limit; compact-sphere extrema squeeze the germ",
            (critical, res),
        )
    return ParityCertificate("lagrange_critical_complete", False, data=(critical, res))


def lojasiewicz_exponent_bound(expr, variables, target):
    cheap = _cheap_lojasiewicz_exponent_bound(expr, variables, target)
    if cheap.certified:
        return cheap
    exact = general_lojasiewicz_exponent(expr, variables, target)
    if exact.certified:
        return ParityCertificate(
            "lojasiewicz",
            True,
            exact.exponent,
            "exact radial Łojasiewicz exponent bound certified by QE",
            (exact,),
        )
    return cheap


def lagrange_critical_path_certificate(expr, variables, target):
    cheap = _cheap_lagrange_critical_path_certificate(expr, variables, target)
    if cheap.certified:
        return cheap
    return complete_lagrange_critical_certificate(expr, variables, target)
