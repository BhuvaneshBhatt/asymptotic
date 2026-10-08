"""Certified leading asymptotic data for common special functions at infinity."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class InfinityAsymptotic:
    expression: sp.Expr
    leading: sp.Expr
    relative_order: sp.Expr | None
    conditions: sp.Expr
    provider: str = "special_function_infinity"


def special_function_infinity(expr, variable, *, assumptions=True):
    expr = sp.sympify(expr)
    x = sp.sympify(variable)
    # SymPy evaluates uppergamma(n,x), n positive integer, to exp(-x)
    # times its exact terminating polynomial. Recognize that canonical form.
    scaled = sp.expand(expr * sp.exp(x))
    if scaled.is_polynomial(x) and expr.has(sp.exp(-x)):
        poly = sp.Poly(scaled, x)
        if poly.degree() >= 0:
            lead = poly.LC() * x ** poly.degree() * sp.exp(-x)
            return InfinityAsymptotic(expr, lead, 1 / x, sp.Gt(x, 0))
    # Standard real x -> +infinity leading forms. Conditions are explicit.
    if expr.func is sp.erf and expr.args == (x,):
        return InfinityAsymptotic(expr, sp.S.One, sp.exp(-(x**2)) / x, sp.Gt(x, 0))
    if expr.func is sp.erfc and expr.args == (x,):
        return InfinityAsymptotic(
            expr, sp.exp(-(x**2)) / (sp.sqrt(sp.pi) * x), x**-2, sp.Gt(x, 0)
        )
    if expr.func is sp.gamma and expr.args == (x,):
        lead = sp.sqrt(2 * sp.pi / x) * (x / sp.E) ** x
        return InfinityAsymptotic(expr, lead, x**-1, sp.Gt(x, 0))
    if (
        expr.func in (sp.besselj, sp.bessely)
        and len(expr.args) == 2
        and expr.args[1] == x
    ):
        nu = expr.args[0]
        phase = x - sp.pi * nu / 2 - sp.pi / 4
        trig = sp.cos(phase) if expr.func is sp.besselj else sp.sin(phase)
        return InfinityAsymptotic(
            expr, sp.sqrt(2 / (sp.pi * x)) * trig, x**-1, sp.Gt(x, 0)
        )
    if expr.func is sp.airyai and expr.args == (x,):
        return InfinityAsymptotic(
            expr,
            sp.exp(-sp.Rational(2, 3) * x ** sp.Rational(3, 2))
            / (2 * sp.sqrt(sp.pi) * x ** sp.Rational(1, 4)),
            x ** -sp.Rational(3, 2),
            sp.Gt(x, 0),
        )
    if expr.func is sp.airyaiprime and expr.args == (x,):
        return InfinityAsymptotic(
            expr,
            -(x ** sp.Rational(1, 4))
            * sp.exp(-sp.Rational(2, 3) * x ** sp.Rational(3, 2))
            / (2 * sp.sqrt(sp.pi)),
            x ** -sp.Rational(3, 2),
            sp.Gt(x, 0),
        )
    if expr.func is sp.airybi and expr.args == (x,):
        return InfinityAsymptotic(
            expr,
            sp.exp(sp.Rational(2, 3) * x ** sp.Rational(3, 2))
            / (sp.sqrt(sp.pi) * x ** sp.Rational(1, 4)),
            x ** -sp.Rational(3, 2),
            sp.Gt(x, 0),
        )
    if expr.func is sp.airybiprime and expr.args == (x,):
        return InfinityAsymptotic(
            expr,
            x ** sp.Rational(1, 4)
            * sp.exp(sp.Rational(2, 3) * x ** sp.Rational(3, 2))
            / sp.sqrt(sp.pi),
            x ** -sp.Rational(3, 2),
            sp.Gt(x, 0),
        )
    if expr.func is sp.zeta and expr.args == (x,):
        return InfinityAsymptotic(expr, sp.S.One, 2 ** (-x), sp.Gt(x, 1))
    if expr.func is sp.primepi and expr.args == (x,):
        return InfinityAsymptotic(expr, x / sp.log(x), 1 / sp.log(x), sp.Gt(x, 1))
    if expr.func is sp.polygamma and len(expr.args) == 2 and expr.args[1] == x:
        n = expr.args[0]
        if n == 0:
            return InfinityAsymptotic(expr, sp.log(x), 1 / x, sp.Gt(x, 0))
        if n == 1:
            return InfinityAsymptotic(expr, 1 / x, 1 / x, sp.Gt(x, 0))
    if expr.func is sp.uppergamma and len(expr.args) == 2 and expr.args[1] == x:
        a = expr.args[0]
        return InfinityAsymptotic(expr, x ** (a - 1) * sp.exp(-x), 1 / x, sp.Gt(x, 0))
    return None
