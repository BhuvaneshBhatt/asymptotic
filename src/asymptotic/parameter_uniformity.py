"""Certified parameter-uniform local expansions."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_assumption_sign
from .coverage import CoverageCertificate
from .proof_obligations import ProofObligation
from .uniform_remainder import UniformRemainder


@dataclass(frozen=True)
class ParameterInterval:
    parameter: sp.Symbol
    lower: sp.Expr
    upper: sp.Expr

    def __post_init__(self):
        if self.parameter in (
            sp.sympify(self.lower).free_symbols | sp.sympify(self.upper).free_symbols
        ):
            raise ValueError(
                "parameter interval endpoints must be parameter-independent"
            )

    @property
    def condition(self):
        return sp.And(
            sp.Ge(self.parameter, self.lower), sp.Le(self.parameter, self.upper)
        )

    @property
    def absolute_bound(self):
        return sp.Max(sp.Abs(self.lower), sp.Abs(self.upper))


@dataclass(frozen=True)
class ParameterUniformExpansion:
    expression: sp.Expr
    approximation: sp.Expr
    variable: sp.Symbol
    target: sp.Expr
    parameter_domains: tuple[ParameterInterval, ...]
    remainder_certificate: UniformRemainder
    order: int
    method: str
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def domain(self):
        return (
            sp.And(*(d.condition for d in self.parameter_domains))
            if self.parameter_domains
            else sp.S.true
        )

    @property
    def remainder_scale(self):
        return self.remainder_certificate.scale

    @property
    def certified(self):
        return self.remainder_certificate.certified and not self.obligations

    def uniform_remainder(self):
        return self.remainder_certificate


def _coverage(statement):
    return CoverageCertificate.complete("parameter_uniformity", statement)


def _unknown(expr, x, domains, order, statement):
    r = UniformRemainder.unknown(
        x,
        uniform_variables=tuple(d.parameter for d in domains),
        domain=sp.And(*(d.condition for d in domains)),
        provider="parameter_uniformity",
        statement=statement,
    )
    return ParameterUniformExpansion(
        expr, sp.S.NaN, x, 0, domains, r, order, "unsupported", r.obligations
    )


def uniform_parameter_expand(
    expr, variable, *, parameters, target=0, order=4, radius_bound=None
):
    """Expand at a finite target with one remainder constant valid on parameter boxes.

    Parameter-uniform expansion certifies polynomial exactness and the elementary entire/geometric
    families exp(a*x), sin(a*x), cos(a*x), and log(1+a*x).  Unsupported forms
    decline with a proof obligation rather than returning a pointwise bound.
    """
    expr = sp.sympify(expr)
    x = sp.sympify(variable)
    target = sp.sympify(target)
    domains = tuple(parameters)
    if order < 1:
        raise ValueError("order must be positive")
    if target != 0:
        shifted = sp.Dummy("h", real=True)
        sub = uniform_parameter_expand(
            expr.subs(x, shifted + target),
            shifted,
            parameters=domains,
            target=0,
            order=order,
            radius_bound=radius_bound,
        )
        if not sub.certified:
            return _unknown(
                expr,
                x,
                domains,
                order,
                "translated parameter-uniform theorem not certified",
            )
        approx = sp.expand(sub.approximation.subs(shifted, x - target))
        rem = sub.remainder_certificate.substitute(x - target, x)
        return ParameterUniformExpansion(
            expr, approx, x, target, domains, rem, order, "translated_" + sub.method
        )
    domain = sp.And(*(d.condition for d in domains)) if domains else sp.S.true
    params = {d.parameter: d for d in domains}
    if expr.free_symbols - {x} - set(params):
        return _unknown(
            expr, x, domains, order, "all symbolic parameters require bounded domains"
        )
    try:
        approx = sp.series(expr, x, 0, order).removeO().expand()
    except (ValueError, NotImplementedError, TypeError):
        return _unknown(
            expr, x, domains, order, "bounded Taylor construction unavailable"
        )
    remainder = sp.simplify(expr - approx)
    if remainder == 0:
        rem = UniformRemainder.exact_zero(
            x,
            uniform_variables=tuple(params),
            domain=domain,
            provider="parameter_uniform_polynomial",
        )
        return ParameterUniformExpansion(
            expr, approx, x, 0, domains, rem, order, "exact_polynomial"
        )
    if radius_bound is None:
        return _unknown(
            expr,
            x,
            domains,
            order,
            "a positive radius_bound is required for a uniform analytic remainder",
        )
    rho = sp.sympify(radius_bound)
    # Recognize f(a*x), one bounded parameter, with exact classical derivative bounds.
    if len(domains) != 1:
        return _unknown(
            expr,
            x,
            domains,
            order,
            "elementary uniform theorem supports one bounded parameter",
        )
    d = domains[0]
    a = d.parameter
    A = d.absolute_bound
    method = None
    C = None
    if expr == sp.exp(a * x):
        method = "uniform_exp_taylor"
        C = sp.simplify(A**order * sp.exp(A * rho) / sp.factorial(order))
    elif expr in (sp.sin(a * x), sp.cos(a * x)):
        method = "uniform_trig_taylor"
        C = sp.simplify(A**order / sp.factorial(order))
    elif expr == sp.log(1 + a * x):
        # |remainder| <= (A|x|)^n/[n(1-A|x|)] for A*rho<1.
        if bounded_assumption_sign(1 - A * rho) != 1:
            return _unknown(
                expr,
                x,
                domains,
                order,
                "log/geometric uniformity requires sup|a|*radius_bound < 1",
            )
        method = "uniform_log_taylor"
        C = sp.simplify(A**order / (order * (1 - A * rho)))
    elif expr == 1 / (1 + a * x):
        if bounded_assumption_sign(1 - A * rho) != 1:
            return _unknown(
                expr,
                x,
                domains,
                order,
                "geometric uniformity requires sup|a|*radius_bound < 1",
            )
        method = "uniform_geometric"
        C = sp.simplify(A**order / (1 - A * rho))
    if method is None:
        return _unknown(
            expr,
            x,
            domains,
            order,
            "no parameter-uniform remainder theorem for this analytic family",
        )
    rem = UniformRemainder.big_o(
        x**order,
        x,
        constant=C,
        radius_bound=rho,
        uniform_variables=(a,),
        domain=domain,
        exact_expression=remainder,
        provider=method,
        statement="single constant valid over the full parameter interval",
        coverage=_coverage(
            "parameter interval and radial neighborhood covered uniformly"
        ),
    )
    return ParameterUniformExpansion(expr, approx, x, 0, domains, rem, order, method)
