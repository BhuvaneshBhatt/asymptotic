"""Reviewed SymPy property entries for elementary functions."""

from __future__ import annotations

import sympy as sp

from ..model import (
    AssumptionProperties,
    DomainProperties,
    FunctionProperties,
    RealUnivariateProperties,
    SingularityLocus,
    SingularityProperties,
)
from ..registry import FunctionPropertyRegistry


def _entire(expr: sp.Expr, *, real_range: sp.Set | sp.Expr | None = None) -> FunctionProperties:
    args = tuple(expr.args)
    real = sp.And(*(sp.Q.real(arg) for arg in args)) if args else sp.S.true
    variable = args[0] if len(args) == 1 else None
    return FunctionProperties(
        expression=expr,
        arguments=args,
        assumptions=AssumptionProperties(real=real, real_if_defined=real),
        domain=DomainProperties(args, real_domain=sp.S.true, complex_domain=sp.S.true),
        singularities=SingularityProperties(
            locally_analytic=True,
            branch_cuts=(),
            definition_cuts=(),
            poles=(),
            essential=(),
            branch_points=(),
        ),
        real_univariate=(
            RealUnivariateProperties(variable=variable, range=real_range)
            if variable is not None and real_range is not None
            else None
        ),
    )


def single_entire(expr: sp.Expr) -> FunctionProperties:
    return _entire(expr)


def log_properties(expr: sp.Expr) -> FunctionProperties:
    (z,) = expr.args
    real = sp.And(sp.Q.real(z), z > 0)
    negative_axis = sp.And(sp.Eq(sp.im(z), 0), sp.re(z) <= 0)
    return FunctionProperties(
        expression=expr,
        arguments=(z,),
        assumptions=AssumptionProperties(real=real, real_if_defined=real),
        domain=DomainProperties((z,), real_domain=real, complex_domain=sp.Ne(z, 0)),
        singularities=SingularityProperties(
            locally_analytic=True,
            branch_cuts=(SingularityLocus(negative_axis, jump=2 * sp.pi * sp.I),),
            definition_cuts=(),
            poles=(),
            essential=(),
            branch_points=(SingularityLocus(sp.Eq(z, 0)),),
        ),
        real_univariate=RealUnivariateProperties(variable=z, range=sp.S.Reals),
    )


def sqrt_properties(expr: sp.Expr) -> FunctionProperties:
    z = expr.base
    real = sp.And(sp.Q.real(z), z >= 0)
    negative_axis = sp.And(sp.Eq(sp.im(z), 0), sp.re(z) < 0)
    return FunctionProperties(
        expression=expr,
        arguments=(z,),
        assumptions=AssumptionProperties(real=real, real_if_defined=real, nonnegative=real),
        domain=DomainProperties((z,), real_domain=real, complex_domain=sp.S.true),
        singularities=SingularityProperties(
            locally_analytic=False,
            branch_cuts=(SingularityLocus(negative_axis, jump=2 * sp.I * sp.sqrt(-z)),),
            definition_cuts=(),
            poles=(),
            essential=(),
            branch_points=(SingularityLocus(sp.Eq(z, 0)),),
        ),
        real_univariate=RealUnivariateProperties(variable=z, range=sp.Interval(0, sp.oo)),
    )


def asin_properties(expr: sp.Expr) -> FunctionProperties:
    (z,) = expr.args
    real = sp.And(sp.Q.real(z), z >= -1, z <= 1)
    cuts = sp.And(sp.Eq(sp.im(z), 0), sp.Or(sp.re(z) <= -1, sp.re(z) >= 1))
    return FunctionProperties(
        expression=expr,
        arguments=(z,),
        assumptions=AssumptionProperties(real=real, real_if_defined=real),
        domain=DomainProperties((z,), real_domain=real, complex_domain=sp.S.true),
        singularities=SingularityProperties(
            locally_analytic=True,
            branch_cuts=(SingularityLocus(cuts),),
            definition_cuts=(),
            poles=(),
            essential=(),
            branch_points=(SingularityLocus(sp.Eq(z, -1)), SingularityLocus(sp.Eq(z, 1))),
        ),
        real_univariate=RealUnivariateProperties(
            variable=z, range=sp.Interval(-sp.pi / 2, sp.pi / 2), injective=True
        ),
    )


def acos_properties(expr: sp.Expr) -> FunctionProperties:
    (z,) = expr.args
    real = sp.And(sp.Q.real(z), z >= -1, z <= 1)
    cuts = sp.And(sp.Eq(sp.im(z), 0), sp.Or(sp.re(z) <= -1, sp.re(z) >= 1))
    return FunctionProperties(
        expression=expr,
        arguments=(z,),
        assumptions=AssumptionProperties(real=real, real_if_defined=real, nonnegative=real),
        domain=DomainProperties((z,), real_domain=real, complex_domain=sp.S.true),
        singularities=SingularityProperties(
            locally_analytic=True,
            branch_cuts=(SingularityLocus(cuts),),
            definition_cuts=(),
            poles=(),
            essential=(),
            branch_points=(SingularityLocus(sp.Eq(z, -1)), SingularityLocus(sp.Eq(z, 1))),
        ),
        real_univariate=RealUnivariateProperties(
            variable=z, range=sp.Interval(0, sp.pi), injective=True
        ),
    )


def atan_properties(expr: sp.Expr) -> FunctionProperties:
    (z,) = expr.args
    real = sp.Q.real(z)
    cuts = sp.And(sp.Eq(sp.re(z), 0), sp.Abs(sp.im(z)) >= 1)
    return FunctionProperties(
        expression=expr,
        arguments=(z,),
        assumptions=AssumptionProperties(real=real, real_if_defined=real),
        domain=DomainProperties((z,), real_domain=real, complex_domain=sp.S.true),
        singularities=SingularityProperties(
            locally_analytic=True,
            branch_cuts=(SingularityLocus(cuts),),
            definition_cuts=(),
            poles=(),
            essential=(),
            branch_points=(SingularityLocus(sp.Eq(z, sp.I)), SingularityLocus(sp.Eq(z, -sp.I))),
        ),
        real_univariate=RealUnivariateProperties(
            variable=z, range=sp.Interval.open(-sp.pi / 2, sp.pi / 2), injective=True
        ),
    )


def register(registry: FunctionPropertyRegistry) -> None:
    for function in (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.tanh):
        registry.register(function, single_entire)
    registry.register(sp.log, log_properties)
    registry.register("sqrt", sqrt_properties)
    registry.register(sp.asin, asin_properties)
    registry.register(sp.acos, acos_properties)
    registry.register(sp.atan, atan_properties)
