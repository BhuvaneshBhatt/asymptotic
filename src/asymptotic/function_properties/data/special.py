"""Small reviewed seed set of SymPy special-function properties."""

from __future__ import annotations

import sympy as sp

from ..model import (
    AssumptionProperties,
    DomainProperties,
    FunctionProperties,
    SingularityLocus,
    SingularityProperties,
)
from ..registry import FunctionPropertyRegistry
from .elementary import single_entire


def gamma_properties(expr: sp.Expr) -> FunctionProperties:
    (z,) = expr.args
    nonpositive_integer = sp.And(sp.Q.integer(z), z <= 0)
    real = sp.And(sp.Q.real(z), sp.Not(nonpositive_integer))
    return FunctionProperties(
        expression=expr,
        arguments=(z,),
        assumptions=AssumptionProperties(real=real, real_if_defined=sp.Q.real(z)),
        domain=DomainProperties(
            (z,),
            real_domain=real,
            complex_domain=sp.Not(nonpositive_integer),
        ),
        singularities=SingularityProperties(
            locally_analytic=True,
            branch_cuts=(),
            definition_cuts=(),
            poles=(SingularityLocus(nonpositive_integer),),
            essential=(),
            branch_points=(),
        ),
    )


def register(registry: FunctionPropertyRegistry) -> None:
    for function in (sp.erf, sp.erfc, sp.erfi, sp.airyai, sp.airybi):
        registry.register(function, single_entire)
    registry.register(sp.gamma, gamma_properties)
