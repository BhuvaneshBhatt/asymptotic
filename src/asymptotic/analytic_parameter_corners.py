"""Analytic compositions of parameter-dependent rational corners."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult
from .parameter_conditions import rational_corner_stratification
from .stratification import AsymptoticStratification, ParameterStratum

FUNCTIONS = (sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.exp, sp.erf, sp.erfc)


def analytic_corner_stratification(expr, variables, target, domain, assumptions):
    """Compose a rational corner with an analytic function using attained values.

    Positive parameters use the uniform zero bound. On the other cells,
    normalize the rational witnesses to +1/2 and -1/2 so periodic outer
    functions cannot collapse the selected limits or introduce nearby poles.
    """
    if (
        not isinstance(expr, sp.Expr)
        or expr.func not in FUNCTIONS
        or len(expr.args) != 1
    ):
        return None
    inner = expr.args[0]
    certified = rational_corner_stratification(
        inner, variables, target, domain, assumptions
    )
    if certified is None:
        return None
    a = next(iter(inner.free_symbols - set(variables)))
    if a.is_real is not True:
        return None
    match = next(
        (
            (x, y, q)
            for x, y in (variables, variables[::-1])
            for q in range(1, 5)
            if inner == x * y**2 / (x ** (2 * q) + a * y**2)
        ),
        None,
    )
    if match is None:
        return None
    x, y, q = match
    outer = expr.func
    axis_value = outer(0)
    j = sp.Dummy("attained_analytic_corner_index", positive=True, integer=True)
    # At negative a the coefficient -1/a makes the inner limit exactly -1/2.
    y_curve = sp.Piecewise(
        (j ** (-(2 * q - 1)) / sp.sqrt(2), sp.Eq(a, 0)),
        (j ** (-2 * q) * (1 - j ** (-2) / a) / sp.sqrt(-a), True),
    )
    curve_value = sp.Piecewise(
        (outer(sp.Rational(1, 2)), sp.Eq(a, 0)),
        (outer(-sp.Rational(1, 2)), True),
    )
    positive = SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.PROVED,
        axis_value,
        (
            LimitEvidence(
                "analytic_rational_corner_bound",
                "For a>0 the inner rational expression is bounded in modulus by "
                "abs(x)/a and tends uniformly to zero. The outer function is "
                "analytic at zero, so its limit is its value there.",
                value=axis_value,
            ),
        ),
        variables,
        domain,
    )
    nonpositive = SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "analytic_corner_axis_subsequence",
                "At x=1/j**2,y=0 the denominator is positive and the inner value is zero.",
                ((x, j ** (-2)), (y, 0)),
                axis_value,
            ),
            LimitEvidence(
                "analytic_corner_pole_free_subsequence",
                "At a=0 the inner ratio is exactly 1/2. At a<0, set "
                "y=x**q*(1-x/a)/sqrt(-a); its denominator is "
                "x**(2*q)*(1-(1-x/a)**2), nonzero for every x>0, "
                "and the inner limit is -1/2. Eventually the inner argument "
                "lies within 1/4 of -1/2. Every admitted outer function is "
                "analytic there; tan has no pole since abs(argument)<3/4<pi/2. "
                "Their values at either 1/2 or -1/2 differ from their value "
                "at zero, giving two attained limits on each nonpositive cell.",
                ((x, j ** (-2)), (y, y_curve)),
                curve_value,
            ),
        ),
        variables,
        domain,
    )
    return AsymptoticStratification(
        (a,),
        (ParameterStratum(a > 0, positive), ParameterStratum(a <= 0, nonpositive)),
        exhaustive=True,
    )
