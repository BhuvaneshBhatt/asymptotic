"""Semantic certificates apply only to registered function definitions."""

import pytest
import sympy as sp

from asymptotic import complex_limit, limit
from asymptotic.function_normalization import normalize_functions
from asymptotic.limit_models import LimitStatus
from asymptotic.parameter_special_germs import special_parameter_limit
from asymptotic.rational_pullback_germs import rounded_tail_certificate
from asymptotic.reference_normalization import Normal, Series, StepFactorialPower

x, h, n = sp.symbols("x h n")


@pytest.mark.parametrize(
    "name,args,variable",
    [
        ("pole_sensitive_step_factorial_power", (x, -3, h), h),
        ("analytic_step_factorial_power", (x, n, h), n),
        ("StepFactorialPower", (x, -3, h), h),
    ],
)
def test_undefined_factorial(name, args, variable):
    expression = sp.Function(name)(*args)
    assert (
        special_parameter_limit(expression, variable, sp.S.Zero, sp.S.true, sp.S.true)
        is None
    )
    assert (
        limit(expression, variable, 0, return_result=True).status is LimitStatus.UNKNOWN
    )
    assert normalize_functions(expression, (variable,), (sp.S.Zero,))[0] == expression


@pytest.mark.parametrize("fake_outer,fake_inner", [(True, False), (False, True)])
def test_undefined_series_wrappers(fake_outer, fake_inner):
    outer = sp.Function("Normal") if fake_outer else Normal
    inner = sp.Function("Series") if fake_inner else Series
    expression = outer(inner(StepFactorialPower(x, n, 1), sp.Tuple(x, 0, 3)))
    assert (
        special_parameter_limit(expression, n, sp.Integer(2), sp.S.true, sp.S.true)
        is None
    )
    assert limit(expression, n, 2, return_result=True).status is LimitStatus.UNKNOWN


def test_undefined_integral():
    expression = sp.Function("Integrate")(sp.exp(x), sp.Tuple(x, 0, 1))
    assert normalize_functions(expression, (h,), (sp.S.Zero,))[0] == expression


def test_undefined_rounding():
    positive = sp.Symbol("positive", positive=True)
    expression = sp.Function("Round")(positive) / positive
    assert rounded_tail_certificate(expression, positive, sp.oo) is None
    assert (
        limit(expression, positive, sp.oo, return_result=True).status
        is LimitStatus.UNKNOWN
    )


@pytest.mark.parametrize("variable", [None, (x,), sp.Integer(1), x + h])
def test_complex_variable_validation(variable):
    with pytest.raises(TypeError, match="limit variable must be a symbol"):
        complex_limit(x, variable)


def test_falling_factorial_branch_boundary():
    expression = Normal(Series(sp.ff(x, n), sp.Tuple(x, 0, 3)))
    assert (
        special_parameter_limit(expression, n, sp.S.Zero, sp.S.true, sp.S.true) is None
    )
    assert limit(expression, n, 0, return_result=True).status is LimitStatus.UNKNOWN
    # For a noninteger order the two base-side continuations disagree at zero.
    positive = sp.gamma(x + 1) / sp.gamma(x + sp.Rational(1, 2))
    negative = sp.I * sp.gamma(sp.Rational(1, 2) - x) / sp.gamma(-x)
    assert sp.limit(positive, x, 0, dir="+") == 1 / sp.sqrt(sp.pi)
    assert sp.limit(negative, x, 0, dir="-") == 0
