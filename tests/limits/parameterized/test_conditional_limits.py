import sympy as sp

from asymptotic.conditional import ConditionalExpression
from asymptotic.limits import LimitStatus, limit


def test_parameter_weighted_limit_returns_by_default():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    value = limit(expr, (x, y), (0, 0))
    assert isinstance(value, ConditionalExpression)
    assert value.value == 0
    assert value.condition == (p < 19)


def test_parameter_weighted_limit_exposes_evidence_in_result_mode():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    from asymptotic.stratification import AsymptoticStratification

    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert isinstance(result, AsymptoticStratification)
    proved = [s for s in result.strata if s.result.status is LimitStatus.PROVED]
    assert len(proved) == 1
    assert proved[0].result.value == 0
    assert proved[0].condition == (p < 19)
    assert result.mathematical_value == ConditionalExpression(0, p < 19)


def test_true_condition_collapses_to_ordinary_value():
    from asymptotic.conditional import conditional_expression

    assert conditional_expression(sp.Integer(3), sp.S.true) == 3


def test_complete_multiple_strata_render_as_piecewise():
    from asymptotic.conditional import stratification_expression
    from asymptotic.limits import SimultaneousLimitResult
    from asymptotic.stratification import AsymptoticStratification, ParameterStratum

    p = sp.symbols("p", real=True)
    x = sp.symbols("x", real=True)
    low = SimultaneousLimitResult(x, (x,), (0,), LimitStatus.PROVED, 0)
    high = SimultaneousLimitResult(x, (x,), (0,), LimitStatus.PROVED, sp.oo)
    strat = AsymptoticStratification(
        (p,),
        (ParameterStratum(p < 1, low), ParameterStratum(p >= 1, high)),
        exhaustive=True,
    )
    value = stratification_expression(strat)
    assert isinstance(value, sp.Piecewise)
    assert sp.simplify(value.subs(p, 0)) == 0
    assert value.subs(p, 2) == sp.oo
