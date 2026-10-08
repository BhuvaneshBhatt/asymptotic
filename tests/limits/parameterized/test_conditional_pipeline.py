import sympy as sp

from asymptotic.conditional import ConditionalExpression, mathematical_result
from asymptotic.limits import LimitStatus, SimultaneousLimitResult, limit
from asymptotic.parameter_conditions import conditional_limit_stratification
from asymptotic.stratification import AsymptoticStratification, ParameterStratum


def test_parameter_conditioned_limit_uses_structured_mode():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    structured = limit(expr, (x, y), (0, 0), return_result=True)
    assert isinstance(structured, AsymptoticStratification)
    assert structured.parameters == (p,)
    assert structured.mathematical_value == ConditionalExpression(0, p < 19)
    assert limit(expr, (x, y), (0, 0)) == structured.mathematical_value


def test_universal_renderer_accepts_any_proved_result_strata():
    p = sp.symbols("p", real=True)
    x = sp.symbols("x", real=True)
    low = SimultaneousLimitResult(x, (x,), (0,), LimitStatus.PROVED, 2)
    high = SimultaneousLimitResult(x, (x,), (0,), LimitStatus.PROVED, 5)
    strat = AsymptoticStratification(
        (p,),
        (ParameterStratum(p < 0, low), ParameterStratum(p >= 0, high)),
        exhaustive=True,
    )
    value = mathematical_result(strat)
    assert isinstance(value, sp.Piecewise)
    assert value.subs(p, -1) == 2
    assert value.subs(p, 1) == 5


def test_universal_renderer_accepts_plain_symbolic_strata():
    p = sp.symbols("p", real=True)
    strat = AsymptoticStratification(
        (p,),
        (ParameterStratum(p != 0, sp.Symbol("a")),),
        exhaustive=False,
    )
    assert mathematical_result(strat) == ConditionalExpression(sp.Symbol("a"), p != 0)


def test_limit_frontend_has_no_weighted_order_dependency():
    import importlib
    import inspect

    engine = importlib.import_module("asymptotic._limit_engine")
    source = inspect.getsource(engine._limit_impl)
    assert "parameter_weighted_order_certificate" not in source
    assert "conditional_limit_stratification" in source


def test_discovery_api_returns_first_class_stratification():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    strat = conditional_limit_stratification(expr, (x, y), (0, 0))
    assert isinstance(strat, AsymptoticStratification)
    assert any(s.condition == (p < 19) for s in strat.strata)
