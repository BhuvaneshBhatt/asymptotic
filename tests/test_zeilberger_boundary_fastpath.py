import importlib

import sympy as sp

sums = importlib.import_module("asymptotic.sums")


def test_binomial_finite_support_boundary_is_zero():
    from asymptotic.instrumentation import symbolic_metrics

    n = sp.Symbol("n", nonnegative=True, integer=True)
    k = sp.Symbol("k", nonnegative=True, integer=True)
    expr = -k * sp.binomial(n, k) / (2 * k - 2 * n - 2)
    with symbolic_metrics() as metrics:
        assert sums._boundary_value(expr, k, sp.oo) == 0
    assert metrics.limit_calls == 0


def test_binomial_fastpath_requires_nonnegative_integer_top():
    a = sp.Symbol("a", positive=True)
    k = sp.Symbol("k", integer=True, nonnegative=True)
    calls = []

    def evaluate(expr, variable, point, **kwargs):
        calls.append((expr, variable, point, kwargs))
        return sp.S.Zero

    expr = sp.binomial(a, k)
    assert sums._boundary_value(expr, k, sp.oo, limit_evaluator=evaluate) == 0
    assert calls == [(expr.rewrite(sp.gamma), k, sp.oo, {"allow_general": True})]
