"""Regression tests for package-native symbolic instrumentation."""

import sympy as sp

from asymptotic._symbolic_policy import (
    SymbolicPolicy,
    bounded_assumption_entails,
    bounded_limit,
    bounded_primitive,
    bounded_solve_one,
)
from asymptotic.instrumentation import SymbolicMetrics, symbolic_metrics


def test_symbolic_metrics_count_general_fallbacks_without_runtime_patching():
    x = sp.symbols("x", positive=True)
    with symbolic_metrics() as metrics:
        assert bounded_limit(sp.sin(x) / x, x, sp.oo) == 0
        roots = bounded_solve_one(sp.exp(x) - 2, x, allow_general=True)
        assert roots == (sp.log(2),)
        primitive = bounded_primitive(sp.sin(x), x, allow_general=True)
        assert primitive == -sp.cos(x)

    assert metrics.limit_calls == 1
    assert metrics.general_limit_calls == 1
    assert metrics.solve_one_calls == 1
    assert metrics.general_solve_calls == 1
    assert metrics.primitive_calls == 1
    assert metrics.general_integrate_calls == 1


def test_symbolic_metrics_record_budget_declines():
    x = sp.symbols("x")
    policy = SymbolicPolicy(limit_ops=0, solve_ops=0, assumption_ops=0)
    expr = sum(sp.sin((i + 1) * x) for i in range(5))
    with symbolic_metrics() as metrics:
        assert bounded_limit(expr, x, sp.oo, policy=policy) is None
        assert bounded_solve_one(expr, x, policy=policy, allow_general=True) is None
        assert bounded_assumption_entails(sp.Eq(expr, 0), policy=policy) is None
    assert metrics.declined_by_budget >= 3
    assert metrics.general_limit_calls == 0
    assert metrics.general_solve_calls == 0


def test_nested_symbolic_metric_contexts_are_independent():
    x = sp.symbols("x", positive=True)
    with symbolic_metrics() as outer:
        bounded_limit(1 / x, x, sp.oo)
        with symbolic_metrics() as inner:
            bounded_limit(sp.sin(x) / x, x, sp.oo)
        bounded_limit(1 / x**2, x, sp.oo)

    assert isinstance(outer, SymbolicMetrics)
    assert outer.limit_calls == 2
    assert inner.limit_calls == 1


def test_symbolic_metrics_include_domain_specific_structural_counts():
    from asymptotic import AsymptoticRemainder
    from asymptotic.multivariate import multivariate_scaling_regimes
    from asymptotic.parameter_auto import automatic_parameter_stratification
    from asymptotic.remainder_theorems import certify_reciprocal_remainder

    x, z, y = sp.symbols("x z y", positive=True)
    a = sp.symbols("a", real=True)
    xr = sp.symbols("xr", real=True)
    with symbolic_metrics() as metrics:
        automatic_parameter_stratification(
            (a * x,),
            lambda condition: condition,
            parameters=(a,),
        )
        multivariate_scaling_regimes(y**2 - x - z**2, y, (x, z))
        certify_reciprocal_remainder(
            sp.sin(xr),
            AsymptoticRemainder.exact_zero(xr, sp.oo),
        )

    assert metrics.parameter_strata >= 1
    assert metrics.newton_cones_generated >= 1
    assert metrics.unknown_remainders >= 1
