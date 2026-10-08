import math

import sympy as sp

from asymptotic import (
    Evidence,
    EvidenceStatus,
)
from asymptotic.perturbation_problem import (
    ConvergenceDiagnostics,
    MethodRecommendation,
    PerturbationCondition,
    PerturbationProblem,
    ValidationReport,
)


def test_algebraic_classification_and_auto_solve():
    x, eps = sp.symbols("x eps")
    problem = PerturbationProblem(x - 1 - eps, x, eps)
    assert problem.kind == "algebraic"
    recommendation = problem.recommend_method()
    assert isinstance(recommendation, MethodRecommendation)
    assert recommendation.method == "regular"
    result = problem.solve(order=1)
    assert sp.expand(result.approximation[0] - (1 + eps)) == 0


def test_symbolic_validation_reports_residual_order():
    x, eps = sp.symbols("x eps")
    result = PerturbationProblem(x - 1 - eps * x, x, eps).solve(order=1)
    report = result.validate()
    assert isinstance(report, ValidationReport)
    assert report.residual_orders == (sp.Integer(2),)
    assert report.evidence.status is EvidenceStatus.PROVED


def test_numerical_convergence_estimates_order():
    x, eps = sp.symbols("x eps")
    result = PerturbationProblem(x - 1 - eps * x, x, eps).solve(order=1)
    report = result.convergence_diagnostics((0.1, 0.05, 0.025), lambda e: 1 / (1 - e))
    assert isinstance(report, ConvergenceDiagnostics)
    assert report.observed_order is not None
    assert math.isclose(report.observed_order, 2.1, rel_tol=0.08)


def test_canonical_derivative_condition():
    t = sp.symbols("t")
    u = sp.Function("u")(t)
    condition = PerturbationCondition(u, 2, 0, derivative_order=1)
    assert condition.kind == "derivative"
    assert condition.residual() == sp.Subs(sp.Derivative(u, t), t, 0) - 2


def test_evidence_combination_keeps_weakest_status():
    proof = Evidence(EvidenceStatus.PROVED, "symbolic")
    numeric = Evidence(EvidenceStatus.NUMERICALLY_SUPPORTED, "numeric")
    combined = Evidence.combine(proof, numeric, statement="combined")
    assert combined.status is EvidenceStatus.NUMERICALLY_SUPPORTED


def test_limit_condition_is_evaluated_on_approximation():
    x = sp.symbols("x", positive=True)
    u = sp.Function("u")(x)
    condition = PerturbationCondition(u, 1, sp.oo, kind="limit", direction="-")
    from asymptotic.perturbation_problem import _condition_on_approximation

    residual = _condition_on_approximation(condition, (1 + sp.exp(-x),), (u,))
    assert residual == 0


def test_periodic_condition_has_canonical_residual():
    t = sp.symbols("t")
    u = sp.Function("u")(t)
    condition = PerturbationCondition(
        u, 0, 0, kind="periodic", reference_point=2 * sp.pi
    )
    assert (
        sp.simplify(condition.residual() - (u.subs(t, 0) - u.subs(t, 2 * sp.pi))) == 0
    )


def test_matching_condition_can_compare_two_profiles():
    x, X = sp.symbols("x X")
    outer = sp.Function("outer")(x)
    inner = sp.Function("inner")(X)
    condition = PerturbationCondition(
        outer,
        0,
        sp.oo,
        kind="matching",
        reference_unknown=inner,
        reference_point=sp.oo,
    )
    assert condition.residual() == sp.Limit(outer, x, sp.oo) - sp.Limit(inner, X, sp.oo)


def test_convergence_reports_expected_symbolic_order():
    x, eps = sp.symbols("x eps")
    result = PerturbationProblem(x - 1 - eps * x, x, eps).solve(order=1)
    report = result.convergence_diagnostics(
        (0.1, 0.05, 0.025, 0.0125), lambda e: 1 / (1 - e)
    )
    assert report.expected_order == 2.0
    assert report.order_difference is not None
    assert abs(report.order_difference) < 0.2


def test_multiple_scales_frontend_returns_physical_time_approximation():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")(t)
    problem = PerturbationProblem(
        sp.diff(y, t, 2) + y + eps * y**3,
        y,
        eps,
    )
    result = problem.solve(method="multiple-scales", order=1)
    native_scales = set(result.native.scales)
    assert not (
        set().union(*(expr.free_symbols for expr in result.approximation))
        & native_scales
    )
    assert t in set().union(*(expr.free_symbols for expr in result.approximation))
