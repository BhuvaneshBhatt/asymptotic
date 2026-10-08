import sympy as sp

from asymptotic.limit_planner import (
    limit_metrics,
    measure_limit_stage,
    plan_limit,
    reset_limit_metrics,
)
from asymptotic.limits import limit

x, y = sp.symbols("x y", real=True)


def test_planner_prefers_exceptional_paths_for_rational_germ():
    plan = plan_limit(x**2 / (4 * y - 5 * x), (x, y), (0, 0))
    assert plan.rational
    assert plan.stages.index("exceptional_paths") < plan.stages.index("newton_fan")
    assert plan.stages.index("newton_fan") < plan.stages.index("semialgebraic")


def test_planner_prefers_expansion_for_singular_transcendental_germ():
    plan = plan_limit(sp.log(x) + sp.sin(y) / x, (x, y), (0, 0))
    assert plan.singular and plan.transcendental
    assert plan.stages.index("local_expansion") < plan.stages.index("newton_fan")


def test_limit_metrics_are_observational():
    reset_limit_metrics()
    with measure_limit_stage("probe"):
        pass
    metrics = limit_metrics()
    assert metrics["counts"]["probe"] == 1
    assert metrics["seconds"]["probe"] >= 0


def test_rational_positive_certificate_precedes_path_search():
    x, y = sp.symbols("x y", real=True)
    result = limit(
        (x**3 + x**2 * y**2 + 2 * x**2 * y + x * y**2)
        / (x**2 * y**2 + 2 * x**2 + 2 * x * y + y**2),
        (x, y),
        (0, 0),
        return_result=True,
    )
    assert result.status.name == "PROVED"
    assert result.value == 0
    assert result.evidence[0].method == "candidate_rational"


def test_exceptional_curve_must_be_real_and_local():
    x, y = sp.symbols("x y", real=True)
    result = limit(
        (x**2 + y**4) / (-(x**4) / 3 + x**2 + y**4),
        (x, y),
        (0, 0),
        return_result=True,
    )
    assert result.status.name == "PROVED"
    assert result.value == 1


def test_high_dimensional_rational_limit_avoids_path_enumeration():
    variables = sp.symbols("x0:6", real=True)
    numerator = sum(v**5 for v in variables)
    denominator = sum(v**2 for v in variables)
    result = limit(
        numerator / denominator,
        variables,
        (0,) * len(variables),
        return_result=True,
    )
    assert result.status.name == "PROVED"
    assert result.value == 0
