import sympy as sp

from asymptotic.limits import LimitStatus, limit
from asymptotic.probability import _CONTINUOUS_ROUTES
from asymptotic.sums import _SUM_ROUTE_REGISTRY, sum
from asymptotic.theorem_registry import CertificationStrength, Theorem, TheoremRegistry


def test_registry_priority_preserves_route_order_before_strength():
    registry = TheoremRegistry(
        [
            Theorem(
                "early",
                "x",
                lambda _: True,
                lambda _: (),
                lambda _: 1,
                CertificationStrength.FORMAL,
                priority=0,
            ),
            Theorem(
                "late",
                "x",
                lambda _: True,
                lambda _: (),
                lambda _: 2,
                CertificationStrength.EXACT,
                priority=1,
            ),
        ]
    )
    assert registry.select(object()).theorem.name == "early"


def test_sum_routes_are_registered_in_public_dispatch_order():
    assert [t.name for t in _SUM_ROUTE_REGISTRY.family("sum")][:3] == [
        "sum:exact",
        "sum:series",
        "sum:summation-by-parts",
    ]
    k, n = sp.symbols("k n", integer=True, positive=True)
    result = sum(k, k, 1, n, parameter=n)
    assert result.status == "EXACT"


def test_continuous_probability_routes_are_registry_driven():
    assert [t.name for t in _CONTINUOUS_ROUTES.family("probability-continuous")] == [
        "probability:exact-density",
        "probability:moving-domain-laplace",
        "probability:laplace",
    ]


def test_piecewise_exceptional_target_does_punctured_germ():
    x, y = sp.symbols("x y", real=True)
    expr = sp.Piecewise((x**2 + y**2, sp.Ne(x**2 + y**2, 0)), (1, True))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_cusp_domain_uses_uniform_ratio_squeeze():
    x, y = sp.symbols("x y", real=True)
    domain = sp.And(sp.Abs(x) <= y**2, y > 0)
    result = limit(x / y, (x, y), (0, 0), domain=domain, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    assert result.evidence[-1].method == "domain_ratio_squeeze"


def test_infinite_target_dispatch_uses_projective_chart():
    x, y = sp.symbols("x y", real=True)
    result = limit(1 / x + 1 / y, (x, y), (sp.oo, sp.oo), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0


def test_complex_log_rank_two_origin_has_direction_dependent_phase():
    x, y = sp.symbols("x y", real=True)
    result = limit(sp.log(x + sp.I * y), (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert result.evidence[-1].method == "complex_log_phase_conflict"
