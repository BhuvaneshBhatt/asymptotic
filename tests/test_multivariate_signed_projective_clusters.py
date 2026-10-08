import sympy as sp

from asymptotic.cluster_semantics import ClusterLimitStatus
from asymptotic.limits import LimitStatus, limit
from asymptotic.parameter_clusters import parameter_cluster_strata
from asymptotic.stratification import AsymptoticStratification


def test_signed_supercritical_cluster_set_is_two_sided_unbounded():
    x, y, p = sp.symbols("x y p", real=True)
    expr = (x**2 - y**2) / (x**2 + y**2) ** p
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), p > 1)
    assert len(strata) == 1
    cluster = strata[0].cluster_result
    assert cluster.certified
    assert cluster.cluster_set == sp.S.Reals
    assert cluster.liminf == -sp.oo
    assert cluster.limsup == sp.oo
    assert cluster.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST


def test_negative_supercritical_cluster_set_is_negative_half_line():
    x, y, p = sp.symbols("x y p", real=True)
    expr = -((x * y) ** 2) / (x**2 + y**2) ** p
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), p > 2)
    cluster = strata[0].cluster_result
    assert cluster.cluster_set == sp.Interval(-sp.oo, 0)
    assert cluster.liminf == -sp.oo
    assert cluster.limsup == 0


def test_critical_diagonal_quadratic_has_parameter_dependent_extrema():
    x, y, p, q = sp.symbols("x y p q", real=True)
    expr = (q * x**2 + y**2) / (x**2 + y**2) ** p
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), sp.Eq(p, 1))
    assert len(strata) == 3
    by_q = {
        sp.simplify_logic(s.condition): s.cluster_result.cluster_set for s in strata
    }
    assert any(v == sp.Interval(q, 1) for k, v in by_q.items() if k.has(q < 1))
    assert any(v == sp.FiniteSet(1) for k, v in by_q.items() if k.has(sp.Eq(q, 1)))
    assert any(v == sp.Interval(1, q) for k, v in by_q.items() if k.has(q > 1))


def test_critical_projective_pole_has_disconnected_cluster_set():
    x, y, p = sp.symbols("x y p", real=True)
    expr = 1 / ((x**2 + y**2) ** p * (x**2 - y**2))
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), sp.Eq(p, -1))
    assert len(strata) == 1
    cluster = strata[0].cluster_result
    expected = sp.Union(sp.Interval(-sp.oo, -1), sp.Interval(1, sp.oo))
    assert cluster.cluster_set == expected
    assert cluster.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST


def test_parameter_pipeline_uses_signed_cluster_geometry():
    x, y, p = sp.symbols("x y p", real=True)
    expr = (x**2 - y**2) / (x**2 + y**2) ** p
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert isinstance(result, AsymptoticStratification)
    supercritical = [s for s in result.strata if s.condition == (p > 1)]
    assert len(supercritical) == 1
    assert supercritical[0].result.status is LimitStatus.DOES_NOT_EXIST


def test_parameter_cluster_refuses_nonhomogeneous_projective_geometry():
    x, y, p = sp.symbols("x y p", real=True)
    expr = (1 + x**2 - y**2) / (x**2 + y**2) ** p
    assert not parameter_cluster_strata(expr, (x, y), (0, 0), p > 1)


def test_supercritical_quadratic_refines_second_parameter_sign():
    x, y, p, q = sp.symbols("x y p q", real=True)
    expr = (q * x**2 + y**2) / (x**2 + y**2) ** p
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), p > 1)
    assert len(strata) == 3
    sets = {s.cluster_result.cluster_set for s in strata}
    assert sp.S.Reals in sets
    assert sp.Interval(0, sp.oo) in sets
    assert sp.FiniteSet(sp.oo) in sets


def test_parameter_pipeline_refines_critical_angular_parameter():
    x, y, p, q = sp.symbols("x y p q", real=True)
    expr = (q * x**2 + y**2) / (x**2 + y**2) ** p
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert isinstance(result, AsymptoticStratification)
    critical = [
        s
        for s in result.strata
        if sp.simplify(s.condition.subs(p, 1)) is not sp.S.false
    ]
    assert any(
        s.result.status is LimitStatus.PROVED and s.result.value == 1 for s in critical
    )
    assert any(s.result.status is LimitStatus.DOES_NOT_EXIST for s in critical)
