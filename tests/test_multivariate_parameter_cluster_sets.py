import sympy as sp

from asymptotic.cluster_semantics import ClusterLimitStatus
from asymptotic.limits import LimitStatus, limit
from asymptotic.parameter_clusters import parameter_cluster_strata
from asymptotic.stratification import AsymptoticStratification


def test_supercritical_radial_power_has_cluster_set():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), p > 19)
    assert len(strata) == 1
    cluster = strata[0].cluster_result
    assert cluster.certified
    assert cluster.cluster_set == sp.Interval(0, sp.oo)
    assert cluster.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST


def test_parameter_limit_stratification_has_supercritical_regime():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert isinstance(result, AsymptoticStratification)
    assert result.exhaustive
    by_condition = {s.condition: s.result.status for s in result.strata}
    assert by_condition[p < 19] is LimitStatus.PROVED
    assert by_condition[p > 19] is LimitStatus.DOES_NOT_EXIST
    boundary = [s for s in result.strata if s.condition not in (p < 19, p > 19)]
    assert len(boundary) == 1
    assert boundary[0].result.status is LimitStatus.DOES_NOT_EXIST
    assert all(s.result.status is not LimitStatus.UNKNOWN for s in result.strata)


def test_parameter_cluster_family_certifies_sign_indefinite_numerator():
    x, y, p = sp.symbols("x y p", real=True)
    expr = (x**2 - y**2) / (x**2 + y**2) ** p
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), p > 1)
    assert len(strata) == 1
    assert strata[0].cluster_result.cluster_set == sp.S.Reals


def test_parameter_cluster_family_refuses_relative_domain():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**2 * y**2 / (x**2 + y**2) ** p
    assert not parameter_cluster_strata(expr, (x, y), (1, 0), p > 2)
    assert not parameter_cluster_strata(expr, (x, y), (0, 0), p > 2, domain=x > 0)
