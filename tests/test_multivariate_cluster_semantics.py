import sympy as sp

from asymptotic.cluster_semantics import ClusterLimitStatus, cluster_limit_semantics
from asymptotic.conditional import ConditionalExpression
from asymptotic.coverage import CoverageCertificate
from asymptotic.limits import LimitStatus, limit
from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    ClusterSetResult,
)
from asymptotic.stratification import AsymptoticStratification


def _cluster(cluster_set, status=AdvancedLimitStatus.CERTIFIED):
    x, y = sp.symbols("x y", real=True)
    return ClusterSetResult(
        x,
        (x, y),
        (0, 0),
        cluster_set,
        None,
        None,
        status,
        "test_complete_atlas",
        "test cluster set",
        coverage=(
            CoverageCertificate.complete(
                "test_cover", "fixture covers all approaches", ("all",)
            )
            if status is AdvancedLimitStatus.CERTIFIED
            else None
        ),
    )


def test_singleton_complete_cluster_set_is_a_limit():
    result = cluster_limit_semantics(_cluster(sp.FiniteSet(3)))
    assert result.status is ClusterLimitStatus.LIMIT
    assert result.value == 3


def test_non_singleton_complete_cluster_set_is_dne():
    result = cluster_limit_semantics(_cluster(sp.Interval(-1, 1)))
    assert result.status is ClusterLimitStatus.DOES_NOT_EXIST
    assert result.value is None


def test_partial_cluster_set_never_proves_dne():
    result = cluster_limit_semantics(
        _cluster(sp.FiniteSet(0, 1), AdvancedLimitStatus.UNKNOWN)
    )
    assert result.status is ClusterLimitStatus.UNKNOWN


def test_parameter_stratification_explicitly_covers_unresolved_complement():
    x, y, p = sp.symbols("x y p", real=True)
    expr = x**16 * y**22 / (x**2 + y**2) ** p
    strat = limit(expr, (x, y), (0, 0), return_result=True)
    assert isinstance(strat, AsymptoticStratification)
    assert strat.exhaustive
    assert len(strat.strata) == 3
    proved = [s for s in strat.strata if s.result.status is LimitStatus.PROVED]
    unknown = [s for s in strat.strata if s.result.status is LimitStatus.UNKNOWN]
    dne = [s for s in strat.strata if s.result.status is LimitStatus.DOES_NOT_EXIST]
    assert len(proved) == 1
    assert not unknown
    assert len(dne) == 2
    assert proved[0].condition == (p < 19)
    assert any(s.condition == (p > 19) for s in dne)
    assert any(sp.simplify(s.condition.subs(p, 19)) is sp.S.true for s in dne)
    assert strat.mathematical_value == ConditionalExpression(0, p < 19)


def test_core_limit_uses_complete_cluster_set_for_angular_dne():
    x, y = sp.symbols("x y", real=True)
    expr = x**2 / (x**2 + y**2)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST


def test_cluster_result_exposes_same_unified_semantics():
    result = _cluster(sp.FiniteSet(7))
    assert result.limit_semantics.status is ClusterLimitStatus.LIMIT
    assert result.limit_semantics.value == 7
