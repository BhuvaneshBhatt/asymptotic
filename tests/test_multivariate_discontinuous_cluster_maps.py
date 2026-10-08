import sympy as sp

from asymptotic.cluster_maps import discontinuous_outer_cluster_set, map_cluster_set
from asymptotic.coverage import CoverageCertificate
from asymptotic.multivariate_limits_advanced import (
    AdvancedLimitStatus,
    ClusterSetResult,
)


def _inner(expr, variables, target, cluster_set, certified=True):
    return ClusterSetResult(
        expr,
        variables,
        target,
        cluster_set,
        cluster_set.inf if cluster_set is not sp.S.EmptySet else None,
        cluster_set.sup if cluster_set is not sp.S.EmptySet else None,
        AdvancedLimitStatus.CERTIFIED if certified else AdvancedLimitStatus.UNKNOWN,
        "test_complete_cluster",
        "test fixture",
        coverage=(
            CoverageCertificate.complete(
                "test_cover", "fixture covers all approaches", ("all",)
            )
            if certified
            else None
        ),
    )


def test_sign_maps_interval_across_discontinuity():
    assert map_cluster_set(sp.sign, sp.Interval(-2, 3)) == sp.FiniteSet(-1, 0, 1)


def test_fractional_part_crossing_integer_closes_to_unit_interval():
    assert map_cluster_set(
        sp.frac, sp.Interval(sp.Rational(1, 2), sp.Rational(3, 2))
    ) == sp.Interval(0, 1)


def test_floor_maps_compact_interval_to_integer_range():
    assert map_cluster_set(
        sp.floor, sp.Interval(sp.Rational(1, 2), sp.Rational(7, 2))
    ) == sp.Range(0, 4)


def test_complete_inner_cluster_is_required():
    x, y = sp.symbols("x y", real=True)
    inner = _inner(x / y, (x, y), (0, 0), sp.Interval(-1, 1), certified=False)
    result = discontinuous_outer_cluster_set(
        sp.sign(x / y), (x, y), (0, 0), inner_cluster_result=inner
    )
    assert not result.certified


def test_discontinuous_outer_result_uses_cluster_semantics():
    x, y = sp.symbols("x y", real=True)
    inner = _inner(x / y, (x, y), (0, 0), sp.Interval(-1, 1))
    result = discontinuous_outer_cluster_set(
        sp.sign(x / y), (x, y), (0, 0), inner_cluster_result=inner
    )
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(-1, 0, 1)
