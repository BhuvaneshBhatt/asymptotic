import sympy as sp

from asymptotic.cluster_semantics import ClusterLimitStatus
from asymptotic.parameter_clusters import parameter_cluster_strata
from asymptotic.projective_clusters import rational_projective_cluster_set


def test_general_rational_projective_range_disconnected_image():
    x, y = sp.symbols("x y", real=True)
    result = rational_projective_cluster_set(x**2 + y**2, x**2 - y**2, (x, y))
    assert result.certified
    expected = sp.Union(sp.Interval(-sp.oo, -1), sp.Interval(1, sp.oo))
    assert result.cluster_set == expected
    assert len(result.components) == 2


def test_general_projective_range_is_reciprocal_difference():
    x, y = sp.symbols("x y", real=True)
    result = rational_projective_cluster_set(x * y, x**2 - y**2, (x, y))
    assert result.certified
    assert result.cluster_set == sp.S.Reals


def test_higher_degree_projective_divisors_component_union():
    x, y = sp.symbols("x y", real=True)
    result = rational_projective_cluster_set(
        x**4 + y**4, (x**2 - y**2) * (x**2 - 4 * y**2), (x, y)
    )
    assert result.certified
    assert isinstance(result.cluster_set, sp.Union)
    assert len(result.components) == 2
    assert result.cluster_set.inf is -sp.oo
    assert result.cluster_set.sup is sp.oo


def test_parameter_cluster_projective_decomposition():
    x, y, p = sp.symbols("x y p", real=True)
    expr = 1 / ((x**2 + y**2) ** p * (x**2 - y**2))
    strata = parameter_cluster_strata(expr, (x, y), (0, 0), sp.Eq(p, -1))
    assert len(strata) == 1
    cluster = strata[0].cluster_result
    assert cluster.provider == "rational_projective_range_decomposition"
    assert cluster.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST
