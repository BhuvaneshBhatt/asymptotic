import sympy as sp

from asymptotic.cluster_semantics import ClusterLimitStatus
from asymptotic.complex_cluster_geometry import (
    branch_divisor_atlas,
    complex_branch_cluster_set,
)


def test_nested_log_propagates_inner_branch_sides():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(sp.sin(sp.log(z)), z, -1)
    assert result.certified
    assert len(result.cluster_set) == 2
    assert result.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST
    assert result.divisors


def test_nested_power_of_log_uses_composed_branch_values():
    z = sp.Symbol("z")
    expr = sp.sqrt(sp.log(z))
    result = complex_branch_cluster_set(expr, z, -1)
    assert result.certified
    assert len(result.cluster_set) == 2
    assert result.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST


def test_branch_charts_are_attached_to_common_blowup_geometry():
    z = sp.Symbol("z")
    charts, divisors, coverage = branch_divisor_atlas(sp.log(z), z, -2)
    assert coverage.certified
    assert len(charts) == 2
    assert charts[0].blowup_chart.provider == "complex_branch_blowup"
    assert charts[0].transitions
    assert divisors[0].kind == "log"


def test_analytic_outer_function_can_collapse_monodromy():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(sp.cos(sp.log(z)), z, -1)
    assert result.certified
    # cos(i*pi) is the same from both sides
    assert len(result.cluster_set) == 1
