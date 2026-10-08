import sympy as sp

from asymptotic.cluster_semantics import ClusterLimitStatus
from asymptotic.complex_cluster_geometry import complex_branch_cluster_set


def test_principal_log_has_two_branch_cut_cluster_values():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(sp.log(z), z, -1)
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(sp.I * sp.pi, -sp.I * sp.pi)
    assert result.limit_semantics.status is ClusterLimitStatus.DOES_NOT_EXIST
    assert result.coverage.certified


def test_principal_square_root_branch_values_at_negative_axis():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(sp.sqrt(z), z, -1)
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(sp.I, -sp.I)


def test_integer_power_collapses_branch_atlas_to_singleton_semantics():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(z**3, z, -2)
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(-8)
    assert result.limit_semantics.status is ClusterLimitStatus.LIMIT


def test_upper_branch_sector_is_singleton():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(sp.log(z), z, -1, domain=sp.im(z) > 0)
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(sp.I * sp.pi)


def test_semialgebraic_relative_domain_intersects_branch_atlas():
    z = sp.Symbol("z")
    result = complex_branch_cluster_set(sp.log(z), z, -1, domain=sp.re(z) < 0)
    assert result.certified
    assert result.cluster_set == sp.FiniteSet(sp.I * sp.pi, -sp.I * sp.pi)
