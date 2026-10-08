import sympy as sp

from asymptotic.coverage import CoverageStatus
from asymptotic.multivariate_limits_advanced import vector_cluster_set


def test_direction_map_joint_cluster_cartesian_square():
    x, y = sp.symbols("x y", real=True)
    r = sp.sqrt(x * x + y * y)
    out = vector_cluster_set((x / r, y / r), (x, y), (0, 0))
    assert out.certified and out.coverage.status is CoverageStatus.COMPLETE
    assert "Eq(" in str(out.cluster_set) and "**2" in str(out.cluster_set)


def test_squared_direction_joint_cluster_preserves_sum_constraint():
    x, y = sp.symbols("x y", real=True)
    d = x * x + y * y
    out = vector_cluster_set((x * x / d, y * y / d), (x, y), (0, 0))
    assert out.certified and out.coverage.status is CoverageStatus.COMPLETE
    assert "1 -" in str(out.cluster_set)


def test_shared_periodic_phase_is_circle_not_product_of_intervals():
    x, y = sp.symbols("x y", real=True)
    q = 1 / (x * x + y * y)
    out = vector_cluster_set((sp.cos(q), sp.sin(q)), (x, y), (0, 0))
    assert out.certified and out.coverage.status is CoverageStatus.COMPLETE
    assert "**2" in str(out.cluster_set)
