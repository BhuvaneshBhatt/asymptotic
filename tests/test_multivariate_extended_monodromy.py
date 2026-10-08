import sympy as sp

from asymptotic.branch_monodromy import ExtendedClusterKind, classify_monodromy_cluster
from asymptotic.complex_cluster_geometry import branched_blowup_atlas


def divisor(expr):
    x, y = sp.symbols("x y", real=True)
    charts, cov = branched_blowup_atlas(expr, (x, y), (0, 0))
    assert cov.certified
    return charts[0].divisors


def test_log_is_infinite_discrete_translate():
    x, y = sp.symbols("x y", real=True)
    z = x + sp.I * y
    e = sp.log(z)
    c = classify_monodromy_cluster(e, divisor(e), branch_point=True)
    assert c.kind == ExtendedClusterKind.INFINITE_DISCRETE


def test_rational_power_is_finite_cyclic_orbit():
    x, y = sp.symbols("x y", real=True)
    z = x + sp.I * y
    e = z ** sp.Rational(1, 2)
    c = classify_monodromy_cluster(e, divisor(e))
    assert c.kind == ExtendedClusterKind.FINITE_CYCLIC


def test_irrational_power_is_dense_phase_orbit():
    x, y = sp.symbols("x y", real=True)
    z = x + sp.I * y
    e = z ** sp.sqrt(2)
    c = classify_monodromy_cluster(e, divisor(e))
    assert c.kind == ExtendedClusterKind.DENSE_PHASE
