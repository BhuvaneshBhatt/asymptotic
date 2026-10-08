import sympy as sp

from asymptotic.multivariate_geometry_extended import (
    compactified_valuation_atlas,
    complex_newton_geometry,
    periodic_phase_cluster_geometry,
    phase_geometry,
)
from asymptotic.multivariate_limits_advanced import complex_multivariate_limit


def test_two_independent_phases_give_two_torus_factors():
    x, y = sp.symbols("x y", real=True)
    g = phase_geometry(
        (sp.sin(1 / x**2), sp.cos(1 / x**2), sp.sin(1 / y**2), sp.cos(1 / y**2)),
        (x, y),
        (0, 0),
    )
    assert g.certified and len(g.independent_phases) == 2
    assert len(g.phase_variables) == 2


def test_phase_difference_relation_is_recorded():
    x, y = sp.symbols("x y", real=True)
    p = 1 / (x**2 + y**2)
    g = phase_geometry((sp.sin(p), sp.cos(p + 1)), (x, y), (0, 0))
    assert g.certified and len(g.independent_phases) == 1
    assert g.relations and sp.simplify(abs(g.relations[0].offset)) == 1


def test_compactified_atlas_attaches_newton_fan_to_signed_ends():
    x = sp.symbols("x", real=True)
    a = compactified_valuation_atlas((x / (1 + x**2),), (x,), ((sp.oo,), (-sp.oo,)))
    assert len(a.charts) == 2
    assert {c.end_signature for c in a.charts} == {(1,), (-1,)}
    assert all(c.fan is not None for c in a.charts)


def test_complex_newton_detects_direction_dependence_z_over_w():
    z, w = sp.symbols("z w")
    g = complex_newton_geometry(z / w, (z, w), (0, 0))
    assert g.classification in {"mixed_divisor_orders", "newton_direction_dependent"}
    assert g.valuations


def test_complex_newton_balance_for_cusp_quotient():
    z, w = sp.symbols("z w")
    g = complex_newton_geometry(z**2 / (z**2 + w**3), (z, w), (0, 0))
    assert g.valuations
    assert any(v.weight == (3, 2) and v.order == 0 for v in g.valuations)
    assert g.classification in {"newton_direction_dependent", "mixed_divisor_orders"}


def test_phase_bundle_keeps_amplitude_and_two_phase_correlation():
    x, y = sp.symbols("x y", real=True)
    p = 1 / x**2
    q = 1 / y**2
    r = periodic_phase_cluster_geometry(
        ((2 + x) * sp.sin(p), sp.cos(p), sp.sin(q), sp.cos(q)), (x, y), (0, 0)
    )
    assert r is not None and len(r[1].independent_phases) == 2


def test_complex_limit_uses_newton_removable_geometry():
    z, w = sp.symbols("z w")
    # common monomial divisor cancels intrinsically
    r = complex_multivariate_limit(z * w / (z * w), (z, w), (0, 0), return_result=True)
    assert r.certified and r.value == 1
