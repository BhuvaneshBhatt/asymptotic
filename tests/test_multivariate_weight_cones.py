import sympy as sp

from asymptotic.multivariate import (
    multivariate_scaling_regimes,
    newton_polyhedron_terms,
)
from asymptotic.stratification import AsymptoticStratification


def test_newton_support_extracts_multivariate_points():
    x, z, y = sp.symbols("x z y")
    terms = newton_polyhedron_terms(y**2 - x - z**2, y, (x, z))
    assert {term.support_point for term in terms} == {
        (0, 0, 2),
        (1, 0, 0),
        (0, 2, 0),
    }


def test_automatic_weight_cones_find_both_chambers_and_wall():
    x, z, y = sp.symbols("x z y")
    regimes = multivariate_scaling_regimes(y**2 - x - z**2, y, (x, z))
    assert len(regimes) == 3

    active = {frozenset(term.expression for term in r.active_terms): r for r in regimes}
    x_face = active[frozenset((y**2, -x))]
    z_face = active[frozenset((y**2, -(z**2)))]
    wall = active[frozenset((y**2, -x, -(z**2)))]

    assert x_face.cone.contains((1, 1)) is True
    assert z_face.cone.contains((4, 1)) is True
    assert wall.cone.contains((2, 1)) is True
    assert set(wall.balances[0].coefficients) == {-sp.sqrt(2), sp.sqrt(2)}


def test_discovered_regime_representatives_reproduce_path_balances():
    x, z, y = sp.symbols("x z y")
    regimes = multivariate_scaling_regimes(y**2 - x - z**2, y, (x, z))
    for regime in regimes:
        assert regime.balances
        assert regime.cone.contains(regime.representative_weights) is True


def test_weight_cone_discovery_automatically_stratifies_deleted_faces():
    x, z, y, a = sp.symbols("x z y a")
    result = multivariate_scaling_regimes(y**2 - a * x - z**2, y, (x, z))
    assert isinstance(result, AsymptoticStratification)
    zero = result.select(sp.Eq(a, 0))
    nonzero = result.select(sp.Ne(a, 0))
    assert zero is not None and nonzero is not None
    assert all(-a * x not in {t.expression for t in r.active_terms} for r in zero.result)
    assert any(any(t.coefficient == -a for t in r.active_terms) for r in nonzero.result)
