"""Focused contracts for multivariate geometric capabilities."""

import sympy as sp

from asymptotic.branch_monodromy import monodromy_cluster
from asymptotic.complex_cluster_geometry import branched_blowup_atlas
from asymptotic.relative_growth import relative_growth_valuation_fan
from asymptotic.resolution_progress import progress_certificate

x, y = sp.symbols("x y", real=True)


def test_branch_winding():
    z = x + sp.I * y
    expr = sp.log(sp.sqrt(z))
    charts, coverage = branched_blowup_atlas(expr, (x, y), (0, 0))
    assert len(charts) == 2
    assert all(chart.divisors for chart in charts)
    assert coverage.certified
    assert coverage.covered_regimes == (0, 1)
    assert coverage.missing_regimes == ()

    family, _finite, state = monodromy_cluster(expr, charts[0].divisors)
    assert state.windings
    assert family is not None


def test_relative_growth_valuation():
    cones, coverage = relative_growth_valuation_fan(
        (x, y), sp.Eq(sp.log(sp.Abs(y)) / sp.log(sp.Abs(x)), 1)
    )
    assert coverage.certified
    assert tuple(cone.weights for cone in cones) == ((1, 1),)


def test_resolution_dimension_drop():
    certificate = progress_certificate(2, 1, x * y, (x, y))
    assert certificate.certified
    assert certificate.child.dimension == 1
    assert certificate.parent.dimension == 2


def test_unsupported_growth_relation():
    cones, coverage = relative_growth_valuation_fan((x, y), sp.Gt(y, x))
    assert cones == ()
    assert not coverage.certified
