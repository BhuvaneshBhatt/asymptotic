import sympy as sp

from asymptotic.local_strata import (
    LocalStratumProtocol,
    as_local_stratum,
    combine_local_strata,
    frontier_incidence_complex,
)
from asymptotic.multivariate_geometry_extended import (
    compactified_valuation_atlas,
    complex_newton_geometry,
    phase_geometry,
)
from asymptotic.newton_geometry import (
    newton_polyhedral_fan,
    structured_semialgebraic_geometry,
)


def test_all_geometry_subsystems_share_local_stratum_protocol():
    x, y = sp.symbols("x y", real=True)
    fan = newton_polyhedral_fan((x**2 + y**4,), (x, y), (0, 0))
    cone = next(c for c in fan.cones if c.representative_weight is not None)
    assert isinstance(cone, LocalStratumProtocol)

    phase = phase_geometry((sp.sin(1 / (x**2 + y**2)),), (x, y), (0, 0))
    assert isinstance(phase, LocalStratumProtocol)

    atlas = compactified_valuation_atlas((x / (1 + x**2),), (x,), ((sp.oo,),))
    assert isinstance(atlas.charts[0], LocalStratumProtocol)

    z, w = sp.symbols("z w")
    cg = complex_newton_geometry(z / w, (z, w), (0, 0))
    assert cg.valuations and isinstance(cg.valuations[0], LocalStratumProtocol)


def test_recursive_stratum_product_is_subsystem_neutral():
    x, y = sp.symbols("x y", real=True)
    fan = newton_polyhedral_fan((x**2 + y**4,), (x, y), (0, 0))
    cone = next(c for c in fan.cones if c.representative_weight is not None)
    phase = phase_geometry((sp.sin(1 / (x**2 + y**2)),), (x, y), (0, 0))
    node = combine_local_strata(cone, phase, kind="newton_phase_fiber")
    assert node.stratum_kind == "newton_phase_fiber"
    assert len(node.child_strata) == 2
    assert node.stratum_certified
    assert as_local_stratum(cone).valuation_data == cone.representative_weight


def test_frontier_complex_distinguishes_segment_and_endpoints():
    x = sp.symbols("x", real=True)
    region = sp.And(x >= 0, x <= 1)
    complex_ = frontier_incidence_complex(region, (x,))
    assert complex_ is not None
    dims = [s.dimension for s in complex_.strata]
    assert 1 in dims and 0 in dims
    assert sum(d == 0 for d in dims) >= 2
    assert complex_.incidence


def test_structured_cluster_geometry_carries_frontier_complex():
    t = sp.symbols("t", real=True)
    cs = sp.ConditionSet(t, sp.And(t >= 0, t <= 1), sp.S.Reals)
    geometry = structured_semialgebraic_geometry(cs)
    assert geometry is not None
    assert geometry.frontier_complex is not None
    assert {s.dimension for s in geometry.frontier_complex.strata} >= {0, 1}
