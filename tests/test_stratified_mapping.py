import sympy as sp

from asymptotic.local_strata import LocalStratum
from asymptotic.mapped_strata import StratumMap, mapped_local_stratum
from asymptotic.multivariate_geometry_extended import complex_newton_geometry
from asymptotic.newton_geometry import newton_polyhedral_fan
from asymptotic.stratified_mapping import (
    check_whitney_conditions,
    clear_qe_projection_cache,
    completeness_certificate,
    complex_cluster_geometry,
    phase_bundle_over_stratum,
    qe_projection_cache_stats,
    refine_mapped_stratum,
    unified_valuation_compactification,
)


def test_rank_refinement_splits_cusp_rank_drop_and_rebuilds_frontier():
    t, u = sp.symbols("t u", real=True)
    base = LocalStratum("base", sp.And(t >= -1, t <= 1), 1)
    mapping = StratumMap((t,), (t**2,), (u,))
    mapped = mapped_local_stratum(base, mapping)
    result = refine_mapped_stratum(mapped)
    assert result.certified
    assert {s.rank for s in result.rank_strata} == {0, 1}
    assert result.frontier_complex is not None


def test_completeness_certificate_accounts_recursive_kinds():
    x, y = sp.symbols("x y", real=True)
    fan = newton_polyhedral_fan((x**2 + y**4,), (x, y), (0, 0))
    cert = completeness_certificate(fan.cones[0])
    assert cert.certified
    assert "newton_cone" in cert.exhausted_kinds
    assert cert.proof_certificate is not None
    assert cert.proof_certificate.certified


def test_complex_cluster_geometry_z_over_w_is_projective_sphere():
    z, w = sp.symbols("z w")
    result = complex_cluster_geometry(z / w, (z, w), (0, 0))
    assert result.certified
    assert result.cluster_set == sp.S.Complexes
    assert result.projective_cluster == "riemann_sphere"


def test_phase_bundle_is_mapped_fiber_product():
    x, y = sp.symbols("x y", real=True)
    phase = 1 / (x**2 + y**2)
    base = LocalStratum("newton_cone", sp.S.true, 1)
    bundle = phase_bundle_over_stratum(
        base, (sp.sin(phase), sp.cos(phase)), (x, y), (0, 0)
    )
    assert bundle is not None
    assert bundle.fiber_product.mode == "fiber_product"
    assert bundle.stratum_certified


def test_unified_compactification_contains_real_complex_valuations():
    x, y = sp.symbols("x y", real=True)
    z, w = sp.symbols("z w")
    fan = newton_polyhedral_fan((x**2 + y**4,), (x, y), (0, 0))
    cg = complex_newton_geometry(z / w, (z, w), (0, 0))
    compact = unified_valuation_compactification(fan=fan, complex_geometry=cg)
    assert compact.coverage_certified
    assert {"newton_cone", "complex_divisor"} <= {n.kind for n in compact.nodes}


def test_projection_scheduler_defers_expensive_full_qe():
    xs = sp.symbols("x0:5", real=True)
    u = sp.symbols("u", real=True)
    base = LocalStratum(
        "base", sp.And(*(x >= -1 for x in xs), *(x <= 1 for x in xs)), 5
    )
    mapping = StratumMap(xs, (sum(x**2 for x in xs),), (u,))
    result = refine_mapped_stratum(mapped_local_stratum(base, mapping))
    assert result.rank_strata
    assert all(s.projection_schedule is not None for s in result.rank_strata)


def test_rank_refinement_exposes_regularity_and_composable_proof():
    x, y, u, v = sp.symbols("x y u v", real=True)
    base = LocalStratum("base", sp.And(x**2 + y**2 <= 1), 2)
    mapping = StratumMap((x, y), (x, x * y), (u, v))
    result = refine_mapped_stratum(mapped_local_stratum(base, mapping))
    assert result.regularity is not None
    assert result.proof_certificate is not None
    assert result.proof_certificate.dependencies
    assert {s.rank for s in result.rank_strata} >= {1, 2}


def test_whitney_curve_tangent_and_secant_are_certified():
    t, u, v = sp.symbols("t u v", real=True)
    base = LocalStratum("curve", sp.And(t >= -1, t <= 1), 1)
    mapping = StratumMap((t,), (t**2, t**3), (u, v))
    result = check_whitney_conditions(mapped_local_stratum(base, mapping))
    assert result.certified
    assert result.pairs[0].condition_a is True
    assert result.pairs[0].condition_b is True
    assert result.pairs[0].tangent_limit.direction == (2, 0)


def test_qe_projection_cache_reuses_completed_projection():
    x, u = sp.symbols("x u", real=True)
    base = LocalStratum("interval", sp.And(x >= -1, x <= 1), 1)
    mapping = StratumMap((x,), (x**2,), (u,))
    mapped = mapped_local_stratum(base, mapping)
    clear_qe_projection_cache()
    refine_mapped_stratum(mapped)
    first = qe_projection_cache_stats()
    refine_mapped_stratum(mapped)
    second = qe_projection_cache_stats()
    assert second.hits > first.hits
