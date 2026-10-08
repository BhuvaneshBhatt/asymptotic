import sympy as sp

from asymptotic.local_strata import LocalStratum, frontier_incidence_complex
from asymptotic.mapped_strata import (
    StratumMap,
    induced_frontier_map,
    mapped_local_stratum,
)
from asymptotic.stratum_evaluator import evaluate_local_stratum


def test_mapped_stratum_preserves_correlated_fiber_image():
    t, c, s, u, v = sp.symbols("t c s u v", real=True)
    base = LocalStratum("newton_cone", sp.And(t >= 0, t <= 1), 1)
    phase = LocalStratum("phase_torus", sp.Eq(c**2 + s**2, 1), 1)
    mapping = StratumMap((t, c, s), (t * c, t * s), (u, v))
    mapped = mapped_local_stratum(
        base, mapping, phase, relation=sp.Eq(c**2 + s**2, 1), mode="fiber_product"
    )
    result = evaluate_local_stratum(mapped)
    assert result.certified
    assert result.provider == "mapped_stratum_image"
    assert isinstance(result.limiting_image, sp.ImageSet)


def test_fiber_product_relation_is_not_treated_as_independent_product():
    t, c, s, u, v = sp.symbols("t c s u v", real=True)
    base = LocalStratum("base", sp.And(t >= 0, t <= 1), 1)
    phase = LocalStratum("phase", sp.Eq(c**2 + s**2, 1), 1)
    mapping = StratumMap((t, c, s), (c, s), (u, v))
    mapped = mapped_local_stratum(
        base, mapping, phase, relation=sp.Eq(c, t), mode="fiber_product"
    )
    assert mapped.local_constraint.has(sp.Eq(c, t))
    result = evaluate_local_stratum(mapped)
    assert result.certified


def test_induced_frontier_map_detects_dimension_collapse():
    t, u = sp.symbols("t u", real=True)
    frontier = frontier_incidence_complex(sp.And(t >= 0, t <= 1), (t,))
    assert frontier is not None and frontier.certified
    mapping = StratumMap((t,), (sp.S.Zero,), (u,))
    induced = induced_frontier_map(frontier, mapping)
    assert induced.certified
    assert any(s.collapsed for s in induced.strata if s.source_dimension > 0)
    assert all(s.image_dimension == 0 for s in induced.strata)
