import sympy as sp

from asymptotic.singular_expansion_geometry import (
    projective_infinity_expansion,
    singular_expansion_atlas,
)


def test_cusp_has_two_exact_ramified_branches_and_valuation():
    x, y = sp.symbols("x y", real=True)
    result = singular_expansion_atlas(1 + x + y, y**2 - x**3, (x, y), order=3)
    assert result.decomposition.certified
    assert len(result.branches) == 2
    assert {b.valuation for b in result.branches} == {(2, 3)}
    assert all(b.exact_branch for b in result.branches)


def test_tacnode_and_reducible_crossing_are_covered_exactly():
    x, y = sp.symbols("x y", real=True)
    tacnode = singular_expansion_atlas(1 + x**2 + y, y**2 - x**4, (x, y), order=3)
    crossing = singular_expansion_atlas(1 + x + y, (y - x) * (y + x), (x, y), order=3)
    assert tacnode.decomposition.certified
    assert len(tacnode.branches) == 2
    assert crossing.decomposition.certified
    assert len(crossing.decomposition.components) == 2


def test_translated_cusp_uses_local_target():
    x, y = sp.symbols("x y", real=True)
    curve = (y - 2) ** 2 - (x - 1) ** 3
    result = singular_expansion_atlas(x + y, curve, (x, y), target=(1, 2), order=3)
    assert result.decomposition.certified
    assert len(result.branches) == 2
    assert {b.valuation for b in result.branches} == {(2, 3)}


def test_weighted_chart_exposes_uniform_remainder_when_certified():
    x, y = sp.symbols("x y", real=True)
    result = singular_expansion_atlas(1 / (1 + x + y), y**2 - x**3, (x, y), order=2)
    assert result.charts
    certified = [c for c in result.charts if c.certified]
    assert certified
    assert certified[0].expansion.uniform_remainder().certified


def test_nonpolynomial_curve_declines_with_branch_obligation():
    x, y = sp.symbols("x y", real=True)
    result = singular_expansion_atlas(x + y, y - sp.sin(x), (x, y))
    assert not result.certified
    assert result.obligations


def test_projective_infinity_reciprocal_compactification():
    x, y = sp.symbols("x y", real=True)
    result = projective_infinity_expansion(1 / x + 1 / y, (x, y), order=2)
    assert result.projective_atlas.certified
    assert result.transformed_expression == sum(result.reciprocal_variables)
    assert result.certified
