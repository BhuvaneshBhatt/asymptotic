import sympy as sp

from asymptotic.algebraic_curve_coverage import (
    certified_bivariate_curve_decomposition,
    valuation_from_approach,
    weighted_atlases_for_decomposition,
)


def test_cusp_exact_branches_cover_and_generate_weighted_atlas():
    x, y = sp.symbols("x y", real=True)
    d = certified_bivariate_curve_decomposition(y**2 - x**3, y, x)
    assert d.certified
    approaches = tuple(a for c in d.components for a in c.approaches)
    assert len(approaches) == 2
    assert {valuation_from_approach(a) for a in approaches} == {(2, 3)}
    atlases = weighted_atlases_for_decomposition(d, (x, y), (0, 0))
    assert len(atlases) == 1 and atlases[0].coverage_certified


def test_reducible_curve_factor_coverage_is_composed():
    x, y = sp.symbols("x y", real=True)
    d = certified_bivariate_curve_decomposition((y - x) * (y + x), y, x)
    assert d.certified and len(d.components) == 2
