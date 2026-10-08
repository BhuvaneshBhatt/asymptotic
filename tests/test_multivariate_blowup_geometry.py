import sympy as sp

from asymptotic.blowup_geometry import (
    Valuation,
    projective_chart,
    weighted_spherical_atlas,
)


def test_weighted_valuation_and_initial_form():
    x, y = sp.symbols("x y")
    valuation = Valuation((x, y), (1, 2))
    expr = x**4 + x**2 * y + y**3
    assert valuation.polynomial(expr) == 4
    assert sp.expand(valuation.initial_form(expr) - (x**4 + x**2 * y)) == 0


def test_weighted_atlas_uses_one_common_chart_substitution():
    x, y = sp.symbols("x y", real=True)
    atlas = weighted_spherical_atlas((x, y), (0, 0), (1, 2))
    assert atlas.certified
    assert len(atlas.charts) == 2
    chart = atlas.charts[0]
    assert chart.substitution[x] == chart.radial_variable * chart.angular_variables[0]
    assert (
        chart.substitution[y] == chart.radial_variable**2 * chart.angular_variables[1]
    )


def test_projective_chart_is_a_blowup_chart_not_a_separate_geometry():
    x, y = sp.symbols("x y", real=True)
    chart = projective_chart((x, y), (0, 0))
    transformed = chart.transform((x**2 + y**2) / (x**2 - y**2))
    r = chart.radial_variable
    t = chart.angular_variables[1]
    assert sp.simplify(transformed - (1 + t**2) / (1 - t**2)) == 0
    assert not transformed.has(r)
