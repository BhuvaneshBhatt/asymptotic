import sympy as sp

from asymptotic._multivariate_germ import _weighted_leading, _weighted_poly_order
from asymptotic.blowup_geometry import (
    Valuation,
    newton_valuation_rays,
    weighted_spherical_atlas,
)


def test_weighted_germ_helpers_delegate_to_common_valuation_semantics():
    x, y = sp.symbols("x y")
    expr = x**4 + x**2 * y + y**3
    v = Valuation((x, y), (1, 2))
    assert _weighted_poly_order(expr, (x, y), (1, 2)) == v.polynomial(expr)
    assert _weighted_leading(expr, (x, y), (1, 2)) == (
        v.polynomial(expr),
        v.initial_form(expr),
    )


def test_weighted_atlas_exposes_common_coverage_certificate():
    x, y = sp.symbols("x y")
    atlas = weighted_spherical_atlas((x, y), (0, 0), (1, 2))
    assert atlas.coverage.certified
    assert atlas.coverage_certified
    assert atlas.certified


def test_newton_rays_return_same_engine_coverage_certificate():
    x, y = sp.symbols("x y")
    rays, coverage, fan = newton_valuation_rays(
        (x**4 + x * y**2 + y**5,), (x, y), (0, 0)
    )
    assert rays
    assert coverage.certified == fan.coverage_certified
