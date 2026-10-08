import sympy as sp

from asymptotic._limit_engine import _exact_radial_reduction


def test_radial_expression():
    x, y = sp.symbols("x y", real=True)
    reduction = _exact_radial_reduction(
        sp.sin(x**2 + y**2), (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true
    )
    assert reduction is not None
    radial, radius_squared = reduction
    assert radial == sp.sin(radius_squared)


def test_radial_domain():
    x, y = sp.symbols("x y", real=True)
    domain = x**2 + y**2 < 1
    reduction = _exact_radial_reduction(
        1 / (1 + x**2 + y**2), (x, y), (sp.S.Zero, sp.S.Zero), domain
    )
    assert reduction is not None


def test_angular_domain_declines():
    x, y = sp.symbols("x y", real=True)
    reduction = _exact_radial_reduction(
        x**2 + y**2, (x, y), (sp.S.Zero, sp.S.Zero), x > y
    )
    assert reduction is None


def test_nonradial_expression_declines():
    x, y = sp.symbols("x y", real=True)
    reduction = _exact_radial_reduction(
        x + y, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true
    )
    assert reduction is None
