import sympy as sp

from asymptotic.recursive_resolution import ExceptionalStratum, finite_stratum_atlas
from asymptotic.semialgebraic_angular import semialgebraic_angular_image


def test_tubular_base_preserves_relative_inequality():
    x, y = sp.symbols("x y", real=True)
    s = ExceptionalStratum(sp.And(sp.Eq(x, 0), y >= 0), "half-line", (x,), 1)
    atlas = finite_stratum_atlas(s, (x, y))
    assert atlas is not None and atlas.smooth_charts
    chart = atlas.smooth_charts[0]
    by = chart.base_symbols[1]
    assert chart.condition.has(sp.Ge(by, 0))
    assert chart.exceptional_domain.has(sp.Ge(by, 0))


def test_kkt_angular_image_on_closed_sector():
    x, y = sp.symbols("x y", real=True)
    domain = sp.And(sp.Eq(x * x + y * y, 1), x >= 0)
    result = semialgebraic_angular_image(x**4 + y**4, (x, y), domain)
    assert result.certified
    assert result.minimum == sp.Rational(1, 2)
    assert result.maximum == 1
