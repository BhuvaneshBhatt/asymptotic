import sympy as sp

from asymptotic.algebraic_series import algebraic_series_branches, implicit_map_series
from asymptotic.limits import LimitStatus, limit
from asymptotic.univariate_limits import (
    complex_direction_limit,
    extremal_limit,
    periodic_composition_cluster,
)


def test_complex_direction_disagreement():
    z = sp.symbols("z")
    assert complex_direction_limit(sp.re(z) / sp.Abs(z), z, point=0) is sp.nan


def test_exact_extremal_periodic_limit():
    x = sp.symbols("x", real=True)
    r = extremal_limit(3 + 2 * sp.sin(x), x)
    assert (r.lower, r.upper) == (1, 5)


def test_periodic_composition():
    x = sp.symbols("x", positive=True)
    r = periodic_composition_cluster(lambda t: 2 + sp.sin(t), x**2, x)
    assert r is not None and r.set == sp.Interval(1, 3)


def test_all_algebraic_branches():
    x, y = sp.symbols("x y", real=True)
    bs = algebraic_series_branches(y**2 - x**2, y, x, dependent_limit=0, order=4)
    assert set(bs) == {x, -x}


def test_implicit_nonzero_base_point():
    x, y = sp.symbols("x y")
    r = implicit_map_series(
        (y - x**2,), (y,), x, point=1, dependent_limit=(1,), order=4
    )
    assert r is not None and sp.expand(r.series[0] - x**2) == 0


def test_assumption_conditioned_positive_scale_metamorphism():
    x, a = sp.symbols("x a", positive=True)
    base = limit(sp.sin(x) / x, (x,), (0,), return_result=True)
    scaled = limit(
        sp.sin(a * x) / (a * x), (x,), (0,), assumptions=sp.Gt(a, 0), return_result=True
    )
    assert base.status is scaled.status is LimitStatus.PROVED
    assert base.value == scaled.value == 1


def test_target_translation_metamorphism():
    x, u = sp.symbols("x u", real=True)
    a = sp.Rational(7, 3)
    r1 = limit(sp.sin(x) / x, (x,), (0,), return_result=True)
    r2 = limit(sp.sin(u - a) / (u - a), (u,), (a,), return_result=True)
    assert r1.status is r2.status is LimitStatus.PROVED and r1.value == r2.value == 1


def test_positive_output_unit_metamorphism():
    x, c = sp.symbols("x c", positive=True)
    r = limit(
        c * sp.sin(x) / x, (x,), (0,), assumptions=sp.Gt(c, 0), return_result=True
    )
    assert r.status is LimitStatus.PROVED and sp.simplify(r.value - c) == 0
