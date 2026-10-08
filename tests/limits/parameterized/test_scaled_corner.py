"""Coefficient-zero corners use finite attained limits on the other cell."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.parameter_conditions import scaled_corner_stratification


@pytest.mark.parametrize("logarithmic", [False, True])
def test_scaled_cells(logarithmic):
    h, k = sp.symbols("h k", real=True)
    a = sp.Symbol("a")
    radius_squared = h * h + k * k
    core = (
        sp.log(1 + k * k / radius_squared) / sp.sqrt(radius_squared)
        if logarithmic
        else 1 + k * k / radius_squared ** sp.Rational(3, 2)
    )
    expr = (1 - sp.cosh(a)) * core
    result = limit(expr, (h, k), (0, 0), return_result=True)
    zero = next(
        cell.result
        for cell in result.strata
        if cell.result.status is LimitStatus.PROVED
    )
    conflict = next(
        cell.result
        for cell in result.strata
        if cell.result.status is LimitStatus.DOES_NOT_EXIST
    )
    assert zero.value == 0
    expected = [0, 1] if logarithmic else [1, 2]
    for evidence, value in zip(conflict.evidence, expected, strict=True):
        path = dict(evidence.substitutions)
        j = next(iter(set().union(*(v.free_symbols for v in path.values()))))
        actual = core.subs(path, simultaneous=True)
        assert sp.limit(actual, j, sp.oo) == value
        assert evidence.value == value * (1 - sp.cosh(a))
        assert radius_squared.subs(path, simultaneous=True).is_positive is True
        if logarithmic:
            assert (1 + k * k / radius_squared).subs(
                path, simultaneous=True
            ).is_positive is True
    assert expr.subs(a, 0) == 0
    assert expr.subs(a, 2 * sp.pi * sp.I) == 0
    assert {e.value.subs(a, sp.pi * sp.I) for e in conflict.evidence} == set(
        2 * v for v in expected
    )


def test_scaled_domain():
    h, k = sp.symbols("h k", real=True)
    a = sp.Symbol("a")
    expr = a * sp.log(1 + k * k / (h * h + k * k)) / sp.sqrt(h * h + k * k)
    assert (
        scaled_corner_stratification(expr, (h, k), (0, 0), sp.Eq(k, 0), sp.S.true)
        is None
    )
    undefined = sp.Function("undefined")
    assert (
        scaled_corner_stratification(
            expr.subs(a, undefined(a)), (h, k), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )


@pytest.mark.parametrize("restriction", ["integer", "positive", "negative", "nonzero"])
def test_corner_coordinate_domain(restriction):
    from asymptotic.parameter_conditions import displaced_pole_stratification

    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True, **{restriction: True})
    a = sp.Symbol("a")
    radius = x * x + y * y
    pole = sp.cancel(x * y / ((x - a) ** 2 + y * y))
    corner = a * sp.log(1 + y * y / radius) / sp.sqrt(radius)
    assert (
        displaced_pole_stratification(pole, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )
    assert (
        scaled_corner_stratification(corner, (x, y), (0, 0), sp.S.true, sp.S.true)
        is None
    )
