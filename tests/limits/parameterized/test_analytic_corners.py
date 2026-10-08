"""Analytic parameter corners use attained, pole-free subsequences."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic._multivariate_analytic import (
    removable_transcendental_normalization_certificate,
)
from asymptotic.analytic_parameter_corners import analytic_corner_stratification


@pytest.mark.parametrize(
    "outer", [sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.exp, sp.erf, sp.erfc]
)
def test_analytic_corner_cells(outer):
    x, y, a = sp.symbols("x y a", real=True)
    inner = x * y**2 / (x**4 + a * y**2)
    expr = outer(inner)
    result = limit(expr, (x, y), (0, 0), return_result=True)
    positive = next(c for c in result.strata if c.result.status.name == "PROVED")
    negative = next(
        c for c in result.strata if c.result.status.name == "DOES_NOT_EXIST"
    )
    assert sp.simplify(sp.Equivalent(positive.condition, a > 0)) is sp.S.true
    assert sp.simplify(sp.Equivalent(negative.condition, a <= 0)) is sp.S.true
    assert positive.result.value == outer(0)
    first, second = negative.result.evidence
    assert first.value == outer(0)
    for parameter, attained in [
        (0, sp.Rational(1, 2)),
        (-sp.Rational(1, 3), -sp.Rational(1, 2)),
        (-2, -sp.Rational(1, 2)),
    ]:
        path = dict(second.substitutions)
        index = next(iter(set().union(*(v.free_symbols for v in path.values())) - {a}))
        actual = inner.subs(path, simultaneous=True).subs(a, parameter)
        assert sp.limit(actual, index, sp.oo) == attained
        denominator = (x**4 + a * y**2).subs(path, simultaneous=True).subs(a, parameter)
        assert sp.factor(denominator).is_zero is False
        assert second.value.subs(a, parameter) == outer(attained)
        assert (
            sp.simplify(second.value.subs(a, parameter) - first.value).is_zero is False
        )
        if outer is sp.tan:
            assert abs(actual.subs(index, 10) - attained) < sp.Rational(1, 4)
            assert sp.cos(actual.subs(index, 10)).is_zero is False


@pytest.mark.parametrize("q", [1, 3, 4])
def test_analytic_corner_orders(q):
    x, y, a = sp.symbols("x y a", real=True)
    expr = sp.tan(x * y**2 / (x ** (2 * q) + a * y**2))
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert {c.result.status.name for c in result.strata} == {"PROVED", "DOES_NOT_EXIST"}
    assert (
        next(c.result.value for c in result.strata if c.result.status.name == "PROVED")
        == 0
    )


def test_analytic_corner_domain():
    x, y, a = sp.symbols("x y a", real=True)
    expr = sp.tan(x * y**2 / (x**4 + a * y**2))
    assert (
        analytic_corner_stratification(expr, (x, y), (0, 0), sp.Eq(y, 0), sp.S.true)
        is None
    )


def test_removable_stratified_result():
    x, y, a = sp.symbols("x y a", real=True)
    expr = sp.tan(x * y**2 / (x**4 + a * y**2))
    result = removable_transcendental_normalization_certificate(expr, (x, y), (0, 0))
    assert result.certified is False
