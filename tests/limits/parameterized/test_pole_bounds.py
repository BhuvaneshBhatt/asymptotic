"""Pole bounds prove real sign and an attained defined-domain approach."""

import pytest
import sympy as sp

from asymptotic.limit_models import LimitStatus
from asymptotic.multivariate_pole_bounds import regular_numerator_pole_certificate


@pytest.mark.parametrize("family", ["quartic", "translated", "absolute", "diagonal"])
def test_signed_poles(family):
    x, y = sp.symbols("x y", real=True)
    if family == "quartic":
        expr = (
            (1 - (y + 2) * sp.cos(y * (x - 1) ** 2))
            / ((x - 1) ** 4 + (y + 2) ** 4)
            / sp.sqrt((x - 1) ** 2 + (y + 2) ** 2)
        )
        target, expected = (sp.Integer(1), sp.Integer(-2)), sp.oo
    elif family == "translated":
        expr = (
            -sp.Rational(1, 2) * y - 1 + (3 * x * y * y - y**3) / (x * x + y * y)
        ) / sp.sqrt((x - 1) ** 2 + (y - 1) ** 2)
        target, expected = (sp.Integer(1), sp.Integer(1)), -sp.oo
    elif family == "absolute":
        expr = -4 * sp.cos(sp.Abs(x * y) ** sp.Rational(1, 29)) / sp.Abs(x * y) + 4
        target, expected = (sp.S.Zero, sp.S.Zero), -sp.oo
    else:
        expr = 1 / (x - y) ** 2
        target, expected = (sp.S.Zero, sp.S.Zero), sp.oo
    status, value, evidence = regular_numerator_pole_certificate(
        expr, (x, y), target, sp.S.true
    )
    assert status is LimitStatus.PROVED
    assert value is expected
    path = dict(evidence.substitutions)
    j = next(iter(set().union(*(v.free_symbols for v in path.values()))))
    along = expr.subs(path, simultaneous=True)
    assert sp.limit(along, j, sp.oo) is expected
    denominator = sp.fraction(sp.together(expr))[1].subs(path, simultaneous=True)
    tail_index = sp.Dummy("tail_index", positive=True, integer=True)
    assert denominator.subs(j, tail_index + 2).is_positive is True


def test_pole_scope():
    x, y = sp.symbols("x y", real=True)
    origin = (sp.S.Zero, sp.S.Zero)
    assert (
        regular_numerator_pole_certificate(
            (1 + sp.I * x) / (x * x + y * y), (x, y), origin, sp.S.true
        )
        is None
    )
    assert (
        regular_numerator_pole_certificate(
            1 / (x * x - y * y), (x, y), origin, sp.S.true
        )
        is None
    )
    assert (
        regular_numerator_pole_certificate(
            1 / (x * x + y * y), (x, y), origin, sp.Eq(x, 0)
        )
        is None
    )
    assert (
        regular_numerator_pole_certificate(
            x / (x * x + y * y), (x, y), origin, sp.S.true
        )
        is None
    )


def test_restricted_coordinates():
    x, y = sp.symbols("x y", real=True)
    assert (
        regular_numerator_pole_certificate(
            1 / (x * x + y * y), (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.Eq(x, 0)
        )
        is None
    )
    n = sp.Symbol("n", integer=True)
    assert (
        regular_numerator_pole_certificate(
            1 / (n * n + y * y), (n, y), (sp.S.Zero, sp.S.Zero), sp.S.true
        )
        is None
    )
