"""Uniform norm estimates, original domain witnesses and decline boundaries."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.uniform_radial_bounds import (
    radial_pole_certificate,
    radial_vanishing_certificate,
    signed_power_vanishing_certificate,
)

x, y = sp.symbols("x y", real=True)
s = x * x + y * y


@pytest.mark.parametrize(
    "expr",
    [
        sp.exp(-1 / s) / (x**4 + y**4),
        sp.exp(-1 / s) / (x**6 + y**6),
        sp.Abs(x * y) * sp.log(s),
        y * y * sp.Abs(sp.log(s)),
        x**20 * y**22 / s ** sp.Rational(1, 10),
        x**5 * y**3 / ((x**4 + y * y) * sp.Max(sp.Abs(x), sp.Abs(y))),
        y * sp.sqrt(sp.Abs(x)) / sp.Max(sp.Abs(x), sp.Abs(y)),
        y
        * y
        / ((sp.Abs(x) + sp.Abs(y)) * sp.sqrt(1 + y**4 / (sp.Abs(x) + sp.Abs(y)) ** 2)),
        x * y / (1 - sp.Abs(x * y) ** sp.Rational(1, 3) * sp.sign(x * y)),
    ],
)
def test_uniform_zero(expr):
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    witness = dict(result.evidence[0].substitutions)
    j = next(iter(witness[x].free_symbols))
    assert sp.limit(expr.subs(witness, simultaneous=True), j, sp.oo) == 0
    for power in expr.atoms(sp.Pow):
        if power.exp.is_negative is True:
            along = power.base.subs(witness, simultaneous=True)
            assert all(along.subs(j, n).is_zero is False for n in (2, 3, 7))


@pytest.mark.parametrize(
    "expr",
    [
        x * x / s,
        x * y / (x * x + y**4),
        x / (x * x - y * y),
        sp.exp(1 / s) * x,
        sp.log(x * y) * x,
    ],
)
def test_radial_declines(expr):
    assert (
        radial_vanishing_certificate(
            expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
        )
        is None
    )


def test_shifted_bound():
    expr = ((x + 2) ** 2 + y * y) / (
        sp.Abs((x + 2) ** 4 + y**4) ** sp.Rational(1, 3) * sp.sign((x + 2) ** 4 + y**4)
    )
    result = limit(expr, (x, y), (-2, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    witness = dict(result.evidence[0].substitutions)
    j = next(iter(witness[x].free_symbols))
    assert sp.limit(witness[x], j, sp.oo) == -2
    assert sp.limit(expr.subs(witness, simultaneous=True), j, sp.oo) == 0


@pytest.mark.parametrize("polarity", [1, -1])
def test_dominant_pole(polarity):
    expr = polarity * (1 / (x**4 + y**4) + sp.Rational(3, 2) * sp.log(s))
    certificate = radial_pole_certificate(
        expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
    )
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == polarity * sp.oo
    witness = dict(certificate[2].substitutions)
    j = next(iter(witness[x].free_symbols))
    assert sp.limit(expr.subs(witness, simultaneous=True), j, sp.oo) == polarity * sp.oo


@pytest.mark.parametrize("power", [sp.S.Zero, sp.Rational(2, 3)])
def test_signed_holes(power):
    inner = x - y
    expr = inner / (sp.Abs(inner) ** power * sp.sign(inner) ** 2)
    certificate = signed_power_vanishing_certificate(
        expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
    )
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == 0
    witness = dict(certificate[2].substitutions)
    assert inner.subs(witness, simultaneous=True).is_zero is False


def test_domain_not_erased():
    expr = x**4 / s
    assert (
        radial_vanishing_certificate(
            expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.Eq(x, 0), sp.S.true
        )
        is None
    )
    a = sp.Symbol("a", real=True)
    assert (
        radial_vanishing_certificate(
            a * expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        radial_pole_certificate(
            1 / s - 1 / s**2, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
        )[1]
        == -sp.oo
    )
