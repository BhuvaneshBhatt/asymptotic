"""Fixed-ratio Debye tails and their proof boundaries."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.large_order_bessel_germs import proportional_bessel_certificate
from asymptotic.limit_models import LimitStatus

x = sp.Symbol("x", real=True)
rho = 2 * sp.log(2 + sp.sqrt(3)) - sp.sqrt(3)
constant = 1 / sp.sqrt(2 * sp.pi * sp.sqrt(3))


@pytest.mark.parametrize(
    "expression,expected",
    [
        (sp.sqrt(x) * sp.exp(rho * x) * sp.besselj(2 * x, x), constant),
        (sp.sqrt(x) * sp.exp(-rho * x) * sp.bessely(2 * x, x), -2 * constant),
        (sp.sqrt(x) * sp.exp(rho * x) * sp.besselj(2 * x * x, x * x), sp.S.Zero),
        (sp.sqrt(x) * sp.exp(rho * x) * sp.besselj(3 * x, x), sp.S.Zero),
        (sp.sqrt(x) * sp.exp(rho * x) * sp.besselj(sp.Rational(3, 2) * x, x), sp.oo),
        (
            sp.sqrt(x) * sp.exp(2 * rho * x) * sp.besselj(4 * x, 2 * x),
            constant / sp.sqrt(2),
        ),
        (x * x * sp.besselj(2 * x, x), sp.S.Zero),
        (x * sp.exp(rho * x) * sp.besselj(2 * x, x), sp.oo),
        (-x * sp.exp(-rho * x) * sp.bessely(2 * x, x), sp.oo),
        (sp.sqrt(x) * sp.exp(rho * x + 1) * sp.besselj(2 * x, x), sp.E * constant),
    ],
)
def test_debye_tail(expression, expected):
    result = limit(expression, x, sp.oo, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == expected or sp.simplify(result.value - expected) == 0
    assert any(e.method == "proportional_large_bessel_order" for e in result.evidence)


@pytest.mark.parametrize(
    "expression",
    [
        sp.besselj(x, x),
        sp.besselj(x / 2, x),
        sp.besselj(2 * x + 1, x),
        sp.besselj(2 * x, -x),
        sp.besselj(2 * x**9, x**9),
        sp.besselj((x + 1) ** 1000000, x),
        sp.exp((x + 1) ** 64) * sp.besselj(2 * x, x),
        sp.I * sp.besselj(2 * x, x),
        sp.exp(x * x + sp.I * x) * sp.besselj(2 * x, x),
        sp.exp(sp.Function("f")()) * sp.besselj(2 * x, x),
        sp.besselj(sp.Symbol("a") * x, x),
        sp.besselj(sp.Float(2) * x, x),
        (sp.sqrt(x) * sp.exp(rho * x) * sp.besselj(2 * x, x) - constant) * x,
        1 / sp.besselj(2 * x, x),
        sp.besselj(2 * x, x) * sp.exp(sp.besselj(2 * x, x)),
    ],
)
def test_debye_boundary(expression):
    assert (
        proportional_bessel_certificate(expression, x, sp.oo, sp.S.true, sp.S.true)
        is None
    )


def test_real_tail_domain():
    expression = sp.besselj(2 * x, x)
    assert (
        proportional_bessel_certificate(expression, x, 0, sp.S.true, sp.S.true) is None
    )
    assert (
        proportional_bessel_certificate(expression, x, sp.oo, x > 0, sp.S.true) is None
    )
    assert (
        proportional_bessel_certificate(expression, x, sp.oo, sp.S.true, sp.S.false)
        is None
    )


def test_ball_precision_restored():
    from flint import ctx

    before = ctx.prec
    result = proportional_bessel_certificate(
        sp.sqrt(x) * sp.exp(rho * x) * sp.besselj(3 * x, x),
        x,
        sp.oo,
        sp.S.true,
        sp.S.true,
    )
    assert result[1] == 0
    assert ctx.prec == before
