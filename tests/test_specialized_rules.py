"""Proof conditions for bounded coercivity, rational jets and domain witnesses."""

import pytest
import sympy as sp

from asymptotic._limit_certificates import _positive_definite_pure_even_form
from asymptotic.domain_witnesses import attained_ray, domain_accumulates
from asymptotic.generalized_series import _rational_jet, generalized_series
from asymptotic.multivariate_certificates import (
    general_isolated_zero_certificate,
    general_lojasiewicz_exponent,
)

x, y = sp.symbols("x y", real=True)


def test_even_form_definiteness():
    assert _positive_definite_pure_even_form((x + y) ** 2, (x, y)) is False
    assert (
        _positive_definite_pure_even_form(x**4 + y**4 + x * x * y * y, (x, y)) is True
    )


@pytest.mark.parametrize(
    "expression",
    [
        x * x + 2 * x * y + 2 * y * y,
        x * x + 2 * x * y + 2 * y * y + x**3,
        x**4 + y**6 + x**3 * y**3,
        -(x**4 + y**6 + x**3 * y**3),
    ],
)
def test_coercive_zero(expression):
    result = general_isolated_zero_certificate(expression, (x, y), (0, 0))
    assert result.certified and result.value is sp.S.true
    assert result.method == "isolated_zero_coercive"


def test_shifted_zero():
    result = general_isolated_zero_certificate(
        (x - 2) ** 2 + 2 * (x - 2) * (y + 1) + 2 * (y + 1) ** 2, (x, y), (2, -1)
    )
    assert result.certified and result.value is sp.S.true


@pytest.mark.parametrize("expression", [x * x - y * y, x * x * y * y])
def test_nonisolated_zero(expression):
    result = general_isolated_zero_certificate(expression, (x, y), (0, 0))
    assert result.certified and result.value is sp.S.false


def test_positive_radius_guard():
    result = general_lojasiewicz_exponent(x * x - y * y, (x, y), (0, 0), max_exponent=2)
    assert result.exponent is None and result.provider == "nonisolated_zero_ray"


@pytest.mark.parametrize(
    "expression, order",
    [
        ((1 + x) / (1 - x - x * x), 12),
        ((x**7 + 2 * x**3 + 1) / (1 + x + x**4), 24),
        (x**8 / (1 - x), 4),
        (sp.S.Zero, 4),
    ],
)
def test_rational_remainder(expression, order):
    result = generalized_series(expression, x, terms=order)
    p, q = expression.as_numer_denom()
    residual = sp.Poly(sp.expand(q * result.as_expr() - p), x)
    assert all(residual.nth(k) == 0 for k in range(order))
    assert result.order_bound >= order


def test_exact_polynomial():
    result = generalized_series(1 + x * x, x, terms=4)
    assert result.remainder.exact
    assert result.as_expr() == 1 + x * x


def test_series_chart():
    result = generalized_series(1 / (1 + x), x, point=2, terms=3)
    assert sp.expand(result.as_expr()) == sp.expand(
        sp.Rational(1, 3) - (x - 2) / 9 + (x - 2) ** 2 / 27
    )
    result = generalized_series(1 / (1 + x), x, point=sp.oo, terms=3)
    assert result.as_expr() == 1 / x - 1 / x**2
    result = generalized_series(1 / (1 + x), x, point=-sp.oo, terms=3)
    assert result.as_expr() == 1 / x - 1 / x**2


def test_singular_series_fallback():
    result = generalized_series(1 / x, x, terms=4)
    assert result.as_expr() == 1 / x and result.remainder.exact


def test_jet_degree_budget():
    assert _rational_jet((1 + x + x**2) ** 64, x, 4) is None
    assert _rational_jet((1 + x) ** 1000000, x, 4) is None


def test_float_exactness():
    coefficient = sp.Float("0.1", 40)
    assert _rational_jet(coefficient * x, x, 4) is None
    result = generalized_series(coefficient * x, x, terms=4)
    assert result.as_expr().coeff(x) == coefficient
    assert attained_ray(x > coefficient, (x,), (coefficient,)) is None


@pytest.mark.parametrize(
    "domain,variables,target",
    [
        (sp.And(x - x * x > 0, sp.Ne(x, 0)), (x,), (0,)),
        (sp.And(x > 0, y > x * x, sp.Ne(x + y, 0)), (x, y), (0, 0)),
        (sp.And(x < 2, x > 1), (x,), (2,)),
        (sp.And(x >= 0, sp.Eq(y, 0), sp.Ne(x, 0)), (x, y), (0, 0)),
    ],
)
def test_attained_ray(domain, variables, target):
    direction = attained_ray(domain, variables, target)
    assert direction is not None and any(direction)
    # The returned rational sequence remains in the actual domain.
    for k in (10, 100, 1000):
        point = dict(
            zip(
                variables,
                (a + sp.Rational(d, k) for a, d in zip(target, direction, strict=True)),
                strict=True,
            )
        )
        assert domain.subs(point) is sp.S.true


def test_curved_domain_fallback():
    domain = sp.And(x > 0, sp.Eq(y, x * x))
    assert attained_ray(domain, (x, y), (0, 0)) is None
    assert domain_accumulates(domain, (x, y), (0, 0)) is True


def test_empty_domain():
    domain = sp.And(x > 0, x < 0)
    assert attained_ray(domain, (x,), (0,)) is None
    assert domain_accumulates(domain, (x,), (0,)) is False


def test_ray_bounds():
    variables = sp.symbols("a:4", real=True)
    assert attained_ray(sp.S.true, variables, (0,) * 4) is None
    assert attained_ray(x**17 > 0, (x,), (0,)) is None
    assert attained_ray(sp.Eq((1 + x) ** 1000000, y), (x, y), (0, 0)) is None
    assert attained_ray(x > 0, (x,), (sp.oo,)) is None
