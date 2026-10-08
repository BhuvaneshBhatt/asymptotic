"""Low-order analytic jets preserve cancellation and denominator geometry."""

import pytest
import sympy as sp

from asymptotic._multivariate_analytic import analytic_equivalence_rewrite_certificate


@pytest.mark.parametrize("cancelled_terms", [0, 1, 2])
def test_exponential_cancellation(cancelled_terms):
    x, y = sp.symbols("x y", real=True)
    scale = x * x + y * y
    numerator = sp.exp(scale) - sum(
        scale**k / sp.factorial(k) for k in range(cancelled_terms + 1)
    )
    expr = numerator / scale ** (cancelled_terms + 1)
    certificate = analytic_equivalence_rewrite_certificate(expr, (x, y), (0, 0))
    assert certificate.certified
    assert certificate.value == 1 / sp.factorial(cancelled_terms + 1)
    # The exact positive scale reduces every real approach to one scalar germ.
    t = sp.Symbol("t", positive=True)
    scalar = (
        sp.exp(t) - sum(t**k / sp.factorial(k) for k in range(cancelled_terms + 1))
    ) / t ** (cancelled_terms + 1)
    assert sp.limit(scalar, t, 0) == certificate.value


def test_angular_ratio_declines():
    x, y = sp.symbols("x y", real=True)
    expr = (sp.exp(x * x + y * y) - 1) / (x * x + 2 * y * y)
    certificate = analytic_equivalence_rewrite_certificate(
        expr, (x, y), (0, 0), order=4
    )
    assert not certificate.certified
    assert sp.limit(expr.subs(y, 0), x, 0) == 1
    assert sp.limit(expr.subs(x, 0), y, 0) == sp.Rational(1, 2)


def test_noncoercive_denominator():
    x, y = sp.symbols("x y", real=True)
    expr = (sp.exp(x * x + y * y) - 1) / (x * x)
    certificate = analytic_equivalence_rewrite_certificate(
        expr, (x, y), (0, 0), order=4
    )
    assert not certificate.certified
    assert sp.limit(expr.subs(y, 0), x, 0) == 1
    assert sp.limit(expr.subs(y, x), x, 0) == 2
