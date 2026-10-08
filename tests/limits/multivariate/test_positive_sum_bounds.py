"""Uniform monomial bounds for positive denominator sums."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.limit_models import LimitStatus
from asymptotic.multivariate_pole_bounds import positive_sum_vanishing_certificate


@pytest.mark.parametrize("kind", ["mixed", "weighted", "root"])
def test_positive_sum_zero(kind):
    x, y = sp.symbols("x y", real=True)
    expr = {
        "mixed": (x**4 + 2 * y**3) / (x * x + sp.Abs(y)),
        "weighted": (3 * x**4 - 2 * x * y) / (4 * x * x + 3 * sp.Abs(y)),
        "root": x / (x * x + sp.Abs(y)) ** sp.Rational(1, 3),
    }[kind]
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    evidence = result.evidence[0]
    certificate = positive_sum_vanishing_certificate(
        expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
    )
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == 0
    sequence = dict(evidence.substitutions)
    index = next(iter(sequence[x].free_symbols))
    assert sp.limit(expr.subs(sequence, simultaneous=True), index, sp.oo) == 0
    denominator = sp.denom(expr).subs(sequence, simultaneous=True)
    assert denominator.is_positive is True


def test_positive_sum_declines():
    x, y = sp.symbols("x y", real=True)
    for expr in [
        x * x / (x * x + sp.Abs(y)),
        x / (x * x - y * y),
        (x * x + 1) / (x * x + sp.Abs(y)),
    ]:
        assert (
            positive_sum_vanishing_certificate(
                expr, (x, y), (0, 0), sp.S.true, sp.S.true
            )
            is None
        )
    expr = x**4 / (x * x + sp.Abs(y))
    assert (
        positive_sum_vanishing_certificate(expr, (x, y), (0, 0), sp.Eq(x, 0), sp.S.true)
        is None
    )


def test_complex_inner_removable():
    x, y = sp.symbols("x y", real=True)
    inner = x**3 + y ** sp.Rational(2, 3)
    expr = (1 - sp.exp(inner)) / inner
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == -1
    assert result.evidence[0].method == "holomorphic_removable_germ"
    index = sp.Symbol("j", positive=True, integer=True)
    assert (
        sp.expand_complex(
            sp.limit(
                expr.subs({x: 0, y: -1 / index**3}, simultaneous=True), index, sp.oo
            )
        )
        == -1
    )


def test_positive_sum_exactness_budget():
    x, y = sp.symbols("x y", real=True)
    for numerator in (sp.Float(0.1) * x**4, (x + y) ** 1000000):
        assert (
            positive_sum_vanishing_certificate(
                numerator / (x * x + sp.Abs(y)), (x, y), (0, 0), sp.S.true, sp.S.true
            )
            is None
        )


def test_removable_original_denominator():
    from asymptotic.multivariate_pole_bounds import removable_germ_certificate

    x, y = sp.symbols("x y", real=True)
    expr = sp.Mul(x, sp.exp(y) - 1, sp.Pow(x * y, -1), evaluate=False)
    certificate = removable_germ_certificate(expr, (x, y), (0, 0), sp.S.true, sp.S.true)
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == 1
    sequence = dict(certificate[2].substitutions)
    assert (x * y).subs(sequence, simultaneous=True).is_positive is True
    index = next(iter(sequence[x].free_symbols))
    assert sp.limit(expr.subs(sequence, simultaneous=True), index, sp.oo) == 1


@pytest.mark.parametrize("kind", ["base_power", "exponential", "signed_norm"])
def test_normalized_positive_bound(kind):
    x, y = sp.symbols("x y", real=True)
    radius = x * x + y * y
    expr = {
        "base_power": radius / (2 ** (sp.Abs(x) + sp.Abs(y)) - 1),
        "exponential": radius / (sp.exp(sp.Abs(x) + sp.Abs(y)) - 1),
        "signed_norm": (6 * x - radius)
        / (sp.Abs(radius) ** sp.Rational(1, 3) * sp.sign(radius)),
    }[kind]
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
    certificate = positive_sum_vanishing_certificate(
        expr, (x, y), (sp.S.Zero, sp.S.Zero), sp.S.true, sp.S.true
    )
    assert certificate[0] is LimitStatus.PROVED
    assert certificate[1] == 0
    sequence = dict(result.evidence[0].substitutions)
    index = next(iter(sequence[x].free_symbols))
    assert sp.limit(expr.subs(sequence, simultaneous=True), index, sp.oo) == 0
    denominator = sp.denom(expr).subs(sequence, simultaneous=True)
    if kind == "signed_norm":
        assert denominator.is_positive is True
    else:
        # The positive real exponential is strictly increasing, so
        # log(1+D)>0 proves D>0 without asking for an unavailable sign inference.
        assert (1 + denominator).is_positive is True
        assert sp.expand_log(sp.log(1 + denominator), force=True).is_positive is True


def test_bound_normalization_declines():
    x, y = sp.symbols("x y", real=True)
    for denominator in (
        sp.exp(x - y) - 1,
        sp.Rational(1, 2) ** (sp.Abs(x) + sp.Abs(y)) - 1,
        sp.Abs(x - y) ** sp.Rational(1, 3) * sp.sign(x - y),
    ):
        assert (
            positive_sum_vanishing_certificate(
                (x * x + y * y) / denominator, (x, y), (0, 0), sp.S.true, sp.S.true
            )
            is None
        )


def test_trigonometric_removable_cofactor():
    x, y = sp.symbols("x y", real=True)
    inner = x**4 + y**4
    quotient = sp.sin(inner) * sp.tan(inner) / (1 - sp.cos(inner))
    expr = (1 + sp.log(1 + x * x + y * y)) * quotient
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 2
    evidence = result.evidence[0]
    assert evidence.method == "holomorphic_trigonometric_quotient"
    sequence = dict(evidence.substitutions)
    index = next(iter(sequence[x].free_symbols))
    assert sp.limit(expr.subs(sequence, simultaneous=True), index, sp.oo) == 2
    for j in (2, 3, 8):
        assert (
            sp.denom(expr).subs(sequence, simultaneous=True).subs(index, j).is_zero
            is False
        )
    from asymptotic.multivariate_pole_bounds import removable_germ_certificate

    assert (
        removable_germ_certificate(
            x * quotient / y, (x, y), (0, 0), sp.S.true, sp.S.true
        )
        is None
    )
