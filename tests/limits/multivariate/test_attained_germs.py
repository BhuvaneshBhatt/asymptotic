"""Uniform compositions and attained, denominator-safe conflicting approaches."""

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.attained_ray_germs import (
    elementary_ray_conflict,
    exponential_linear_pole_conflict,
    ray_leading_term,
)
from asymptotic.binomial_pole_witnesses import binomial_pole_conflict
from asymptotic.limit_models import LimitStatus
from asymptotic.regular_germ_composition import (
    nonzero_sign_composition,
    rational_zero_composition,
    signed_root_composition,
)

x, y = sp.symbols("x y", real=True)
zero = (sp.S.Zero, sp.S.Zero)


def signed_root(h):
    return sp.Abs(h) ** sp.Rational(1, 3) * sp.sign(h)


@pytest.mark.parametrize("h", [x * y - 1, sp.Abs(x * y) - 1])
def test_signed_root_endpoint(h):
    expr = sp.cos(signed_root(h))
    result = limit(expr, (x, y), (1, 1), return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == 1
    assert any(e.method == "signed_root_composition" for e in result.evidence)


def test_sign_neighborhood():
    result = nonzero_sign_composition(
        sp.cos(signed_root(x * y - 2)),
        (x, y),
        (sp.S.One, sp.S.One),
        sp.S.true,
        sp.S.true,
    )
    assert result[1] == sp.cos(1)
    assert (
        nonzero_sign_composition(
            sp.sign(-1 + sp.I * x), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        signed_root_composition(
            1 / signed_root(x * y), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )


@pytest.mark.parametrize("head", [sp.sin, sp.cos, sp.exp, sp.atan])
def test_rational_zero_head(head):
    inner = x**3 / (x * x + y * y)
    result = rational_zero_composition(head(inner), (x, y), zero, sp.S.true, sp.S.true)
    assert result[0] is LimitStatus.PROVED
    assert result[1] == head(0)
    assert (
        rational_zero_composition(
            head(x * y / (x * x + y * y)), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )


def test_outer_discontinuity():
    inner = x**3 / (x * x + y * y)
    assert (
        rational_zero_composition(
            sp.sign(sp.sin(inner)), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        rational_zero_composition(sp.sin(inner), (x, y), zero, x > 0, sp.S.true) is None
    )


@pytest.mark.parametrize(
    "expr",
    [
        x * y / (x + y**3),
        signed_root(x) * y**2 / (x + y**3),
        x * y / signed_root(3 * x + 2 * y),
        sp.Piecewise((x * y / (x + y**3), sp.Ne(x + y**3, 0)), (7, True)),
    ],
)
def test_binomial_sequences(expr):
    result = binomial_pole_conflict(expr, (x, y), zero, sp.S.true, sp.S.true)
    assert result[0] is LimitStatus.DOES_NOT_EXIST
    assert set(e.value for e in result[2]) in ({0, sp.oo}, {0, -sp.oo})
    for evidence in result[2]:
        sequence = dict(evidence.substitutions)
        j = next(iter(set().union(*(v.free_symbols for v in sequence.values()))))
        assert tuple(sp.limit(sequence[v], j, sp.oo) for v in (x, y)) == zero
        attained = expr.subs(sequence, simultaneous=True)
        if attained.func is sp.Piecewise:
            branch, condition = attained.args[0]
            assert sp.simplify(condition) is sp.S.true
            attained = branch
        assert sp.limit(attained, j, sp.oo) == evidence.value
        for n in (1, 2, 5):
            assert attained.subs(j, n).has(sp.zoo, sp.nan) is False


def test_binomial_scope():
    assert (
        binomial_pole_conflict(
            x**2 * y**2 / (x * x + y * y), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )
    expr = sp.Piecewise((x * y / (x + y**3), sp.Ne(x - y, 0)), (0, True))
    assert binomial_pole_conflict(expr, (x, y), zero, sp.S.true, sp.S.true) is None


@pytest.mark.parametrize("power", [1, 3, 5])
def test_exponential_opposite_rays(power):
    expr = (x + y) * sp.exp(-1 / (x + y) ** power)
    result = exponential_linear_pole_conflict(expr, (x, y), zero, sp.S.true, sp.S.true)
    assert tuple(e.value for e in result[2]) == (0, -sp.oo)
    for evidence in result[2]:
        sequence = dict(evidence.substitutions)
        j = next(iter(set().union(*(v.free_symbols for v in sequence.values()))))
        assert sp.simplify((x + y).subs(sequence)) != 0
        assert (
            sp.limit(expr.subs(sequence, simultaneous=True), j, sp.oo) == evidence.value
        )
    assert (
        exponential_linear_pole_conflict(expr, (x, y), zero, x > 0, sp.S.true) is None
    )


def test_exponential_even_pole():
    assert (
        exponential_linear_pole_conflict(
            (x + y) * sp.exp(-1 / (x + y) ** 2), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )


def test_radical_ray_conflict():
    expr = sp.Abs(sp.sin(x * y)) ** sp.Rational(1, 15) / sp.sqrt(x * x + y * y)
    result = elementary_ray_conflict(expr, (x, y), zero, sp.S.true, sp.S.true)
    assert tuple(e.value for e in result[2]) == (0, sp.oo)
    for evidence in result[2]:
        sequence = dict(evidence.substitutions)
        j = next(iter(set().union(*(v.free_symbols for v in sequence.values()))))
        assert (
            sp.limit(expr.subs(sequence, simultaneous=True), j, sp.oo) == evidence.value
        )
        assert sp.simplify((x * x + y * y).subs(sequence)) != 0


def test_leading_cancellation():
    t = sp.Symbol("t", positive=True)
    assert ray_leading_term(sp.sin(t) - t, t) is None
    assert ray_leading_term(1 - sp.cos(t), t) == (sp.Rational(1, 2), 2)
    assert ray_leading_term(sp.log(t), t) is None
    assert (
        elementary_ray_conflict(
            x**2 * y**2 / (x * x + y * y), (x, y), zero, sp.S.true, sp.S.true
        )
        is None
    )


def test_flat_product_rate():
    result = limit(
        sp.exp(-1 / ((x * x + y * y) * (x**4 + y**4))), (x, y), zero, return_result=True
    )
    assert result.status is LimitStatus.PROVED
    assert result.value == 0
