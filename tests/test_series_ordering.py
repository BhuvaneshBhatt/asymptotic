"""Exact exponent ordering and product frontier correctness."""

from math import isqrt

import pytest
import sympy as sp

from asymptotic._ordering import exponent_sort_key
from asymptotic.multiseries import (
    MultiseriesTerm,
    add_term_streams,
    multiply_term_lists,
)


def test_close_rationals():
    low = 1 + sp.Rational(1, 2**80)
    high = 1 + sp.Rational(2, 2**80)
    assert float(low) == float(high) == 1
    assert sorted([high, low, sp.S.One], key=exponent_sort_key) == [1, low, high]


def test_float_precision():
    low = sp.Float("1.00000000000000000001", 80)
    high = sp.Float("1.00000000000000000002", 80)
    assert float(low) == float(high)
    assert sorted([high, low], key=exponent_sort_key) == [low, high]


def test_algebraic_bracket():
    denominator = 2**80
    numerator = isqrt(2 * denominator**2)
    low, high = (
        sp.Rational(numerator, denominator),
        sp.Rational(numerator + 1, denominator),
    )
    assert low**2 < 2 < high**2
    assert sorted([high, sp.sqrt(2), low], key=exponent_sort_key) == [
        low,
        sp.sqrt(2),
        high,
    ]


def test_streams_keep_distinct_powers():
    low = 1 + sp.Rational(1, 2**80)
    a = [MultiseriesTerm(sp.S.One, sp.Integer(2))]
    b = [MultiseriesTerm(low, sp.Integer(3))]
    assert list(add_term_streams(a, b)) == a + b


def test_product_close_exponents():
    low = 1 + sp.Rational(1, 2**80)
    high = 1 + sp.Rational(2, 2**80)
    a = [MultiseriesTerm(sp.S.Zero, sp.S.One), MultiseriesTerm(high, sp.Integer(2))]
    b = [MultiseriesTerm(sp.S.Zero, sp.S.One), MultiseriesTerm(low, sp.Integer(3))]
    assert multiply_term_lists(a, b, 3) == [
        MultiseriesTerm(sp.S.Zero, sp.S.One),
        MultiseriesTerm(low, sp.Integer(3)),
        MultiseriesTerm(high, sp.Integer(2)),
    ]


@pytest.mark.parametrize("count", [2, 3, 4])
def test_cancelled_frontier(count):
    a = [MultiseriesTerm(sp.S.Zero, sp.S.One), MultiseriesTerm(sp.S.One, -sp.S.One)]
    b = [MultiseriesTerm(sp.S.Zero, sp.S.One), MultiseriesTerm(sp.S.One, sp.S.One)]
    assert multiply_term_lists(a, b, count) == [
        MultiseriesTerm(sp.S.Zero, sp.S.One),
        MultiseriesTerm(sp.Integer(2), -sp.S.One),
    ]
