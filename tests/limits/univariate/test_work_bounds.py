"""Structural certificate bounds remain independent of mathematical proofs."""

import pytest
import sympy as sp

from asymptotic.local_tail_germs import budget


@pytest.mark.parametrize(
    "expression,threshold,expected",
    [
        (sp.S.One, 0, True),
        (sp.Symbol("x") ** 16, 0, False),
        (sp.Symbol("x") ** 16, 1, True),
        (sp.Symbol("x") ** 17, 120, False),
        (sp.Symbol("x") ** -17, 120, False),
        (sp.sin(sp.Symbol("x") ** 17), 120, False),
        (sp.Symbol("x") ** sp.Rational(17, 2), 120, True),
        (sp.exp(sp.Symbol("x")) / (1 + sp.Symbol("x")), 2, False),
        (sp.exp(sp.Symbol("x")) / (1 + sp.Symbol("x")), 3, True),
    ],
)
def test_work_bounds(expression, threshold, expected):
    assert bool(budget(expression, threshold)) is expected


def test_thresholds_are_independent():
    x = sp.Symbol("x")
    expression = sp.exp(x) / (1 + x)
    assert budget(expression, 20)
    assert not budget(expression, 0)
    assert budget(expression, 20)


def test_symbol_assumptions():
    real = sp.Symbol("x", real=True)
    positive = sp.Symbol("x", positive=True)
    assert real != positive
    for variable in (real, positive):
        assert budget(sp.sin(variable) / variable, 10)
        assert not budget(variable**17, 120)
