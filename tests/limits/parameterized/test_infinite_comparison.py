"""Generic infinite path values do not prove parameter-wise disagreement."""

import sympy as sp

from asymptotic._limit_composition import _exact_equal


def test_infinite_coefficient_cell():
    a = sp.Symbol("a", real=True)
    assert _exact_equal(a * sp.oo, 0) is None
    assert _exact_equal(sp.Mul(1 - sp.cosh(a), sp.oo, evaluate=False), 0) is None
    assert _exact_equal((a * a + 1) * sp.oo, 0) is False


def test_finite_parameter_comparison():
    a = sp.Symbol("a", real=True)
    assert _exact_equal(a - 1, 0) is None
    assert _exact_equal(a * a + 1, 0) is False
    assert _exact_equal((a + 1) ** 2, a * a + 2 * a + 1) is True
