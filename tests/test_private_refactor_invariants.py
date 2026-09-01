"""Behavioral invariants for shared low-level helpers introduced by refactoring."""

import sympy as sp

from asymptotic._integer_utils import integer_lcm
from asymptotic._linear_ode_operator import linear_operator_coefficients
from asymptotic._ordering import exponent_sort_key


def test_integer_lcm_uses_standard_zero_and_sign_semantics():
    assert integer_lcm(6, 8) == 24
    assert integer_lcm(-6, 8) == 24
    assert integer_lcm(0, 8) == 0
    assert integer_lcm(0, 0) == 0


def test_exponent_sort_key_orders_numbers_before_symbolic_exponents():
    a = sp.symbols("a")
    values = [a, sp.Rational(3, 2), sp.Integer(-1), sp.Rational(1, 3)]
    ordered = sorted(values, key=exponent_sort_key)
    assert ordered[:3] == [-1, sp.Rational(1, 3), sp.Rational(3, 2)]
    assert ordered[-1] == a


def test_linear_operator_extraction_rejects_nonlinear_and_inhomogeneous_jets():
    x = sp.symbols("x")
    d = sp.Function("d")(x)
    operator = sp.diff(d, x, 2) + x * sp.diff(d, x) - d
    extracted = linear_operator_coefficients(operator, d, x)
    assert extracted is not None
    coeffs, order = extracted
    assert order == 2
    assert coeffs == (-1, x, 1)
    assert linear_operator_coefficients(d**2 + sp.diff(d, x), d, x) is None
    assert linear_operator_coefficients(1 + sp.diff(d, x), d, x) is None
