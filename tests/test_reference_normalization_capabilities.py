import sympy as sp

from asymptotic.reference_normalization import (
    normalize_complex_argument,
    normalize_derivative,
    normalize_domain_membership,
    normalize_modular,
    real_algebraic_root,
)


def test_exact_real_algebraic_root():
    x = sp.symbols("x", real=True)
    r = real_algebraic_root(x**3 - 2, x, 0)
    assert sp.simplify(r**3 - 2) == 0 and r.is_real


def test_real_membership_normalizes_for_real_symbol():
    x = sp.symbols("x", real=True)
    assert normalize_domain_membership(x, sp.S.Reals) is sp.S.true


def test_complex_argument_is_native_branch_expression():
    z = sp.symbols("z")
    assert normalize_complex_argument(z) == sp.arg(z)


def test_modular_is_native_expression():
    x = sp.symbols("x")
    assert normalize_modular(x, 3) == sp.Mod(x, 3)


def test_derivative_normalizes_before_limit():
    x = sp.symbols("x")
    assert normalize_derivative(sp.sin(x), x) == sp.cos(x)
