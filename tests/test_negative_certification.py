"""Each certifier must remain conservative when a key hypothesis fails."""

import sympy as sp

from asymptotic import AsymptoticRemainder, RemainderKind
from asymptotic.remainder_theorems import (
    certify_algebraic_substitution_remainder,
    certify_differentiation_remainder,
    certify_green_inverse_operator_remainder,
    certify_inverse_remainder,
    certify_nonlinear_lifting_remainder,
    certify_quotient_remainder,
    certify_reciprocal_remainder,
    certify_unary_composition_remainder,
)


def assert_unknown(cert) -> None:
    assert not cert.certified
    assert cert.conclusion.kind is RemainderKind.UNKNOWN


def test_reciprocal_refuses_prefix_with_uncontrolled_zeros():
    x = sp.symbols("x", real=True)
    assert_unknown(
        certify_reciprocal_remainder(sp.sin(x), AsymptoticRemainder.exact_zero(x, sp.oo))
    )


def test_quotient_refuses_denominator_uncertainty_as_large_as_prefix():
    x = sp.symbols("x", positive=True)
    exact = AsymptoticRemainder.exact_zero(x, sp.oo)
    large = AsymptoticRemainder.big_o(1, x, sp.oo)
    assert_unknown(certify_quotient_remainder(1, 1, exact, large))


def test_algebraic_substitution_refuses_rational_pole_at_prefix():
    x, z = sp.symbols("x z", positive=True)
    remainder = AsymptoticRemainder.big_o(x**-1, x, sp.oo)
    cert = certify_algebraic_substitution_remainder(
        1 / z, z, 0, remainder, output_variable=x, point=sp.oo
    )
    assert_unknown(cert)


def test_composition_refuses_function_singular_at_expansion_point():
    x, z = sp.symbols("x z", positive=True)
    remainder = AsymptoticRemainder.big_o(x**-1, x, sp.oo)
    cert = certify_unary_composition_remainder(
        sp.log(z), z, 0, remainder, output_variable=x, point=sp.oo
    )
    assert_unknown(cert)


def test_green_refuses_center_spectrum_without_exponential_dichotomy():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + delta(x)
    cert, _ = certify_green_inverse_operator_remainder(sp.exp(-x), operator, delta, x, sp.oo)
    assert_unknown(cert)


def test_green_refuses_coefficients_without_constant_limit():
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.sin(x) * sp.diff(delta(x), x) - delta(x)
    cert, _ = certify_green_inverse_operator_remainder(sp.exp(-x / 2), operator, delta, x, sp.oo)
    assert_unknown(cert)


def test_inverse_refuses_incorrect_inverse_prefix():
    x, y = sp.symbols("x y", positive=True)
    cert = certify_inverse_remainder(x**2, x, y, 0, source_point=0, target_point=0)
    assert_unknown(cert)


def test_nonlinear_lifting_refuses_degenerate_linearization():
    x = sp.symbols("x", positive=True)
    cert = certify_nonlinear_lifting_remainder(x**-2, 0, x, sp.oo)
    assert_unknown(cert)


def test_differentiation_refuses_unproved_derivative_control():
    x = sp.symbols("x", positive=True)
    remainder = AsymptoticRemainder.big_o(sp.sin(x), x, sp.oo)
    cert = certify_differentiation_remainder(remainder)
    assert_unknown(cert)
