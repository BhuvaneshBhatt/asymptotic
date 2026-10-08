import sympy as sp

from asymptotic.hyperasymptotics import (
    terminant,
)
from asymptotic.uniform_special_expansions import (
    airy_ai_exponentially_improved,
    airy_stokes_expansion,
    bessel_j_exponentially_improved,
    bessel_j_large_order,
    bessel_j_prime_large_order,
    bessel_j_transition,
    bessel_y_exponentially_improved,
    bessel_y_large_order,
    bessel_y_prime_large_order,
    hankel_exponentially_improved,
    hankel_large_order,
    optimally_truncated_airy_ai,
    rescaled_terminant,
)


def test_bessel_uniform_first_pair_has_olver_coefficients():
    from asymptotic.olver_coefficients import olver_coefficient

    nu = sp.symbols("nu", positive=True)
    z = sp.Rational(1, 2)
    result = bessel_j_large_order(nu, z)
    q = sp.sqrt(1 - z**2)
    eta = sp.log((1 + q) / z) - q
    zeta = (sp.Rational(3, 2) * eta) ** sp.Rational(2, 3)
    prefactor = (4 * zeta / (1 - z**2)) ** sp.Rational(1, 4)
    expected = prefactor * (
        sp.airyai(nu ** sp.Rational(2, 3) * zeta) / nu ** sp.Rational(1, 3)
        + sp.airyaiprime(nu ** sp.Rational(2, 3) * zeta)
        * olver_coefficient("B", 0, z)
        / nu ** sp.Rational(5, 3)
    )
    assert sp.simplify(result.prefix - expected) == 0
    assert result.turning_point.turning_point_uniform
    assert result.certified


def test_bessel_uniform_formula_is_finite_at_turning_point():
    from asymptotic.olver_coefficients import olver_coefficient

    nu = sp.symbols("nu", positive=True)
    result = bessel_j_large_order(nu, sp.S.One)
    expected = sp.real_root(2, 3) * (
        sp.airyai(0) / nu ** sp.Rational(1, 3)
        + sp.airyaiprime(0) * olver_coefficient("B", 0, 1) / nu ** sp.Rational(5, 3)
    )
    assert sp.simplify(result.prefix - expected) == 0
    assert result.certified


def test_bessel_transition_matches_airy_scaling():
    nu = sp.symbols("nu", positive=True)
    a = sp.symbols("a", real=True)
    result = bessel_j_transition(nu, a)
    expected = (
        sp.real_root(2, 3)
        * sp.airyai(-sp.real_root(2, 3) * a)
        / nu ** sp.Rational(1, 3)
    )
    assert sp.simplify(result.prefix - expected) == 0
    assert result.certified


def test_symbolic_large_parameter_without_sign_is_not_certified():
    nu = sp.symbols("nu")
    result = bessel_j_large_order(nu, sp.Rational(1, 2))
    assert not result.certified
    assert not result.turning_point.hypotheses_verified


def test_higher_olver_terms_are_generated_and_certified():
    nu = sp.symbols("nu", positive=True)
    result = bessel_j_large_order(nu, sp.Rational(1, 2), terms=2)
    assert result.certified
    assert not result.turning_point.obligations
    assert result.prefix.has(sp.airyaiprime)


def test_airy_stokes_rays_and_anti_stokes_rays_are_explicit():
    z = sp.symbols("z", positive=True)
    result = airy_stokes_expansion(z)
    stokes = result.stokes[0]
    assert stokes.stokes_rays == (-2 * sp.pi / 3, 2 * sp.pi / 3)
    assert stokes.anti_stokes_rays == (-sp.pi / 3, sp.pi / 3, sp.pi)
    assert result.certified


def test_exponential_improvement_records_optimal_and_terminant():
    z = sp.symbols("z", positive=True)
    result = optimally_truncated_airy_ai(z)
    assert result.exponentially_improved
    assert result.optimal_truncation_index is not None
    assert result.exponential_error_scale is not None
    assert result.prefix.has(sp.uppergamma)
    assert result.certified
    assert result.remainder_certificate.hypotheses_verified


def test_airy_first_two_coefficients_agree_with_fixed_order_provider():
    z = sp.symbols("z", positive=True)
    result = airy_stokes_expansion(z, terms=2)
    xi = sp.Rational(2, 3) * z ** sp.Rational(3, 2)
    expected = (
        sp.exp(-xi)
        / (2 * sp.sqrt(sp.pi) * z ** sp.Rational(1, 4))
        * (1 - sp.Rational(5, 72) / xi)
    )
    assert sp.simplify(result.prefix - expected) == 0


def test_bessel_y_uniform_first_pair_uses_bi_and_bi_prime():
    nu = sp.symbols("nu", positive=True)
    result = bessel_y_large_order(nu, sp.S.One)
    assert result.prefix.has(sp.gamma)
    assert result.certified


def test_hankel_uniform_first_pair_uses_rotated_airy_and_derivative():
    nu = sp.symbols("nu", positive=True)
    result = hankel_large_order(nu, sp.S.One, kind=1)
    assert result.prefix.has(sp.gamma)
    assert result.certified


def test_terminant_uses_incomplete_gamma_definition():
    p, z = sp.symbols("p z")
    expected = sp.gamma(p) * sp.uppergamma(1 - p, z) / (2 * sp.pi)
    assert terminant(p, z) == expected
    assert sp.simplify(rescaled_terminant(p, z) - sp.exp(z) * expected) == 0


def test_olver_turning_values_are_removable():
    from asymptotic.olver_coefficients import olver_coefficient

    assert olver_coefficient("A", 0, 1) == 1
    assert olver_coefficient("A", 1, 1) == -sp.Rational(1, 225)
    assert olver_coefficient("B", 0, 1) == sp.real_root(2, 3) / 70
    assert olver_coefficient("C", 0, 1) == sp.real_root(4, 3) / 10
    assert olver_coefficient("D", 0, 1) == 1
    assert olver_coefficient("D", 1, 1) == sp.Rational(23, 3150)


def test_debye_polynomial_recurrence_matches_known_first_terms():
    from asymptotic.olver_coefficients import debye_u, debye_v

    p = sp.Symbol("p")
    assert sp.expand(debye_u(1) - (3 * p - 5 * p**3) / 24) == 0
    assert sp.expand(debye_v(1) - (-9 * p + 7 * p**3) / 24) == 0


def test_bessel_derivative_uses_c_and_d_coefficients_at_turning_point():
    nu = sp.symbols("nu", positive=True)
    one = bessel_j_prime_large_order(nu, 1, terms=1)
    two = bessel_j_prime_large_order(nu, 1, terms=2)
    assert one.certified and two.certified
    assert sp.simplify(two.prefix - one.prefix) != 0


def test_y_derivative_uniform_expansion_is_certified():
    nu = sp.symbols("nu", positive=True)
    result = bessel_y_prime_large_order(nu, sp.Rational(3, 4), terms=2)
    assert result.certified
    assert result.prefix.has(sp.airybi) or result.prefix.has(sp.airybiprime)


def test_airy_terminant_remainder_has_theorem_scale():
    z = sp.symbols("z", positive=True)
    result = airy_ai_exponentially_improved(z, reexpansion_terms=3)
    cert = result.remainder_certificate
    xi = sp.Rational(2, 3) * z ** sp.Rational(3, 2)
    assert result.certified
    assert cert.truncation_index == sp.floor(2 * sp.Abs(xi))
    assert cert.reexpansion_terms == 3
    expected_scale = (
        sp.Abs(sp.exp(-xi) / (2 * sp.sqrt(sp.pi) * z ** sp.Rational(1, 4)))
        * sp.exp(-2 * sp.Abs(xi))
        * sp.Abs(xi) ** -3
    )
    assert sp.simplify(cert.remainder_scale - expected_scale) == 0


def test_hankel_terminant_remainder_is_certified_on_positive_axis():
    z = sp.symbols("z", positive=True)
    result = hankel_exponentially_improved(sp.Rational(1, 3), z, kind=1)
    assert result.certified
    assert result.remainder_certificate.hypotheses_verified
    assert result.prefix.has(sp.uppergamma)


def test_olver_turning_value_table_through_second_pair():
    from asymptotic.olver_coefficients import olver_coefficient

    assert olver_coefficient("A", 2, 1) == sp.Rational(151439, 218295000)
    assert olver_coefficient("B", 1, 1) == (
        -sp.Rational(1213, 1023750) * sp.real_root(2, 3)
    )
    assert olver_coefficient("C", 1, 1) == (
        -sp.Rational(947, 693000) * sp.real_root(4, 3)
    )
    assert olver_coefficient("D", 2, 1) == -sp.Rational(604523, 644962500)


def test_second_olver_pair_improves_representative_large_order_value():
    nu = sp.Integer(40)
    z = sp.Rational(4, 5)
    exact = sp.N(sp.besselj(nu, nu * z), 40)
    first = sp.N(bessel_j_large_order(nu, z, terms=1).prefix, 40)
    second = sp.N(bessel_j_large_order(nu, z, terms=2).prefix, 40)
    assert abs(second - exact) < abs(first - exact)


def test_bessel_j_and_y_terminant_combinations_are_certified():
    z = sp.symbols("z", positive=True)
    j = bessel_j_exponentially_improved(sp.Rational(1, 3), z)
    y = bessel_y_exponentially_improved(sp.Rational(1, 3), z)
    assert j.certified and y.certified
    assert j.prefix.has(sp.uppergamma)
    assert y.prefix.has(sp.uppergamma)
    assert j.remainder_certificate.hypotheses_verified
    assert y.remainder_certificate.hypotheses_verified
