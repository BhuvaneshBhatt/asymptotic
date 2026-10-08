import sympy as sp

from asymptotic.complex_domains import (
    bessel_complex_domain,
    modified_bessel_complex_domain,
)
from asymptotic.olver_coefficients import (
    olver_coefficient,
    olver_zeta,
)
from asymptotic.uniform_special_expansions import (
    bessel_j_large_order,
    bessel_y_large_order,
    hankel_large_order,
    hankel_prime_large_order,
    modified_bessel_i_large_order,
    modified_bessel_i_prime_large_order,
    modified_bessel_k_large_order,
    modified_bessel_k_prime_large_order,
)


def test_olver_zeta_crosses_turning_point_onto_negative_real_axis():
    assert olver_zeta(sp.Rational(1, 2)).is_positive
    assert olver_zeta(sp.Integer(2)).is_negative


def test_complex_olver_conjugation():
    z = sp.Rational(4, 5) + sp.I / 5
    assert (
        abs(
            complex(sp.N(olver_zeta(sp.conjugate(z)) - sp.conjugate(olver_zeta(z)), 30))
        )
        < 1e-25
    )
    for family in "ABCD":
        delta = olver_coefficient(family, 1, sp.conjugate(z)) - sp.conjugate(
            olver_coefficient(family, 1, z)
        )
        assert abs(complex(sp.N(delta, 30))) < 1e-22


def test_negative_axis_is_a_branch_boundary():
    assert bessel_complex_domain(sp.Integer(-2)).component == "cut"
    assert not bessel_complex_domain(sp.Integer(-2)).hypotheses_verified


def test_one_sided_negative_axis_continuations_are_conjugate():
    eps = sp.Rational(1, 10**6)
    upper = olver_zeta(-2 + sp.I * eps)
    lower = olver_zeta(-2 - sp.I * eps)
    assert abs(complex(sp.N(lower - sp.conjugate(upper), 30))) < 1e-20


def test_complex_bessel_domain_is_certified_off_cut():
    z = sp.Rational(3, 4) + sp.I / 10
    domain = bessel_complex_domain(z)
    assert domain.hypotheses_verified
    assert domain.component == "upper"


def test_hankel_uses_all_requested_olver_pairs():
    nu = sp.symbols("nu", positive=True)
    z = sp.S.One
    one = hankel_large_order(nu, z, kind=1, terms=1)
    three = hankel_large_order(nu, z, kind=1, terms=2)
    assert sp.simplify(one.prefix - three.prefix) != 0
    assert three.certified


def test_hankel_identity_holds_coefficientwise():
    nu = sp.symbols("nu", positive=True)
    z = sp.S.One
    j = bessel_j_large_order(nu, z, terms=1).prefix
    y = bessel_y_large_order(nu, z, terms=1).prefix
    h = hankel_large_order(nu, z, kind=1, terms=1).prefix
    value = (h - j - sp.I * y).subs(nu, 40)
    assert abs(complex(sp.N(value, 30))) < 1e-25


def test_modified_bessel_i_and_k_share_debye_coefficients():
    nu = sp.symbols("nu", positive=True)
    z = sp.Rational(3, 2)
    i = modified_bessel_i_large_order(nu, z, terms=3)
    k = modified_bessel_k_large_order(nu, z, terms=3)
    assert i.certified and k.certified
    eta = sp.sqrt(13) / 2 + sp.log(3 / (2 + sp.sqrt(13)))
    assert sp.simplify(sp.diff(sp.log(i.prefix), nu) - eta).limit(nu, sp.oo) == 0
    assert k.remainder_bound.enclosure_certified


def test_modified_bessel_derivatives_are_available():
    nu = sp.symbols("nu", positive=True)
    z = sp.Rational(2, 3)
    assert modified_bessel_i_prime_large_order(nu, z, terms=2).certified
    assert modified_bessel_k_prime_large_order(nu, z, terms=2).certified


def test_modified_complex_sector_and_conjugation():
    nu = sp.symbols("nu", positive=True)
    z = 1 + sp.I / 5
    upper = modified_bessel_i_large_order(nu, z, terms=2)
    lower = modified_bessel_i_large_order(nu, sp.conjugate(z), terms=2)
    assert upper.certified and lower.certified
    delta = lower.prefix.subs(nu, 30) - sp.conjugate(upper.prefix.subs(nu, 30))
    assert abs(complex(sp.N(delta, 30))) < 1e-22


def test_modified_sector_boundary_is_not_certified():
    nu = sp.symbols("nu", positive=True)
    z = sp.exp(sp.I * sp.pi * sp.Rational(51, 100))
    assert not modified_bessel_i_large_order(nu, z).certified


def test_remainder_contract_distinguishes_order_from_enclosure():
    nu = sp.symbols("nu", positive=True)
    z = sp.Rational(5, 4)
    k = modified_bessel_k_large_order(nu, z, terms=3)
    i = modified_bessel_i_large_order(nu, z, terms=3)
    assert k.remainder_bound.order_certified
    assert k.remainder_bound.enclosure_certified
    assert i.remainder_bound.order_certified
    assert i.remainder_bound.enclosure_certified


def test_hankel_identity_numerically_across_turning_region():
    nu = sp.Integer(50)
    for z in (sp.Rational(4, 5), sp.S.One, sp.Rational(6, 5)):
        j = bessel_j_large_order(nu, z, terms=1).prefix
        y = bessel_y_large_order(nu, z, terms=1).prefix
        h = hankel_large_order(nu, z, kind=1, terms=1).prefix
        assert abs(complex(sp.N(h - j - sp.I * y, 30))) < 1e-24


def test_hankel_derivative_consumes_c_and_d_pairs():
    nu = sp.symbols("nu", positive=True)
    one = hankel_prime_large_order(nu, 1, kind=1, terms=1)
    two = hankel_prime_large_order(nu, 1, kind=1, terms=2)
    assert one.certified and two.certified
    assert sp.simplify(one.prefix - two.prefix) != 0


def test_modified_sector_boundaries_are_rejected_from_both_sides():
    for sign in (-1, 1):
        z = sp.exp(sign * sp.I * sp.pi * sp.Rational(51, 100))
        assert not modified_bessel_complex_domain(z).hypotheses_verified


def test_oscillatory_side_coefficients_remain_real():
    for family in "ABCD":
        value = sp.N(olver_coefficient(family, 1, sp.Rational(6, 5)), 30)
        assert abs(complex(value).imag) < 1e-25
