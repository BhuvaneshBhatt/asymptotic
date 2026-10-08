import mpmath as mp
import sympy as sp

from asymptotic import (
    hyperasymptotic_series,
)
from asymptotic.hyperasymptotics import (
    ExponentialScale,
    terminant_reexpand,
)
from asymptotic.olver_coefficients import debye_u
from asymptotic.turning_point_families import (
    associated_legendre_turning_point,
    parabolic_cylinder_turning_point,
    whittaker_turning_point,
)
from asymptotic.uniform_special_expansions import (
    modified_bessel_i_large_order,
    modified_bessel_k_large_order,
)
from asymptotic.uniform_zeros import (
    bessel_zero_large_order,
)
from asymptotic.variation_bounds import (
    debye_variation,
    polynomial_segment_variation,
)


def test_polynomial_total_variation_majorant_is_rigorous():
    p = sp.Symbol("p")
    cert = polynomial_segment_variation(3 * p - 5 * p**3, p, 0, sp.Rational(1, 2))
    assert cert.hypotheses_verified
    # Coefficientwise integration majorizes the endpoint change.
    assert cert.bound >= abs(sp.Rational(3, 2) - sp.Rational(5, 8))


def test_debye_variation_is_computable_without_symbolic_constant():
    cert = debye_variation(debye_u(3), sp.Rational(2, 5))
    assert cert.hypotheses_verified
    assert not cert.bound.has(sp.Symbol("C_N"))
    assert cert.bound.is_rational


def test_modified_bessel_remainder_is_a_numerical_enclosure():
    nu = sp.Integer(30)
    for builder in (modified_bessel_i_large_order, modified_bessel_k_large_order):
        expansion = builder(nu, sp.Rational(3, 2), terms=3)
        bound = expansion.remainder_bound
        assert bound.enclosure_certified
        assert not bound.constant.has(sp.Symbol("C_N"))
        actual = sp.Abs(sp.N(expansion.expression - expansion.prefix, 40))
        assert actual <= sp.N(bound.bound, 40)


def test_generic_terminant_level_tracks_saddle_geometry():
    chi = sp.Symbol("chi", positive=True)
    scale = ExponentialScale(chi, 2, 1, (0,), (sp.pi / 2,), "adjacent")
    level = terminant_reexpand([1, sp.Rational(1, 3)], scale, reexpansion_terms=2)
    assert level.singulant == chi
    assert level.hypotheses_verified
    assert level.multiplier.has(sp.uppergamma)


def test_second_level_hyperasymptotics_is_bounded_and_recursive():
    chi = sp.Integer(8)
    scales = (
        ExponentialScale(chi, 1, 1, (0,), (sp.pi / 2,), "s1"),
        ExponentialScale(2 * chi, sp.Rational(1, 2), -1, (sp.pi,), (-sp.pi / 2,), "s2"),
    )
    result = hyperasymptotic_series(
        1,
        lambda level, count: tuple(sp.factorial(k + level) for k in range(count)),
        scales,
        levels=2,
        reexpansion_terms=2,
        return_result=True,
    )
    assert len(result.levels) == 2
    assert result.levels[0].singulant == chi
    assert result.levels[1].singulant == 2 * chi
    assert result.certified


def test_hyperasymptotic_depth_is_bounded():
    scale = ExponentialScale(8, 1, 1, (), ())
    try:
        hyperasymptotic_series(
            1, lambda level, count: (1,), (scale,), levels=4, return_result=True
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unbounded recursion was accepted")


def test_large_order_bessel_zero_uses_airy_coordinate():
    approx = bessel_zero_large_order(sp.Integer(40), 1)
    mp.mp.dps = 30
    exact = mp.besseljzero(40, 1)
    assert abs(float(approx.zero) - float(exact)) < 0.02
    assert approx.residual < sp.Rational(1, 100)


def test_large_order_derivative_zero_uses_airy_prime_coordinate():
    approx = bessel_zero_large_order(sp.Integer(40), 1, derivative=True)
    mp.mp.dps = 30
    exact = mp.besseljzero(40, 1, derivative=1)
    assert abs(float(approx.zero) - float(exact)) < 0.02


def test_parabolic_cylinder_uses_shared_airy_turning_point_model():
    result = parabolic_cylinder_turning_point(sp.Integer(8), sp.Rational(4, 5))
    assert result.certified
    assert result.family.coefficient_family == "Airy A/B"
    assert result.leading_approximation.has(sp.airyai)


def test_whittaker_and_legendre_adapters_expose_turning_geometry():
    whittaker = whittaker_turning_point(sp.Integer(20), sp.Integer(3), sp.Symbol("x"))
    legendre = associated_legendre_turning_point(
        sp.Integer(30), sp.Integer(4), sp.Symbol("x")
    )
    assert len(whittaker.turning_points) == 2
    assert len(legendre.turning_points) == 2
    assert whittaker.coefficient_family == "Airy A/B"
    assert "transition" in legendre.coefficient_family


def test_bessel_olver_bound_has_no_unspecified_constant():
    from asymptotic.uniform_special_expansions import bessel_j_large_order

    result = bessel_j_large_order(sp.Integer(30), sp.S.One, terms=1)
    assert result.remainder_bound.constant == 2
    assert not result.remainder_bound.bound.has(sp.Symbol("C_N"))


def test_other_family_adapters_produce_actual_leading_approximations():
    from asymptotic.turning_point_families import (
        associated_legendre_large_degree,
        whittaker_m_large_kappa,
    )

    whittaker = whittaker_m_large_kappa(sp.Integer(30), sp.Integer(2), sp.Integer(3))
    legendre = associated_legendre_large_degree(
        sp.Integer(30), sp.Integer(2), sp.Rational(1, 2)
    )
    assert whittaker.leading_approximation.has(sp.besselj)
    assert legendre.leading_approximation.has(sp.besselj)
    assert whittaker.certified and legendre.certified


def test_modified_bessel_enclosures_hold_on_representative_grid():
    for nu in (sp.Integer(20), sp.Integer(40)):
        for z in (sp.Rational(1, 2), sp.S.One, sp.Integer(2)):
            for builder in (
                modified_bessel_i_large_order,
                modified_bessel_k_large_order,
            ):
                expansion = builder(nu, z, terms=3)
                actual = sp.Abs(sp.N(expansion.expression - expansion.prefix, 35))
                assert actual <= sp.N(expansion.remainder_bound.bound, 35)
