import pytest
import sympy as s

from asymptotic import complex_ray_limit
from asymptotic.fixed_ray_branch_germs import (
    DirectionalInfinity,
    original_bessel_ray_certificate,
    rational_ray_pole_certificate,
)
from asymptotic.limit_models import LimitStatus

z = s.Symbol("z")


@pytest.mark.parametrize("ray", [s.I, -s.I, s.exp(s.I * s.pi / 4)])
def test_rational_pole_proves_modulus_and_requested_direction(ray):
    result = complex_ray_limit(1 / z, z, 0, ray=ray, return_result=True)
    assert result.status is LimitStatus.PROVED
    assert result.value == DirectionalInfinity(s.simplify(1 / ray))
    sample = (1 / z).subs(z, ray / s.Integer(1000))
    assert s.simplify(sample / s.Abs(sample) - 1 / ray) == 0
    assert result.evidence[-1].substitutions


def test_complex_sign_tracks_the_requested_ray():
    for ray in [s.I, -s.I, s.exp(s.I * s.pi / 4)]:
        result = complex_ray_limit(s.sign(z), z, 0, ray=ray, return_result=True)
        assert result.status is LimitStatus.PROVED
        assert s.simplify(result.value - ray) == 0


@pytest.mark.parametrize("function", [s.elliptic_k, s.elliptic_e])
@pytest.mark.parametrize("ray", [s.I, -s.I, s.exp(s.I * s.pi / 4)])
def test_elliptic_cut_matches_independent_complex_samples(function, ray):
    expr = function(z) - function(2)
    result = complex_ray_limit(expr, z, 2, ray=ray, return_result=True)
    assert result.status is LimitStatus.PROVED
    sample = expr.subs(z, 2 + ray / s.Integer(100000))
    assert abs(complex(s.N(sample - result.value, 40))) < 0.0001


@pytest.mark.parametrize("function", [s.besselj, s.besseli])
@pytest.mark.parametrize("ray", [s.I, -s.I])
def test_bessel_ratio_retains_continuation_nonzero_denominator(function, ray):
    nu = s.Rational(3, 5)
    expr = function(nu, z) / function(nu, -2)
    result = complex_ray_limit(expr, z, -2, ray=ray, return_result=True)
    assert result.status is LimitStatus.PROVED
    sample = expr.subs(z, -2 + ray / s.Integer(100000))
    assert abs(complex(s.N(sample - result.value, 40))) < 0.0001


def test_pole_and_bessel_providers_zero_strata():
    t = s.Symbol("t", positive=True)
    a = s.Symbol("a")
    assert rational_ray_pole_certificate(a / t, t) is None
    expr = s.besselj(0, z) / s.besselj(0, -3)
    assert original_bessel_ray_certificate(expr, z, s.Integer(-3), s.I) is None
