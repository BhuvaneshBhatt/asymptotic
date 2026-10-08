"""Tail remainders, branch sides and attained pole-avoiding clusters."""

import pytest
import sympy as s

from asymptotic import (
    analytic_limit,
    cluster_set,
    complex_limit,
    complex_ray_limit,
    limit,
)
from asymptotic.complex_ray_germs import special_function_ray_certificate
from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
from asymptotic.limit_models import LimitStatus
from asymptotic.special_function_germs import special_function_certificate
from asymptotic.special_functions import (
    EllipticNome,
    InverseEllipticNome,
    JacobiAmplitude,
    OwenT,
    ScorerGi,
    ScorerGiPrime,
    ScorerHi,
    ScorerHiPrime,
)


@pytest.mark.parametrize(
    "function,point,weight,value",
    [
        (ScorerGi, s.oo, 1, 0),
        (ScorerGi, s.oo, lambda x: x, 1 / s.pi),
        (ScorerGiPrime, s.oo, lambda x: x**2, -1 / s.pi),
        (ScorerHi, -s.oo, lambda x: x, -1 / s.pi),
        (ScorerHiPrime, -s.oo, lambda x: x**2, 1 / s.pi),
    ],
)
def test_weighted_scorer_tail(function, point, weight, value):
    x = s.Symbol("x")
    factor = weight(x) if callable(weight) else weight
    assert limit(factor * function(x), x, point) == value


def test_rational_scorer_argument():
    x = s.Symbol("x")
    assert limit(x * ScorerGi((x * x + 1) / (2 * x)), x, s.oo) == 2 / s.pi
    assert limit(x**3 * ScorerGi(x) - x * x / s.pi, x, s.oo) == 0


def test_scorer_remainder_guard():
    x = s.Symbol("x")
    assert (
        special_function_certificate(x**4 * ScorerGi(x) - x**3 / s.pi, x, s.oo) is None
    )
    assert special_function_certificate(ScorerHi(x), x, s.oo) is None
    assert special_function_certificate(ScorerGi(x), x, -s.oo) is None
    assert s.diff(ScorerGi(x), x, 2) == x * ScorerGi(x) - 1 / s.pi
    assert s.diff(ScorerHi(x), x, 2) == x * ScorerHi(x) + 1 / s.pi


@pytest.mark.parametrize(
    "ray,sign", [(s.I, 1), (-s.I, -1), (1 + s.I, 1), (1 - s.I, -1)]
)
def test_nome_cut(ray, sign):
    z = s.Symbol("z")
    assert complex_ray_limit(EllipticNome(z), z, 2, ray=ray) == sign * s.I * s.exp(
        -s.pi / 2
    )


def test_nome_definition_and_inverse():
    z = s.Symbol("z")
    assert EllipticNome(z).rewrite(s.exp) == s.exp(
        -s.pi * s.elliptic_k(1 - z) / s.elliptic_k(z)
    )
    assert limit(s.diff(InverseEllipticNome(z), z), z, 0) == 16
    assert (
        complex_limit(EllipticNome(z), z, 2, return_result=True).status
        is LimitStatus.DOES_NOT_EXIST
    )


@pytest.mark.parametrize("center,ray", [(s.I, 1), (-s.I, 1), (s.I, -s.I), (-s.I, s.I)])
def test_owen_log_pole(center, ray):
    z = s.Symbol("z")
    assert complex_ray_limit(OwenT(2, z), z, center, ray=ray) == DirectionalInfinity(
        center
    )


def test_owen_cut_and_parameter_guards():
    t = s.Symbol("t", positive=True)
    assert special_function_ray_certificate(OwenT(2, s.I + s.I * t), t) is None
    assert special_function_ray_certificate(OwenT(1 / t, s.I + t), t) is None
    z = s.Symbol("z")
    assert analytic_limit(
        OwenT(2, s.I + (1 + 3 * s.I) * z), z, 0, direction="+"
    ) == DirectionalInfinity(s.I)
    integral = OwenT(2, z).rewrite(s.Integral)
    assert integral.has(s.Integral)
    assert (
        s.simplify(
            s.diff(integral, z) - s.exp(-2 * (1 + z * z)) / (2 * s.pi * (1 + z * z))
        )
        == 0
    )


@pytest.mark.parametrize("ray,value", [(s.I, 2), (-s.I, 1), (1, 2), (-1, 2)])
def test_circular_piecewise_ray(ray, value):
    z = s.Symbol("z")
    expression = s.Piecewise((1, s.Abs(z) <= 1), (2, True))
    assert complex_ray_limit(expression, z, s.I, ray=ray) == value


def test_circular_piecewise_whole_plane():
    z = s.Symbol("z")
    expression = s.Piecewise((1, s.Abs(z) <= 1), (2, True))
    result = complex_limit(expression, z, s.I, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [item for item in result.evidence if item.substitutions]
    assert {item.value for item in witnesses} == {1, 2}


def test_reflection_cluster_attainment():
    x = s.Symbol("x", real=True)
    z = x / (2 * s.pi)
    expression = (
        2
        * s.pi
        * x
        * (-x * x * s.polygamma(1, z) + 2 * s.pi * (x + 2 * s.pi))
        * s.sin(x)
        + (x**3 * s.polygamma(2, z) + 4 * s.pi**2 * (x + 4 * s.pi)) * (s.cos(x) - 1)
    ) / (4 * s.pi**3 * x)
    result = cluster_set(expression, x, -s.oo, return_result=True)
    assert result.cluster_set == s.Interval(-1, 1)
    item = result.evidence[0]
    sequence = item.substitutions[0][1]
    theta = next(
        symbol for symbol in sequence.free_symbols if symbol.is_integer is not True
    )
    n = next(symbol for symbol in sequence.free_symbols if symbol.is_integer is True)
    for phase, value in [(s.pi / 2, 1), (s.pi, 0), (3 * s.pi / 2, -1)]:
        attained = sequence.subs(theta, phase)
        assert s.limit(attained, n, s.oo) == -s.oo
        assert s.sin(attained) == value
        assert s.simplify(attained / (2 * s.pi)).is_integer is False
    p1, p2 = s.symbols("p1 p2")
    reflected = expression.xreplace(
        {
            s.polygamma(1, z): s.pi**2 / s.sin(x / 2) ** 2 - p1,
            s.polygamma(2, z): p2 - 2 * s.pi**3 * s.cos(x / 2) / s.sin(x / 2) ** 3,
        }
    )
    regular = (x * x * p1 + 2 * s.pi * (x + 2 * s.pi)) * s.sin(x) / (2 * s.pi**2) + (
        x**3 * p2 + 4 * s.pi**2 * (x + 4 * s.pi)
    ) * (s.cos(x) - 1) / (4 * s.pi**3 * x)
    assert s.trigsimp(s.expand(reflected - regular), method="fu") == 0


def test_amplitude_local_inverse():
    x = s.Symbol("x")
    phase = x / (1 + x * x)
    expression = (JacobiAmplitude(s.elliptic_f(phase, 2), 2) - phase) / x**8
    assert limit(expression, x, 0) == 0
    outside = JacobiAmplitude(s.elliptic_f(1 + x, 2), 2)
    assert special_function_certificate(outside, x, 0) is None


def test_compact_power_budget():
    x = s.Symbol("x", positive=True)
    assert special_function_certificate(ScorerGi((1 + x) ** 1000000), x, s.oo) is None
    expr = s.Piecewise(
        (1, s.Le((1 + x) ** 1000000, 2, evaluate=False)), (0, True), evaluate=False
    )
    assert special_function_ray_certificate(expr, x) is None
