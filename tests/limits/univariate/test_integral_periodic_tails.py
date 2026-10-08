"""Real charts, attained discontinuities, and convergent integral cancellations."""

import pytest
import sympy as s

from asymptotic import SquareWave, limit
from asymptotic.discontinuous_tail_germs import discontinuous_periodic_tail_certificate
from asymptotic.flat_exponential_germs import flat_exponential_certificate
from asymptotic.integral_definition_germs import (
    appell_integral_continuity_certificate,
    fresnel_auxiliary_tail_certificate,
    gaussian_log_moment_certificate,
)
from asymptotic.limit_models import LimitStatus
from asymptotic.special_functions import AppellF1, FresnelF, FresnelG


@pytest.mark.parametrize("slope", [s.S.One, s.Rational(1, 2), -s.S.One])
@pytest.mark.parametrize("wave", [SquareWave, lambda z: (-1) ** s.floor(z)])
def test_attained_wave_sequences(slope, wave):
    x = s.Symbol("x", real=True)
    expression = (1 + 1 / (x - 3)) * wave(slope * x + s.Rational(2, 7)) + 2
    result = limit(expression, x, s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in result.evidence} == {1, 3}
    for evidence in result.evidence:
        variable, sequence = evidence.substitutions[0]
        assert variable == x
        n = next(iter(sequence.free_symbols))
        assert s.limit(sequence, n, s.oo) == s.oo
        for index in (10, 100):
            argument = sequence.subs(n, index)
            phase = slope * argument + s.Rational(2, 7)
            assert (2 * phase).is_integer is False
            assert argument != 3
            assert wave(phase) == evidence.value - 2


def test_wave_envelope_and_orientation():
    x = s.Symbol("x")
    assert limit(SquareWave(x) / s.log(x), x, s.oo) == 0
    assert limit(SquareWave(x) / x + 3, x, -s.oo) == 3
    result = limit((-1) ** s.floor(x), x, -s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    for evidence in result.evidence:
        sequence = evidence.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        assert s.limit(sequence, n, s.oo) == -s.oo
        assert (-1) ** s.floor(sequence.subs(n, 10)) == evidence.value


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: SquareWave(x) * s.sin(x),
        lambda x: SquareWave(x**2),
        lambda x: 1 / SquareWave(x),
        lambda x: SquareWave(x) / (s.Symbol("a") * x),
    ],
)
def test_wave_declines_unproved_coefficients(expression):
    x = s.Symbol("x", positive=True)
    assert (
        discontinuous_periodic_tail_certificate(expression(x), x, s.oo, s.true, s.true)
        is None
    )


def test_wave_domain_and_integer_contract():
    x = s.Symbol("x", integer=True)
    assert (
        discontinuous_periodic_tail_certificate(
            SquareWave(x / 2), x, s.oo, s.true, s.true
        )
        is None
    )
    x = s.Symbol("x", positive=True)
    assert (
        discontinuous_periodic_tail_certificate(
            SquareWave(x), x, s.oo, s.Eq(x, s.floor(x)), s.true
        )
        is None
    )


@pytest.mark.parametrize("order", [s.Symbol("a"), -100, 2 + 3 * s.I])
@pytest.mark.parametrize("degree", [2, 4, 16])
def test_flat_complex_powers(order, degree):
    x = s.Symbol("x")
    assert limit(x**order * s.exp(-3 / x**degree), x, 0) == 0


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: s.exp(-1 / x),
        lambda x: s.exp(1 / x**2),
        lambda x: x ** (1 / x) * s.exp(-1 / x**2),
        lambda x: x ** (1 / s.Symbol("a")) * s.exp(-1 / x**2),
        lambda x: s.sin(1 / x) * s.exp(-1 / x**2),
    ],
)
def test_flat_declines_other_scales(expression):
    x = s.Symbol("x", real=True)
    assert flat_exponential_certificate(expression(x), x, 0, s.true, s.true) is None


def _gaussian_expression(y):
    return s.erf(y) * s.log(y) - 2 * y * s.hyper(
        (s.S.Half, s.S.Half), (s.Rational(3, 2), s.Rational(3, 2)), -(y**2)
    ) / s.sqrt(s.pi)


@pytest.mark.parametrize("argument", [lambda x: x, lambda x: (x * x + 1) / (x - 1)])
def test_gaussian_log_moment(argument):
    x = s.Symbol("x", positive=True)
    expression = (2 + 1 / x) * _gaussian_expression(argument(x)) + 3
    expected = 3 - 2 * s.log(2) - s.EulerGamma
    assert s.simplify(limit(expression, x, s.oo) - expected) == 0


def test_gaussian_integral_identity():
    import mpmath as mp

    with mp.workdps(45):
        y = mp.mpf("3.5")
        expression = mp.erf(y) * mp.log(y) - 2 * y * mp.hyper(
            [mp.mpf("0.5")] * 2, [mp.mpf("1.5")] * 2, -y * y
        ) / mp.sqrt(mp.pi)
        integral = (
            2
            / mp.sqrt(mp.pi)
            * mp.quad(lambda t: mp.exp(-t * t) * mp.log(t), [0, 1, y])
        )
        assert abs(expression - integral) < mp.mpf("1e-40")


def test_gaussian_declines_complex_and_negative_tails():
    x = s.Symbol("x", positive=True)
    for argument in (-x, s.I * x, x + s.I, 1 / x):
        assert (
            gaussian_log_moment_certificate(
                _gaussian_expression(argument), x, s.oo, s.true, s.true
            )
            is None
        )


@pytest.mark.parametrize("function", [FresnelF, FresnelG])
def test_fresnel_definition_and_tail(function):
    x = s.Symbol("x", positive=True)
    assert limit(function((x * x + 1) / (x + 1)) * (1 + 1 / x) + 2, x, s.oo) == 2
    value = function(s.S.One).rewrite(s.fresnelc)
    expected = (
        s.fresnelc(1) - s.S.Half if function is FresnelF else s.S.Half - s.fresnels(1)
    )
    assert value == expected
    assert (
        fresnel_auxiliary_tail_certificate(x * function(x), x, s.oo, s.true, s.true)
        is None
    )
    for argument in (-x, x + s.I):
        assert (
            fresnel_auxiliary_tail_certificate(
                function(argument), x, s.oo, s.true, s.true
            )
            is None
        )


def test_appell_continuity_and_cut():
    x = s.Symbol("x", real=True)
    function = AppellF1(
        s.Rational(1, 4), s.S.Half, s.S.Half, s.Rational(5, 4), -x * x, -(x**4)
    )
    assert function.func is s.appellf1
    assert limit(s.sqrt(2 + x * x) * function, x, 0) == s.sqrt(2)
    for u in (2 + x, 1 + x):
        cut = s.appellf1(s.S.Half, s.S.Half, s.S.Half, s.Rational(3, 2), u, -x * x)
        assert appell_integral_continuity_certificate(cut, x, 0, s.true, s.true) is None


@pytest.mark.parametrize("base_power", [1, 2, 3])
def test_small_exponent_is_not_amplitude(base_power):
    x = s.Symbol("x", positive=True)
    assert limit((x ** (-base_power)) ** s.exp(-x * x), x, s.oo) == 1
    local = x ** s.exp(-1 / x**2)
    assert flat_exponential_certificate(local, x, 0, s.true, s.true) is None
    assert (
        flat_exponential_certificate(local * s.exp(-1 / x**2), x, 0, s.true, s.true)
        is None
    )


def test_negative_small_exponent():
    x = s.Symbol("x", positive=True)
    assert limit(x ** (-s.exp(-x * x) / 2), x, s.oo) == 1
