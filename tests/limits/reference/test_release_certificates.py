"""Tail certificates retain their mathematical prerequisites and domain seams."""

import json
from pathlib import Path

import pytest
import sympy as s

from asymptotic.bessel_primitive_germs import bessel_primitive_certificate
from asymptotic.dilogarithm_tail_germs import paired_dilogarithm_tail_certificate
from asymptotic.fixed_ray_branch_germs import DirectionalInfinity
from asymptotic.limits import limit
from asymptotic.logarithmic_integral_germs import logarithmic_integral_power_certificate
from asymptotic.parameter_tail_germs import binomial_ratio_certificate
from asymptotic.reference_normalization import scalar_reference_equal


@pytest.mark.parametrize(
    "order", [s.Rational(1, 3), s.Rational(2, 3), s.Rational(4, 3)]
)
def test_bessel_primitive_pair(order):
    x = s.Symbol("x", positive=True)

    def primitive(nu):
        return (
            x ** (nu + 1)
            * s.hyper(((nu + 1) / 2,), ((nu + 3) / 2, nu + 1), x * x / 4)
            / (2**nu * (nu + 1) * s.gamma(nu + 1))
        )

    expr = primitive(-order) - primitive(order)
    result = bessel_primitive_certificate(expr, x, s.oo, s.S.true, s.S.true)
    assert s.simplify(result[1] - 2 * s.sin(s.pi * order / 2)) == 0
    assert (
        bessel_primitive_certificate(
            primitive(order) + primitive(-order), x, s.oo, s.S.true, s.S.true
        )
        is None
    )
    # Independent numerical differentiation checks the exact primitive normalization.
    derivative = s.diff(expr, x)
    expected = 2 * s.sin(s.pi * order) * s.besselk(order, x) / s.pi
    assert abs(complex((derivative - expected).subs(x, 2).evalf(40))) < 1e-35


def test_logarithmic_integral_power_bounds():
    x = s.Symbol("x", positive=True)
    expr = s.exp(x) * s.li(x) ** (1 + 3 / s.log(x)) / s.li(s.exp(x)) ** (1 + 7 / x)
    result = logarithmic_integral_power_certificate(expr, x, s.oo, s.S.true, s.S.true)
    assert result[1] == s.oo
    assert (
        logarithmic_integral_power_certificate(
            expr * s.li(x) ** x, x, s.oo, s.S.true, s.S.true
        )
        is None
    )
    assert (
        logarithmic_integral_power_certificate(-expr, x, s.oo, s.S.true, s.S.true)
        is None
    )


def test_binomial_ratio_requires_pole_avoidance():
    x, p, w = s.symbols("x p w")
    expr = s.binomial(x + w, x + w - p) / (x * s.binomial(x + w, x + w - p + 1))
    finite = s.Q.finite(p) & s.Q.finite(w)
    assert binomial_ratio_certificate(expr, x, s.oo, s.S.true, finite) is None
    result = binomial_ratio_certificate(
        expr, x, s.oo, s.S.true, finite & s.Ne(1 / s.gamma(p), 0)
    )
    assert result[1] == 1 / p


def test_dilogarithm_cut_boundary():
    root = Path(__file__).resolve().parents[3]
    row = json.loads(
        (root / "tests/data/univariate_limit_reference_cases.json").read_text()
    )[264]
    x, p = s.symbols("x s")
    expr = s.sympify(row["expression"], locals={"x": x, "s": p})
    result = paired_dilogarithm_tail_certificate(expr, x, s.oo, s.S.true, s.re(p) > 0)
    assert (
        paired_dilogarithm_tail_certificate(expr, x, s.oo, s.S.true, s.S.true) is None
    )
    for value in [s.S.One, (1 + s.I) / 2, (1 - s.I) / 2]:
        actual = expr.subs({p: value, x: 10**9}).evalf(60)
        target = result[1].subs(p, value).evalf(60)
        assert abs(complex(actual - target)) < 3e-8


def test_binomial_corner_subsequences():
    x, y = s.symbols("x y", real=True)
    result = limit(s.binomial(x, y), (x, y), (-5, -8), return_result=True)
    assert result.status.name == "DOES_NOT_EXIST"
    assert [e.value for e in result.evidence] == [0, -35]


def test_passive_discontinuity_axis():
    x, y = s.symbols("x y", real=True)
    assert (
        limit(s.sign(y * y), (x, y), (0, 0), return_result=True).status.name
        == "DOES_NOT_EXIST"
    )
    assert limit(s.Tuple(x + y, x * y), (x, y), (0, 0)) == s.Tuple(0, 0)


def test_spherical_comparison_requires_cover():
    p = s.Symbol("p", real=True)
    covered = s.Piecewise((DirectionalInfinity(s.I), p > 0), (-s.oo, True))
    assert scalar_reference_equal(covered, s.zoo, "extended_complex")
    assert not scalar_reference_equal(covered, s.zoo)
    uncovered = s.Piecewise((DirectionalInfinity(s.I), p > 0))
    assert not scalar_reference_equal(uncovered, s.zoo, "extended_complex")


@pytest.mark.parametrize("frequency", [s.S.Zero, s.Rational(3, 2), -s.pi])
def test_small_exponential_log_derivative(frequency):
    x = s.Symbol("x", real=True)
    f = (x - 1) / (x**3 + 2) * s.exp(s.I * frequency * x)
    expr = s.diff(f, x) * s.exp(f) / (s.exp(f) - 1)
    result = limit(expr, x, s.oo, assumptions=x > 10, return_result=True)
    assert result.status.name == "PROVED"
    assert result.value == s.I * frequency
    assert result.evidence[0].method == "small_exponential_log_derivative"


def test_small_exponential_guard():
    from asymptotic.exponential_log_derivative_germs import (
        exponential_log_derivative_certificate,
    )

    x = s.Symbol("x", real=True)
    f = s.exp(s.I * x) / x
    expr = s.diff(f, x) * s.exp(f) / (s.exp(f) - 1)
    assert (
        exponential_log_derivative_certificate(2 * expr, x, s.oo, s.S.true, s.S.true)
        is None
    )
    growing = x * s.exp(s.I * x)
    assert (
        exponential_log_derivative_certificate(
            s.diff(growing, x) * s.exp(growing) / (s.exp(growing) - 1),
            x,
            s.oo,
            s.S.true,
            s.S.true,
        )
        is None
    )
    assert (
        exponential_log_derivative_certificate(expr, x, s.oo, s.S.true, x < 10) is None
    )
