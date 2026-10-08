"""Analytic cancellation orders and branch-sensitive real germs."""

import json
from pathlib import Path

import pytest
import sympy as sp

from asymptotic import limit
from asymptotic.analytic_cancellation_germs import (
    acos_endpoint_ratio_certificate,
    analytic_jet,
    argument_step_germ_certificate,
    gamma_analytic_cancellation_certificate,
    logarithmic_power_sum_certificate,
)
from asymptotic.reference_normalization import scalar_reference_namespace


@pytest.mark.parametrize("case_id", ["0146", "0228", "0249", "0259", "0463"])
def test_timeout_values(case_id):
    rows = json.loads(
        (
            Path(__file__).parents[2] / "data/univariate_limit_reference_cases.json"
        ).read_text()
    )
    row = next(r for r in rows if r["id"] == "univariate_reference_" + case_id)
    namespace = scalar_reference_namespace()
    from asymptotic.analytic_limits import analytic_limit

    result = analytic_limit(
        sp.sympify(row["expression"], locals=namespace),
        sp.Symbol(row["variable"]),
        sp.sympify(row["point"]),
        direction=row.get("direction"),
        return_result=True,
    )
    expected = sp.sympify(row["expected"], locals=namespace)
    assert result.status.value == "proved"
    assert (
        sp.simplify(result.value - expected) == 0
        if expected.is_finite
        else result.value == expected
    )


def test_gamma_third_order():
    x = sp.Symbol("x", real=True)
    second = sp.EulerGamma**2 / 2 + sp.pi**2 / 12
    e = (sp.gamma(1 + x) - 1 + sp.EulerGamma * x - second * x**2) / x**3
    value = -(sp.EulerGamma**3) / 6 - sp.EulerGamma * sp.pi**2 / 12 - sp.zeta(3) / 3
    assert sp.expand(limit(e, x, 0) - value) == 0


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: sp.gamma(x) / x,
        lambda x: sp.Function("f")(x),
        lambda x: sp.log(-1 + x),
    ],
)
def test_unverified_jets(expression):
    x = sp.Symbol("x", real=True)
    assert analytic_jet(expression(x), x) is None


def test_gamma_parameter_declines():
    x, a = sp.symbols("x a")
    assert (
        gamma_analytic_cancellation_certificate((sp.gamma(1 + a * x) - 1) / x, x, 0)
        is None
    )


def test_gamma_remainder_order():
    x = sp.Symbol("x", real=True)
    prefix = sp.series(sp.gamma(1 + x), x, 0, 9).removeO()
    assert (
        gamma_analytic_cancellation_certificate((sp.gamma(1 + x) - prefix) / x**9, x, 0)
        is None
    )


def test_acos_unverified_side():
    x = sp.Symbol("x", real=True)
    assert (
        acos_endpoint_ratio_certificate(sp.acos(1 - x) / sp.acos(1 - x * x), x, 0)
        is None
    )


@pytest.mark.parametrize("sign, expected", [(1, 1), (-1, 0)])
def test_argument_step(sign, expected):
    x = sp.Symbol("x", real=True)
    e = sp.Heaviside(sp.arg(1 + sign * sp.I * x * x), 1)
    assert limit(e, x, 0) == expected


def test_argument_step_sides():
    x = sp.Symbol("x", real=True)
    e = sp.Heaviside(sp.arg(1 + sp.I * x), 1)
    assert argument_step_germ_certificate(e, x, sp.S.Zero, sp.S.true, sp.S.true) is None
    assert argument_step_germ_certificate(e, x, sp.S.Zero, x > 0, sp.S.true)[1] == 1


def test_power_log_scale():
    x = sp.Symbol("x", real=True)
    e = sp.log(
        (x ** sp.Rational(1, 5) + 2 * x ** sp.Rational(2, 3)) / (1 - x)
    ) / sp.log(x)
    assert logarithmic_power_sum_certificate(e, x, 0)[1] == sp.Rational(1, 5)


def test_power_log_cancellation():
    x = sp.Symbol("x", real=True)
    e = sp.log(
        (x ** sp.Rational(1, 5) - x ** sp.Rational(1, 5) + 2 * x ** sp.Rational(2, 3))
        / (1 - x)
    ) / sp.log(x)
    assert logarithmic_power_sum_certificate(e, x, 0)[1] == sp.Rational(2, 3)


def test_inverse_sech_modulus():
    x = sp.Symbol("x", real=True)
    for rate in (-2, 3):
        e = sp.exp(-sp.sqrt(1 - sp.asech(rate * x) ** 2)) / x**2
        assert limit(e, x, 0) is sp.zoo


def test_elliptic_amplitude():
    x = sp.Symbol("x", positive=True)
    assert limit(sp.elliptic_f(sp.I / x, sp.Rational(1, 2)) + 2, x, sp.oo) == 2


def test_elliptic_multiplier_declines():
    from asymptotic.analytic_cancellation_germs import (
        vanishing_elliptic_amplitude_certificate,
    )

    x = sp.Symbol("x", positive=True)
    assert (
        vanishing_elliptic_amplitude_certificate(
            x * sp.elliptic_f(1 / x, sp.Rational(1, 2)), x, sp.oo
        )
        is None
    )


def test_complex_binomial_shift():
    n, a = sp.symbols("n a")
    e = sp.sqrt(2 * n) ** (-n * sp.binomial(n, n / 2 + sp.sqrt(n / 2) * a))
    result = limit(e, n, sp.oo, return_result=True)
    assert result.status.value == "unknown"
    assert result.evidence[0].method == "binomial_power_parameter_contract"


def test_complete_elliptic_declines():
    from asymptotic.analytic_cancellation_germs import (
        vanishing_elliptic_amplitude_certificate,
    )

    x = sp.Symbol("x", positive=True)
    assert (
        vanishing_elliptic_amplitude_certificate(
            sp.elliptic_pi(1 / x, sp.Rational(1, 2)), x, sp.oo
        )
        is None
    )
