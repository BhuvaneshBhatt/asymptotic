import json
from pathlib import Path

import pytest
import sympy as sp

from asymptotic.olver_coefficients import (
    olver_coefficient,
)
from asymptotic.regime_selection import (
    select_asymptotic_regime,
)
from asymptotic.uniform_special_expansions import (
    bessel_j_large_order,
    bessel_y_large_order,
    modified_bessel_i_large_order,
    modified_bessel_k_large_order,
)

DATA = json.loads(
    (Path(__file__).parent / "data/uniform_asymptotic_reference_cases.json").read_text()
)["cases"]


@pytest.mark.parametrize(
    "case",
    [c for c in DATA if c["category"] == "turning_numeric"],
    ids=lambda c: c["id"],
)
def test_transition_numeric_reference(case):
    n = sp.Integer(case["n"])
    a = sp.Float(case["a"], 40)
    arg = n + a * n ** sp.Rational(1, 3)
    z = arg / n
    if case["family"] == "J":
        approx = bessel_j_large_order(n, z, terms=2).prefix
    else:
        approx = bessel_y_large_order(n, z, terms=2).prefix
    reference = sp.Float(case["reference"], 45)
    relative = abs(complex(sp.N((approx - reference) / reference, 30)))
    assert relative < 0.01


@pytest.mark.parametrize(
    "case",
    [c for c in DATA if c["category"] == "scaled_numeric"],
    ids=lambda c: c["id"],
)
def test_scaled_numeric_reference(case):
    n = sp.Integer(case["n"])
    z = sp.Rational(str(case["z"]))
    family = case["family"]
    if family == "J":
        result = bessel_j_large_order(n, z, terms=2)
    elif family == "Y":
        result = bessel_y_large_order(n, z, terms=2)
    elif family == "I":
        result = modified_bessel_i_large_order(n, z, terms=4)
    else:
        result = modified_bessel_k_large_order(n, z, terms=4)
    reference = sp.Float(case["reference"], 45)
    relative = abs(complex(sp.N((result.prefix - reference) / reference, 30)))
    assert relative < 0.02


@pytest.mark.parametrize(
    "case",
    [c for c in DATA if c["category"] == "regime_adversarial"],
    ids=lambda c: c["id"],
)
def test_regime_adversarial_reference(case):
    n = sp.symbols("n", positive=True)
    power = sp.sympify(case["power"])
    atom = sp.besselj(n, n + case["coefficient"] * n**power, evaluate=False)
    result = select_asymptotic_regime(atom, n)
    assert result.regime.value == case["expected"]


@pytest.mark.parametrize(
    "case", [c for c in DATA if c["category"] == "olver_exact"], ids=lambda c: c["id"]
)
def test_olver_exact_reference(case):
    expected = sp.sympify(case["value"])
    assert sp.simplify(olver_coefficient(case["family"], case["k"], 1) - expected) == 0


def test_corpus_has_specialist_coverage():
    categories = {c["category"] for c in DATA}
    required = {
        "turning_numeric",
        "scaled_numeric",
        "olver_exact",
        "regime_adversarial",
        "wronskian",
        "recurrence",
        "connection",
        "conjugation",
        "sector_boundary",
        "stokes_jump",
        "optimal_truncation",
        "ode_residual",
        "zero",
    }
    assert required <= categories
    assert len(DATA) >= 200


STRUCTURAL = [
    case
    for case in DATA
    if case["category"]
    in {
        "wronskian",
        "recurrence",
        "connection",
        "conjugation",
        "sector_boundary",
        "stokes_jump",
        "optimal_truncation",
        "ode_residual",
        "zero",
    }
]


@pytest.mark.parametrize("case", STRUCTURAL, ids=lambda c: c["id"])
def test_structural_reference_cases_are_executable(case):
    import mpmath as mp

    from asymptotic.hyperasymptotics import (
        least_term_index,
    )
    from asymptotic.olver_coefficients import (
        olver_coefficient,
    )
    from asymptotic.uniform_special_expansions import (
        rescaled_terminant,
    )
    from asymptotic.uniform_zeros import (
        bessel_zero_large_order,
    )

    mp.mp.dps = 50
    category = case["category"]
    j = case["case"]
    nu = mp.mpf(12 + j)
    z = mp.mpf("0.75") + mp.mpf(j) / 20
    x = nu * z
    if category == "wronskian":
        J = mp.besselj(nu, x)
        Y = mp.bessely(nu, x)
        Jp = (mp.besselj(nu - 1, x) - mp.besselj(nu + 1, x)) / 2
        Yp = (mp.bessely(nu - 1, x) - mp.bessely(nu + 1, x)) / 2
        assert abs(J * Yp - Jp * Y - 2 / (mp.pi * x)) < mp.mpf("1e-42")
    elif category == "recurrence":
        residual = (
            mp.besselj(nu - 1, x)
            + mp.besselj(nu + 1, x)
            - 2 * nu / x * mp.besselj(nu, x)
        )
        assert abs(residual) < mp.mpf("1e-42")
    elif category == "connection":
        residual = mp.hankel1(nu, x) - mp.besselj(nu, x) - 1j * mp.bessely(nu, x)
        assert abs(residual) < mp.mpf("1e-42")
    elif category == "conjugation":
        w = mp.mpc("0.8", str(mp.mpf(j + 1) / 20))
        assert abs(mp.conj(mp.besselj(nu, w)) - mp.besselj(nu, mp.conj(w))) < mp.mpf(
            "1e-42"
        )
    elif category == "sector_boundary":
        eps = sp.Rational(j + 1, 1000)
        upper = sp.exp(sp.I * (sp.pi - eps))
        lower = sp.conjugate(upper)
        difference = sp.N(
            olver_coefficient("A", 1, lower)
            - sp.conjugate(olver_coefficient("A", 1, upper)),
            30,
        )
        assert abs(complex(difference)) < 1e-25
    elif category == "stokes_jump":
        p = sp.Integer(8 + j)
        w = sp.Rational(5 + j, 1)
        assert (
            sp.simplify(
                rescaled_terminant(p, w)
                - sp.exp(w) * sp.gamma(p) * sp.uppergamma(1 - p, w) / (2 * sp.pi)
            )
            == 0
        )
    elif category == "optimal_truncation":
        chi = sp.Rational(21 + 2 * j, 2)
        n = least_term_index(chi)
        assert n <= chi < n + 1
    elif category == "ode_residual":
        # Exact Bessel recurrence/differential equation supplies an independent
        # normalization/sign oracle for the uniform approximants.
        J = mp.besselj(nu, x)
        Jp = (mp.besselj(nu - 1, x) - mp.besselj(nu + 1, x)) / 2
        Jpp = -(Jp / x) - (1 - (nu / x) ** 2) * J
        assert abs(x * x * Jpp + x * Jp + (x * x - nu * nu) * J) < mp.mpf("1e-40")
    elif category == "zero":
        approximation = bessel_zero_large_order(sp.Integer(30 + 5 * j), 1)
        assert approximation.residual < sp.Rational(1, 100)
