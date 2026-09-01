from __future__ import annotations

import math

import sympy as sp
from sympy.stats import Binomial

from asymptotic.statistical_transforms import asymptotic_local_limit


def test_binomial_clt_scale_local_limit_has_gaussian_leading_term():
    n = sp.symbols("n", positive=True, integer=True)
    s = sp.symbols("s", real=True)
    x = Binomial("X_clt_local", n, sp.Rational(1, 3))

    result = asymptotic_local_limit(x, n / 3 + s * sp.sqrt(n), parameter=n, terms=2)

    expected = 3 * sp.exp(-sp.Rational(9, 4) * s**2) / (2 * sp.sqrt(sp.pi))
    assert sp.simplify(sp.limit(sp.sqrt(n) * result.expression, n, sp.oo) - expected) == 0
    assert result.status == "CERTIFIED"
    assert result.remainder is not None


def test_binomial_rounded_clt_scale_retains_lattice_correction():
    n = sp.symbols("n", positive=True, integer=True)
    s = sp.symbols("s", real=True)
    x = Binomial("X_rounded_local", n, sp.Rational(1, 3))
    location = sp.floor(n / 3 + s * sp.sqrt(n))

    result = asymptotic_local_limit(x, location, parameter=n, terms=1)

    assert result.expression.has(sp.floor)
    assert result.status == "CERTIFIED"
    # A concrete lattice point checks that the retained rounding term improves
    # the local mass without pretending the point is continuously valued.
    n_value = 300
    s_value = sp.Rational(3, 2)
    k_value = math.floor(n_value / 3 + float(s_value) * math.sqrt(n_value))
    approx = float(result.expression.subs({n: n_value, s: s_value}).evalf())
    exact = float(
        sp.binomial(n_value, k_value)
        * sp.Rational(1, 3) ** k_value
        * sp.Rational(2, 3) ** (n_value - k_value)
    )
    assert abs(approx / exact - 1) < 0.02


def test_binomial_moderate_clt_displacement_regression():
    n = sp.symbols("n", positive=True, integer=True)
    x = Binomial("X_moderate_local", n, sp.Rational(1, 3))
    # s=3 is well outside the central peak but remains on the O(sqrt(n))
    # local-limit scale.  This catches accidental bounded-offset routing.
    location = n / 3 + 3 * sp.sqrt(n)
    result = asymptotic_local_limit(x, location, parameter=n, terms=2)

    expected_lead = 3 * sp.exp(-sp.Rational(81, 4)) / (2 * sp.sqrt(sp.pi))
    assert sp.simplify(sp.limit(sp.sqrt(n) * result.expression, n, sp.oo) - expected_lead) == 0
    assert result.status == "CERTIFIED"
