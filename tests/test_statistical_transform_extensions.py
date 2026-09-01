from __future__ import annotations

import sympy as sp
from sympy.stats import Binomial

from asymptotic.statistical_transforms import (
    AsymptoticModeResult,
    LogProbabilityResult,
    asymptotic_cross_entropy,
    asymptotic_cumulative_hazard,
    asymptotic_entropy,
    asymptotic_factorial_moment,
    asymptotic_hazard,
    asymptotic_kl_divergence,
    asymptotic_log_probability,
    asymptotic_map,
    asymptotic_mode,
    asymptotic_pgf,
)


def test_binomial_mode_map_factorial_moment_and_pgf():
    n = sp.symbols("n", positive=True, integer=True)
    z = sp.symbols("z")
    x = Binomial("X_transform_ext", n, sp.Rational(1, 3))

    mode = asymptotic_mode(x, parameter=n)
    mapped = asymptotic_map(x, parameter=n)
    assert isinstance(mode, AsymptoticModeResult)
    assert mapped.expression == mode.expression
    assert mode.lattice_candidates

    factorial = asymptotic_factorial_moment(x, order=3, parameter=n)
    assert sp.simplify(factorial.expression - n * (n - 1) * (n - 2) / 27) == 0

    pgf = asymptotic_pgf(x, transform_variable=z, parameter=n)
    assert pgf.expression == ((z + 2) / 3) ** n


def test_binomial_information_transforms_are_consistent():
    n = sp.symbols("n", positive=True, integer=True)
    p = Binomial("X_info_p", n, sp.Rational(1, 3))
    q = Binomial("X_info_q", n, sp.Rational(1, 2))

    entropy = asymptotic_entropy(p, parameter=n, terms=2)
    cross = asymptotic_cross_entropy(p, q, parameter=n, terms=2)
    divergence = asymptotic_kl_divergence(p, q, parameter=n)
    assert sp.simplify(cross.expression - entropy.expression - divergence.expression) == 0
    assert divergence.status == "EXACT"


def test_log_probability_and_hazard_result_contracts():
    n = sp.symbols("n", positive=True, integer=True)
    x = Binomial("X_hazard_ext", 1, sp.Rational(1, 3))

    logp = asymptotic_log_probability(x >= 0, x, parameter=n)
    assert isinstance(logp, LogProbabilityResult)
    assert logp.expression == 0

    cumulative = asymptotic_cumulative_hazard(x, 0, parameter=n)
    hazard = asymptotic_hazard(x, 0, parameter=n)
    assert cumulative.expression == sp.log(3)
    assert sp.simplify(hazard.expression - sp.Rational(2, 3)) == 0
