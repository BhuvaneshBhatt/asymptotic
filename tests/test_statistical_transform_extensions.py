from __future__ import annotations

import sympy as sp
from sympy.stats import Binomial, ContinuousRV

from asymptotic.statistical_transforms import (
    LogProbabilityResult,
    ModeResult,
    cross_entropy,
    cumulative_hazard,
    entropy,
    factorial_moment,
    hazard,
    kl_divergence,
    log_probability,
    map,
    mode,
    pgf,
)


def test_binomial_mode_map_factorial_moment_and_pgf():
    n = sp.symbols("n", positive=True, integer=True)
    z = sp.symbols("z")
    x = Binomial("X_transform_ext", n, sp.Rational(1, 3))

    mode_result = mode(x, parameter=n)
    mapped = map(x, parameter=n)
    assert isinstance(mode_result, ModeResult)
    assert mapped.expression == mode_result.expression
    assert mode_result.lattice_candidates

    factorial = factorial_moment(x, order=3, parameter=n)
    assert sp.simplify(factorial.expression - n * (n - 1) * (n - 2) / 27) == 0

    pgf_result = pgf(x, transform_variable=z, parameter=n)
    assert pgf_result.expression == ((z + 2) / 3) ** n


def test_binomial_information_transforms_are_consistent():
    n = sp.symbols("n", positive=True, integer=True)
    p = Binomial("X_info_p", n, sp.Rational(1, 3))
    q = Binomial("X_info_q", n, sp.Rational(1, 2))

    entropy_result = entropy(p, parameter=n, terms=2)
    cross = cross_entropy(p, q, parameter=n, terms=2)
    divergence = kl_divergence(p, q, parameter=n)
    assert (
        sp.simplify(
            cross.expression - entropy_result.expression - divergence.expression
        )
        == 0
    )
    assert divergence.status == "EXACT"


def test_log_probability_and_hazard_result_contracts():
    n = sp.symbols("n", positive=True, integer=True)
    x = Binomial("X_hazard_ext", 1, sp.Rational(1, 3))

    logp = log_probability(x >= 0, x, parameter=n)
    assert isinstance(logp, LogProbabilityResult)
    assert logp.expression == 0

    cumulative = cumulative_hazard(x, 0, parameter=n)
    hazard_result = hazard(x, 0, parameter=n)
    assert cumulative.expression == sp.log(3)
    assert sp.simplify(hazard_result.expression - sp.Rational(2, 3)) == 0


def test_continuous_mode_preserves_equal_height_dominant_saddles():
    n = sp.symbols("n", positive=True)
    z = sp.symbols("z", real=True)
    x = ContinuousRV(z, sp.exp(-((z**2 - 1) ** 2)), set=sp.S.Reals)

    mode_result = mode(x, parameter=n)

    assert mode_result.lattice_candidates == (-1, 1)
    assert mode_result.expression == -1
