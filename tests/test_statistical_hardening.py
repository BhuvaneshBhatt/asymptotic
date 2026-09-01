from __future__ import annotations

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy.stats import Binomial, Exponential, Normal, P, quantile

from asymptotic import (
    StatisticalAsymptoticResult,
    asymptotic_minimize,
    asymptotic_probability,
    asymptotic_sum,
    laplace_asymptotic_integral,
)
from asymptotic.certification import replay_certification
from asymptotic.statistical_transforms import (
    _support_subset_condition,
    asymptotic_cdf,
    asymptotic_cross_entropy,
    asymptotic_cumulative_hazard,
    asymptotic_kl_divergence,
    asymptotic_local_limit,
    asymptotic_quantile,
    asymptotic_survival,
    asymptotic_variance,
)


@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=-2, max_value=9))
def test_discrete_probability_boundary_identities(threshold: int):
    n = sp.symbols("n", positive=True)
    x = Binomial("X_boundary", 7, sp.Rational(2, 5))

    cdf = asymptotic_probability(x <= threshold, x, parameter=n)
    survival = asymptotic_probability(x > threshold, x, parameter=n)
    left = asymptotic_probability(x < threshold, x, parameter=n)
    right = asymptotic_probability(x >= threshold, x, parameter=n)

    assert sp.simplify(cdf.expression + survival.expression - 1) == 0
    assert sp.simplify(left.expression + right.expression - 1) == 0

    mass = asymptotic_probability(sp.Eq(x, threshold), x, parameter=n)
    previous = asymptotic_probability(x <= threshold - 1, x, parameter=n)
    assert sp.simplify(cdf.expression - previous.expression - mass.expression) == 0


@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=-2, max_value=8))
def test_discrete_noninteger_threshold_has_no_boundary_mass(integer_part: int):
    n = sp.symbols("n", positive=True)
    x = Binomial("X_half_boundary", 6, sp.Rational(1, 3))
    threshold = sp.Rational(2 * integer_part + 1, 2)
    cdf = asymptotic_cdf(x, threshold, parameter=n)
    survival = asymptotic_survival(x, threshold, parameter=n)
    assert sp.simplify(cdf.expression + survival.expression - 1) == 0


def test_discrete_cumulative_hazard_uses_documented_survival_boundary():
    n = sp.symbols("n", positive=True)
    x = Binomial("X_cum_hazard", 1, sp.Rational(1, 3))
    survival = asymptotic_survival(x, 0, parameter=n)
    cumulative = asymptotic_cumulative_hazard(x, 0, parameter=n)
    assert survival.expression == sp.Rational(1, 3)
    assert sp.simplify(cumulative.expression + sp.log(survival.expression)) == 0


def test_discrete_quantile_is_generalized_inverse_not_cdf_equality_root():
    n = sp.symbols("n", positive=True)
    x = Binomial("X_quantile_inverse", 4, sp.Rational(1, 2))
    p = sp.Rational(3, 10)
    result = asymptotic_quantile(x, p, parameter=n)
    expected = quantile(x)(p)
    assert result.status == "EXACT"
    assert result.expression == expected
    assert P(x <= result.expression).doit() >= p
    if result.expression > 0:
        assert P(x <= result.expression - 1).doit() < p


def test_symbolic_discrete_quantile_does_not_use_equality_inversion():
    n = sp.symbols("n", positive=True, integer=True)
    p = sp.symbols("p", positive=True)
    x = Binomial("X_symbolic_quantile", n, sp.Rational(1, 2))
    result = asymptotic_quantile(x, p, parameter=n)
    assert result.method == "generalized-inverse-quantile"
    assert result.status in {"EXACT", "UNKNOWN"}


def test_support_containment_rejects_missing_target_support():
    n = sp.symbols("n", positive=True)
    reference = Normal("X_support_ref", 0, 1)
    target = Exponential("X_support_target", 1)
    cross = asymptotic_cross_entropy(reference, target, parameter=n)
    divergence = asymptotic_kl_divergence(reference, target, parameter=n)
    assert cross.expression is sp.oo
    assert divergence.expression is sp.oo
    assert cross.status == divergence.status == "EXACT"


def test_unknown_support_containment_is_an_exact_set_obligation():
    a, b = sp.symbols("a b", positive=True)
    source = sp.Interval(0, a)
    target = sp.Interval(0, b)
    decision, condition = _support_subset_condition(source, target)
    assert decision is None
    assert condition == sp.Eq(source - target, sp.S.EmptySet, evaluate=False)
    assert "PowerSet" not in str(condition)


def test_variance_uses_centered_guarded_expectation():
    n = sp.symbols("n", positive=True)
    x = Normal("X_cancel", n**3 + 1 / n, n**-4)
    result = asymptotic_variance(x, x, parameter=n, terms=3)
    assert result.method == "variance-centered-expectation"
    assert sp.simplify(result.expression - n**-8) == 0


def test_certified_public_result_families_have_replayable_evidence():
    n = sp.symbols("n", positive=True, integer=True)
    x = sp.symbols("x", real=True)
    k = sp.symbols("k", positive=True, integer=True)

    laplace = laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )
    summation = asymptotic_sum(
        1 / k**2, k, n, sp.oo, parameter=n, terms=3, method="euler-maclaurin"
    )
    optimum = asymptotic_minimize((x - n) ** 2 + 1 / n, x, parameter=n)
    binomial = Binomial("X_cert_local", n, sp.Rational(1, 2))
    local = asymptotic_local_limit(binomial, n / 2, parameter=n, terms=2)

    for result in (laplace, summation, optimum, local):
        assert result.status == "CERTIFIED"
        assert replay_certification(result) is True


def test_unbacked_certified_status_fails_replay_contract():
    n = sp.symbols("n", positive=True)
    fake = StatisticalAsymptoticResult(1 / n, n, sp.oo, "fake", "CERTIFIED")
    assert replay_certification(fake) is False
