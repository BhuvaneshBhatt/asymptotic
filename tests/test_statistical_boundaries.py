from __future__ import annotations

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy.stats import Binomial, Exponential, Normal, P
from sympy.stats import quantile as exact_quantile

from asymptotic import (
    StatisticalResult,
    minimize,
    probability,
    sum,
)
from asymptotic.certification import replay_certification
from asymptotic.probability import (
    laplace_asymptotic_integral,
)
from asymptotic.statistical_transforms import (
    _support_subset_condition,
    cdf,
    cross_entropy,
    cumulative_hazard,
    kl_divergence,
    local_limit,
    quantile,
    survival,
    variance,
)


@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=-2, max_value=9))
def test_discrete_probability_boundary_identities(threshold: int):
    n = sp.symbols("n", positive=True)
    x = Binomial("X_boundary", 7, sp.Rational(2, 5))

    cdf_result = probability(x <= threshold, x, parameter=n, return_result=True)
    survival_result = probability(x > threshold, x, parameter=n, return_result=True)
    left = probability(x < threshold, x, parameter=n, return_result=True)
    right = probability(x >= threshold, x, parameter=n, return_result=True)

    assert sp.simplify(cdf_result.expression + survival_result.expression - 1) == 0
    assert sp.simplify(left.expression + right.expression - 1) == 0

    mass = probability(sp.Eq(x, threshold), x, parameter=n, return_result=True)
    previous = probability(x <= threshold - 1, x, parameter=n, return_result=True)
    assert (
        sp.simplify(cdf_result.expression - previous.expression - mass.expression) == 0
    )


@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=-2, max_value=8))
def test_discrete_noninteger_threshold_has_no_boundary_mass(integer_part: int):
    n = sp.symbols("n", positive=True)
    x = Binomial("X_half_boundary", 6, sp.Rational(1, 3))
    threshold = sp.Rational(2 * integer_part + 1, 2)
    cdf_result = cdf(x, threshold, parameter=n)
    survival_result = survival(x, threshold, parameter=n)
    assert sp.simplify(cdf_result.expression + survival_result.expression - 1) == 0


def test_discrete_cumulative_hazard_uses_documented_survival_boundary():
    n = sp.symbols("n", positive=True)
    x = Binomial("X_cum_hazard", 1, sp.Rational(1, 3))
    survival_result = survival(x, 0, parameter=n)
    cumulative = cumulative_hazard(x, 0, parameter=n)
    assert survival_result.expression == sp.Rational(1, 3)
    assert sp.simplify(cumulative.expression + sp.log(survival_result.expression)) == 0


def test_discrete_quantile_is_generalized_equality_root():
    n = sp.symbols("n", positive=True)
    x = Binomial("X_quantile_inverse", 4, sp.Rational(1, 2))
    p = sp.Rational(3, 10)
    result = quantile(x, p, parameter=n)
    expected = exact_quantile(x)(p)
    assert result.status == "EXACT"
    assert result.expression == expected
    assert P(x <= result.expression).doit() >= p
    if result.expression > 0:
        assert P(x <= result.expression - 1).doit() < p


def test_symbolic_discrete_quantile_does_not_use_equality_inversion():
    n = sp.symbols("n", positive=True, integer=True)
    p = sp.symbols("p", positive=True)
    x = Binomial("X_symbolic_quantile", n, sp.Rational(1, 2))
    result = quantile(x, p, parameter=n)
    assert result.method == "generalized-inverse-quantile"
    assert result.status in {"EXACT", "UNKNOWN"}


def test_support_containment_rejects_missing_target_support():
    n = sp.symbols("n", positive=True)
    reference = Normal("X_support_ref", 0, 1)
    target = Exponential("X_support_target", 1)
    cross = cross_entropy(reference, target, parameter=n)
    divergence = kl_divergence(reference, target, parameter=n)
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
    result = variance(x, x, parameter=n, terms=3)
    assert result.method == "variance-centered-expectation"
    assert sp.simplify(result.expression - n**-8) == 0


def test_certified_public_result_families_have_replayable_evidence():
    n = sp.symbols("n", positive=True, integer=True)
    x = sp.symbols("x", real=True)
    k = sp.symbols("k", positive=True, integer=True)

    laplace = laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )
    summation = sum(
        1 / k**2,
        k,
        n,
        sp.oo,
        parameter=n,
        terms=3,
        method="euler-maclaurin",
        return_result=True,
    )
    optimum = minimize((x - n) ** 2 + 1 / n, x, parameter=n, return_result=True)
    binomial = Binomial("X_cert_local", n, sp.Rational(1, 2))
    local = local_limit(binomial, n / 2, parameter=n, terms=2)

    for result in (laplace, summation, optimum, local):
        assert result.status == "CERTIFIED"
        assert replay_certification(result) is True


def test_unbacked_certified_status_fails_replay_contract():
    n = sp.symbols("n", positive=True)
    fake = StatisticalResult(1 / n, n, sp.oo, "fake", "CERTIFIED")
    assert replay_certification(fake) is False
