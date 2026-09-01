import pytest
import sympy as sp

from asymptotic.instrumentation import symbolic_metrics
from asymptotic.stirling import (
    certified_logfactorial,
    certified_loggamma,
    normalize_positive_pmf,
)


def test_positive_loggamma_stieltjes_certificate():
    n = sp.symbols("n", positive=True)
    result = certified_loggamma(n, parameter=n, terms=3)
    assert result.certified
    assert result.remainder.scale == sp.Abs(sp.bernoulli(6)) / (30 * n**5)
    assert not result.expression.has(sp.loggamma, sp.gamma)


def test_factorial_normalization_is_certified():
    n = sp.symbols("n", positive=True)
    result = certified_logfactorial(n, parameter=n, terms=2)
    assert result.certified
    assert result.kind == "logfactorial"


def test_poisson_pmf_normalizes_to_exponential_scale():
    n = sp.symbols("n", positive=True)
    k = sp.symbols("k", positive=True, integer=True)
    pmf = sp.exp(-n) * n**k / sp.factorial(k)
    with symbolic_metrics() as metrics:
        result = normalize_positive_pmf(pmf, variable=k, parameter=n, terms=2)
    assert result.certified
    assert not result.expression.has(sp.factorial, sp.gamma, sp.binomial)
    assert metrics.pmf_normalizations == 1
    assert metrics.loggamma_normalizations >= 1


def test_binomial_pmf_uses_explicit_probability_assumption():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", nonnegative=True, integer=True)
    p = sp.symbols("p", positive=True)
    pmf = sp.binomial(n, k) * p**k * (1 - p) ** (n - k)
    result = normalize_positive_pmf(
        pmf,
        variable=k,
        parameter=n,
        terms=2,
        assumptions=sp.Q.positive(1 - p) & sp.Q.nonnegative(n - k),
    )
    assert result.certified
    assert not result.expression.has(sp.factorial, sp.gamma, sp.binomial)


def test_branch_unsafe_probability_factor_is_rejected():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", nonnegative=True, integer=True)
    p = sp.symbols("p", positive=True)
    pmf = sp.binomial(n, k) * p**k * (1 - p) ** (n - k)
    with pytest.raises(ValueError, match="positive"):
        normalize_positive_pmf(pmf, variable=k, parameter=n, terms=2)
