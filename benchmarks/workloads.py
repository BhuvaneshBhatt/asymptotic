"""Reusable benchmark workloads with deterministic mathematical inputs."""

from __future__ import annotations

import sympy as sp
from sympy.stats import Normal

from asymptotic import (
    AsymptoticRemainder,
    RemainderKind,
    asymptotic_expectation,
    asymptotic_probability,
    asymptotic_sum,
    implicit_asymptotic,
    laplace_asymptotic_integral,
    multiseries,
    nested_expansion,
    series_reversion,
    transseries_from_expression,
)
from asymptotic.multivariate import multivariate_scaling_regimes
from asymptotic.nonlinear_ode import nonlinear_differential_transseries
from asymptotic.parameter_auto import automatic_parameter_stratification
from asymptotic.remainder_theorems import (
    certify_green_inverse_operator_remainder,
    certify_reciprocal_remainder,
)


def elementary_multiseries(terms: int = 8):
    """Expand a mixed elementary expression in the small scale ``1/x``."""

    x = sp.symbols("x", positive=True)
    expr = sp.exp(1 / x) * (1 + sp.log(1 + 1 / x))
    return multiseries(expr, x, scale=[1 / x], terms=terms)


def nested_logexp(depth: int = 3):
    """Construct a finite-height nested exponential/logarithmic expansion."""

    x = sp.symbols("x", positive=True)
    expr = sp.exp(1 / x)
    return nested_expansion(expr, x, depth=min(depth, 2))


def transseries_conversion(terms: int = 8):
    """Convert a finite mixed-scale expression to generalized transseries."""

    x = sp.symbols("x", positive=True)
    expr = 1 + 1 / x + sp.log(x) / x**2 + sp.exp(-x)
    expansion = transseries_from_expression(expr, x, point=sp.oo, complete=True)
    return expansion.truncation(terms)


def transseries_product_workload(size: int = 8):
    """Multiply two finite transseries with ``size`` nonconstant terms each."""

    if size < 1:
        raise ValueError("size must be positive")
    x = sp.symbols("x", positive=True)
    left_expr = sp.Add(*(sp.Rational(1, i + 1) / x**i for i in range(1, size + 1)))
    right_expr = sp.Add(*(sp.Rational(1, i + 2) / x ** (2 * i - 1) for i in range(1, size + 1)))
    left = transseries_from_expression(left_expr, x, point=sp.oo, complete=True)
    right = transseries_from_expression(right_expr, x, point=sp.oo, complete=True)
    return left * right


def reversion_workload(terms: int = 7):
    """Revert a nonlinear local power series."""

    x, y = sp.symbols("x y")
    return series_reversion(x + 2 * x**2 + x**3, x, y, terms=terms, branch=0)


def implicit_workload(terms: int = 5):
    """Resolve a singular implicit branch through Newton--Puiseux scaling."""

    x, y = sp.symbols("x y", positive=True)
    return implicit_asymptotic(y**2 - x, y, x, terms=min(terms, 4))


def parameter_strata_workload():
    """Split and canonicalize a small parameter-dependent degeneracy."""

    x, a, b = sp.symbols("x a b")
    expressions = (a * x + b * x**2, (a + b) * x)
    return automatic_parameter_stratification(
        expressions,
        lambda condition: condition,
        parameters=(a, b),
        max_splits=3,
    )


def multivariate_weight_cones():
    """Discover Newton weight regimes for a two-variable implicit balance."""

    x, z, y = sp.symbols("x z y", positive=True)
    return multivariate_scaling_regimes(y**2 - x - z**2, y, (x, z))


def nonlinear_ode_workload(terms: int = 3):
    """Lift a nonlinear differential balance through several corrections."""

    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    equation = x * sp.diff(y(x), x) - y(x) + y(x) ** 2
    return nonlinear_differential_transseries(equation, y, x, point=0, terms=terms)


def green_certificate_workload():
    """Certify an asymptotically constant second-order Green inverse."""

    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.diff(delta(x), x) / x - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    if not cert.certified or green is None:
        raise RuntimeError("benchmark Green problem unexpectedly failed certification")
    return green


def remainder_certificate_workload():
    """Exercise nonvanishing and relative-smallness remainder decisions."""

    x = sp.symbols("x", positive=True)
    remainder = AsymptoticRemainder.big_o(x**-2, x, sp.oo)
    cert = certify_reciprocal_remainder(1 + 1 / x, remainder)
    if cert.conclusion.kind is RemainderKind.UNKNOWN:
        raise RuntimeError("benchmark reciprocal problem unexpectedly remained UNKNOWN")
    return cert


def probability_tail_workload(terms: int = 3):
    """Expand a moving Gaussian large-deviation tail."""

    n, a = sp.symbols("n a", positive=True)
    x = Normal("X_benchmark_tail", 0, sp.sqrt(n))
    return asymptotic_probability(x > a * n, x, parameter=n, terms=terms)


def saddle_expectation_workload(terms: int = 3):
    """Expand a concentrating Gaussian expectation through its saddle."""

    n = sp.symbols("n", positive=True)
    x = Normal("X_benchmark_expectation", 0, 1 / sp.sqrt(n))
    return asymptotic_expectation(sp.exp(x), x, parameter=n, terms=terms, method="laplace")


def degenerate_saddle_workload(terms: int = 2):
    """Expand and certify a quartic degenerate Laplace saddle."""

    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)
    return laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=terms
    )


def discrete_saddle_workload(terms: int = 2):
    """Expand a scaled Gaussian lattice sum through the discrete saddle route."""

    n = sp.symbols("n", positive=True)
    k = sp.symbols("k", integer=True)
    return asymptotic_sum(
        sp.exp(-n * (k / n) ** 2 / 2), k, -sp.oo, sp.oo, parameter=n, terms=terms, method="saddle"
    )


MIXED_WORKLOADS = (
    elementary_multiseries,
    nested_logexp,
    transseries_conversion,
    reversion_workload,
    implicit_workload,
    parameter_strata_workload,
    multivariate_weight_cones,
    nonlinear_ode_workload,
    green_certificate_workload,
    remainder_certificate_workload,
    probability_tail_workload,
    saddle_expectation_workload,
    degenerate_saddle_workload,
    discrete_saddle_workload,
)
