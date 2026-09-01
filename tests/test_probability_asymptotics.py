import sympy as sp
from sympy.stats import Normal, Poisson, Uniform

from asymptotic import (
    StatisticalAsymptoticResult,
    asymptotic_expectation,
    asymptotic_probability,
    laplace_asymptotic_integral,
)


def test_exact_expectation_is_first_route_and_keeps_exact_expression():
    n = sp.symbols("n", positive=True)
    x = Normal("X_exact", 0, 1 / sp.sqrt(n))

    result = asymptotic_expectation(sp.exp(x), x, parameter=n, terms=3)

    assert isinstance(result, StatisticalAsymptoticResult)
    assert result.method == "exact-expectation"
    assert result.status == "EXACT"
    assert sp.simplify(result.expression - sp.exp(1 / (2 * n))) == 0
    assert sp.simplify(result.truncate() - (1 + 1 / (2 * n) + 1 / (8 * n**2))) == 0


def test_density_reduction_can_be_requested_directly():
    n = sp.symbols("n", positive=True)
    x = Uniform("X_density", 0, n)

    result = asymptotic_expectation(x**2, x, parameter=n, terms=2, method="density")

    assert result.method == "density-exact-integral"
    assert result.status == "EXACT"
    assert sp.simplify(result.expression - n**2 / 3) == 0
    assert result.reduction is not None
    assert result.reduction.has(sp.Integral)


def test_pmf_reduction_handles_exact_point_probability():
    lam = sp.symbols("lam", positive=True)
    k = Poisson("K_point", lam)

    result = asymptotic_probability(sp.Eq(k, 0), k, parameter=lam, method="pmf")

    assert result.method == "pmf-exact-sum"
    assert result.status == "EXACT"
    assert result.expression == sp.exp(-lam)
    assert result.reduction == sp.exp(-lam)


def test_moving_domain_normal_tail_uses_endpoint_laplace_scale():
    n, a = sp.symbols("n a", positive=True)
    x = Normal("X_tail", 0, sp.sqrt(n))

    result = asymptotic_probability(x > a * n, x, parameter=n, terms=3, method="laplace")

    expected = (
        sp.exp(-(a**2) * n / 2)
        / (a * sp.sqrt(2 * sp.pi * n))
        * (1 - 1 / (a**2 * n) + 3 / (a**4 * n**2))
    )
    assert result.method == "moving-domain/laplace-lower-endpoint"
    assert result.status == "CERTIFIED"
    assert result.certified
    assert result.transformation is not None
    assert result.domain == sp.Interval.open(a, sp.oo)
    assert sp.simplify(result.expression - expected) == 0


def test_interior_saddle_expectation_recovers_gaussian_mgf_expansion():
    n = sp.symbols("n", positive=True)
    x = Normal("X_saddle", 0, 1 / sp.sqrt(n))

    result = asymptotic_expectation(sp.exp(x), x, parameter=n, terms=3, method="laplace")

    expected = 1 + 1 / (2 * n) + 1 / (8 * n**2)
    assert result.method == "laplace-interior-saddle"
    assert result.status == "FORMAL"
    assert sp.simplify(result.expression - expected) == 0


def test_generic_interior_laplace_integral():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(
        sp.exp(-n * x**2 / 2), x, (-sp.oo, sp.oo), parameter=n, terms=3
    )

    assert result.method == "laplace-interior-saddle"
    assert sp.simplify(result.expression - sp.sqrt(2 * sp.pi / n)) == 0


def test_generic_endpoint_laplace_integral_has_watson_coefficients():
    n, a, x = sp.symbols("n a x", positive=True)

    result = laplace_asymptotic_integral(sp.exp(-n * x**2 / 2), x, (a, sp.oo), parameter=n, terms=3)

    expected = sp.exp(-(a**2) * n / 2) * (1 / (a * n) - 1 / (a**3 * n**2) + 3 / (a**5 * n**3))
    assert result.method == "laplace-lower-endpoint"
    assert sp.simplify(result.expression - expected) == 0


def test_laplace_route_returns_unknown_reduction_when_no_laplace_scale_exists():
    n = sp.symbols("n", positive=True)
    x = Uniform("X_unknown", 0, n)

    result = asymptotic_expectation(x, x, parameter=n, method="laplace")

    assert result.status == "UNKNOWN"
    assert result.method == "density-reduction"
    assert result.expression.has(sp.Integral)


def test_continuous_point_event_is_exactly_zero_without_integration():
    n = sp.symbols("n", positive=True)
    x = Normal("X_point", 0, n)

    result = asymptotic_probability(sp.Eq(x, 0), x, parameter=n, method="density")

    assert result.status == "EXACT"
    assert result.expression == 0
    assert result.method == "continuous-point-event"


def test_method_type_mismatch_is_rejected():
    n = sp.symbols("n", positive=True)
    x = Normal("X_method", 0, 1)

    try:
        asymptotic_expectation(x, x, parameter=n, method="pmf")
    except TypeError as exc:
        assert "discrete" in str(exc)
    else:
        raise AssertionError("continuous PMF method should fail")


def test_equal_competing_saddles_are_summed_not_silently_dropped():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(
        sp.exp(-n * (x**2 - 1) ** 2),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        terms=1,
    )

    assert result.method == "laplace-co-dominant-points"
    expected = sp.sqrt(sp.pi / n)
    assert sp.simplify(result.expression - expected) == 0


def test_degenerate_saddle_uses_quartic_not_gaussian_scale():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )

    assert result.method == "laplace-degenerate-saddle-order-4"
    assert result.status == "CERTIFIED"
    expected = sp.gamma(sp.Rational(1, 4)) / (2 * n ** sp.Rational(1, 4))
    assert sp.simplify(result.expression - expected) == 0
