import sympy as sp

from asymptotic import asymptotic_sum
from asymptotic.sum_advanced import CreativeTelescopingCertificate


def test_zeilberger_generates_replayable_recurrence_and_solves_sum():
    n = sp.symbols("n", nonnegative=True, integer=True)
    k = sp.symbols("k", nonnegative=True, integer=True)

    result = asymptotic_sum(
        sp.binomial(n, k),
        k,
        0,
        sp.oo,
        parameter=n,
        point=sp.oo,
        terms=3,
        method="zeilberger",
    )

    assert result.method == "creative-telescoping"
    assert result.status == "FORMAL"
    assert sp.simplify(result.expression - 2**n) == 0
    assert isinstance(result.certificate, CreativeTelescopingCertificate)
    assert result.certificate.replay()


def test_partial_sum_recurrence_feeds_asymptotic_rsolve():
    n = sp.symbols("n", nonnegative=True, integer=True)
    k = sp.symbols("k", positive=True, integer=True)

    result = asymptotic_sum(
        1 / k,
        k,
        1,
        n,
        parameter=n,
        point=sp.oo,
        terms=3,
        method="zeilberger",
    )

    assert result.method == "creative-telescoping"
    assert result.status in {"FORMAL", "CERTIFIED"}
    assert result.expression.has(sp.log(n)) or result.expression.has(sp.harmonic(n))


def test_mellin_shift_is_certified_when_vertical_decay_is_proved():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", positive=True, integer=True)

    result = asymptotic_sum(
        sp.besselk(0, x * k),
        k,
        1,
        sp.oo,
        parameter=x,
        point=0,
        terms=3,
        method="mellin",
    )

    assert result.status == "CERTIFIED"
    assert result.certificate is not None
    assert result.certificate.initial_strip == (1, sp.oo)
    assert result.certificate.initial_line == 2
    assert result.certificate.replay()
    assert result.remainder is not None and result.remainder.is_certified
    assert sp.simplify(sp.limit(x * result.expression, x, 0) - sp.pi / 2) == 0


def test_poisson_gaussian_sum_has_exponentially_small_certified_tail():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", integer=True)

    result = asymptotic_sum(
        sp.exp(-x * k**2),
        k,
        -sp.oo,
        sp.oo,
        parameter=x,
        point=0,
        terms=2,
        method="poisson",
    )

    assert result.status == "CERTIFIED"
    assert result.method == "poisson-summation"
    assert sp.simplify(result.expression - sp.sqrt(sp.pi / x)) == 0
    assert result.remainder is not None
    assert result.remainder.scale.has(sp.exp(-(sp.pi**2) / x))


def test_finite_oscillatory_exponential_sum_is_exact():
    x = sp.symbols("x", real=True)
    k = sp.symbols("k", integer=True)

    result = asymptotic_sum(
        sp.exp(sp.I * x * k),
        k,
        0,
        10,
        parameter=x,
        point=0,
        terms=3,
        method="oscillatory",
    )

    assert result.status == "EXACT"
    exact = sp.summation(sp.exp(sp.I * x * k), (k, 0, 10))
    assert sp.simplify(result.expression - exact) == 0
    prefix = sp.series(result.expression, x, 0, 3).removeO().expand()
    assert prefix.coeff(x, 0) == 11
    assert prefix.coeff(x, 1) == 55 * sp.I


def test_multidimensional_separable_sum_factors_without_nested_generic_sum():
    x = sp.symbols("x")
    i, j = sp.symbols("i j", integer=True)

    result = asymptotic_sum(
        (1 + x * i) * (1 + x * j),
        (i, j),
        (0, 0),
        (2, 3),
        parameter=x,
        point=0,
        terms=3,
        method="series",
    )

    assert result.method == "multidimensional-separable"
    assert sp.expand(result.expression - (3 + 3 * x) * (4 + 6 * x)) == 0


def test_termwise_infinite_series_requires_and_replays_uniform_majorant():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", positive=True, integer=True)

    result = asymptotic_sum(
        1 / (k**2 * (1 + x / k)),
        k,
        1,
        sp.oo,
        parameter=x,
        point=0,
        terms=3,
        method="series",
    )

    assert result.status == "CERTIFIED"
    assert result.certificate is not None and result.certificate.replay()
    expected = sp.zeta(2) - x * sp.zeta(3) + x**2 * sp.zeta(4)
    assert sp.simplify(result.expression - expected) == 0


def test_poisson_refuses_phase_outside_principal_dual_cell():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", integer=True)

    result = asymptotic_sum(
        sp.exp(-x * k**2 + sp.Rational(3, 2) * sp.pi * sp.I * k),
        k,
        -sp.oo,
        sp.oo,
        parameter=x,
        point=0,
        terms=2,
        method="poisson",
    )

    assert result.status == "UNKNOWN"


def test_poisson_requires_positive_small_parameter_for_certification():
    x = sp.symbols("x", real=True)
    k = sp.symbols("k", integer=True)

    result = asymptotic_sum(
        sp.exp(-x * k**2),
        k,
        -sp.oo,
        sp.oo,
        parameter=x,
        point=0,
        terms=2,
        method="poisson",
    )

    assert result.status == "UNKNOWN"
