import sympy as sp

from asymptotic import asymptotic_sum
from asymptotic.instrumentation import symbolic_metrics


def test_termwise_parameter_series_sums_coefficients():
    x = sp.symbols("x", positive=True)
    k = sp.symbols("k", integer=True)
    m = sp.symbols("m", positive=True, integer=True)

    result = asymptotic_sum(
        sp.exp(-x * k),
        k,
        1,
        m,
        parameter=x,
        point=0,
        terms=3,
        method="series",
    )

    expected = m - x * m * (m + 1) / 2 + x**2 * m * (m + 1) * (2 * m + 1) / 12
    assert result.method == "termwise-series"
    assert result.status == "CERTIFIED"
    assert result.certificate is not None and result.certificate.replay()
    assert sp.simplify(result.expression - expected) == 0


def test_summation_by_parts_builds_exponential_tail_prefix():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", integer=True)
    q = sp.exp(-1)

    result = asymptotic_sum(
        sp.exp(-k) / k,
        k,
        n,
        sp.oo,
        parameter=n,
        terms=3,
        method="summation-by-parts",
    )

    expected = sp.exp(-n) * (
        1 / ((1 - q) * n) - q / ((1 - q) ** 2 * n**2) + q * (1 + q) / ((1 - q) ** 3 * n**3)
    )
    scaled = sp.cancel(sp.exp(n) * (result.expression - expected))
    assert result.method == "summation-by-parts"
    assert result.status == "FORMAL"
    assert sp.limit(n**3 * scaled, n, sp.oo) == 0


def test_riemann_route_recovers_leading_scaled_integral():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", integer=True)

    result = asymptotic_sum(
        (k / n) ** 2,
        k,
        1,
        n,
        parameter=n,
        terms=2,
        method="riemann",
    )

    assert result.method == "riemann-sum"
    assert result.status == "FORMAL"
    assert sp.simplify(result.expression - n / 3) == 0
    assert result.transformation is not None


def test_mellin_route_handles_bessel_k_lattice_sum():
    s = sp.symbols("s", positive=True)
    k = sp.symbols("k", positive=True, integer=True)

    result = asymptotic_sum(
        sp.besselk(0, s * k),
        k,
        1,
        sp.oo,
        parameter=s,
        point=0,
        terms=3,
        method="mellin",
    )

    assert result.method == "mellin-poles"
    assert result.status == "CERTIFIED"
    assert result.certificate is not None and result.certificate.replay()
    assert sp.simplify(sp.limit(s * result.expression, s, 0) - sp.pi / 2) == 0
    remainder = result.expression - sp.pi / (2 * s)
    assert sp.simplify(sp.limit(remainder / sp.log(s), s, 0) - sp.Rational(1, 2)) == 0


def test_new_sum_routes_do_not_enter_general_solve_or_rsolve():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", integer=True)

    with symbolic_metrics() as metrics:
        result = asymptotic_sum(
            (k / n) ** 2,
            k,
            1,
            n,
            parameter=n,
            terms=2,
            method="riemann",
        )

    assert result.status == "FORMAL"
    assert metrics.general_solve_calls == 0
    assert metrics.general_rsolve_calls == 0
