import sympy as sp

from asymptotic import (
    AsymptoticSumResult,
    RemainderKind,
    airy_uniform_saddle_asymptotic,
    asymptotic_sum,
    coalescing_saddle_asymptotic,
    laplace_asymptotic_integral,
)
from asymptotic.probability import LaplaceRemainderCertificate


def test_quartic_degenerate_saddle_has_gamma_scale_and_certificate():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )

    assert result.method == "laplace-degenerate-saddle-order-4"
    assert result.status == "CERTIFIED"
    assert (
        sp.simplify(result.expression - sp.gamma(sp.Rational(1, 4)) / (2 * n ** sp.Rational(1, 4)))
        == 0
    )
    assert isinstance(result.certificate, LaplaceRemainderCertificate)
    assert result.certificate.certified
    assert result.certificate.local_orders == (4,)
    assert result.remainder is not None
    assert result.remainder.kind is RemainderKind.BIG_O


def test_sextic_degenerate_saddle_uses_one_sixth_scale():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(
        sp.exp(-n * x**6), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )

    expected = sp.gamma(sp.Rational(1, 6)) / (3 * n ** sp.Rational(1, 6))
    assert result.method == "laplace-degenerate-saddle-order-6"
    assert sp.simplify(result.expression - expected) == 0


def test_stationary_quartic_endpoint_is_not_double_counted():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(sp.exp(-n * x**4), x, (0, sp.oo), parameter=n, terms=2)

    expected = sp.gamma(sp.Rational(1, 4)) / (4 * n ** sp.Rational(1, 4))
    assert result.method == "laplace-lower-degenerate-endpoint-order-4"
    assert sp.simplify(result.expression - expected) == 0


def test_quartic_coalescence_gives_uniform_transition_profile():
    n = sp.symbols("n", positive=True)
    mu = sp.symbols("mu", real=True)
    x = sp.symbols("x", real=True)

    result = coalescing_saddle_asymptotic(
        sp.exp(-n * (x**4 / 4 + mu * x**2 / 2)),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        control_parameter=mu,
        terms=1,
    )

    assert result.method == "laplace-coalescing-quartic"
    assert result.status == "FORMAL"
    at_transition = sp.simplify(result.expression.subs(mu, 0).doit())
    expected = sp.gamma(sp.Rational(1, 4)) / (sp.sqrt(2) * n ** sp.Rational(1, 4))
    assert sp.simplify(at_transition - expected) == 0
    assert result.expression.has(mu * sp.sqrt(n))


def test_global_certificate_stays_formal_for_nonpolynomial_amplitude():
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)

    result = laplace_asymptotic_integral(
        sp.exp(-(x**2)) * sp.exp(-n * x**2 / 2),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        terms=2,
    )

    assert result.status == "FORMAL"
    assert result.certificate is not None
    assert not result.certificate.certified
    assert "amplitude" in result.certificate.reason


def test_euler_maclaurin_tail_has_certified_remainder():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", positive=True, integer=True)

    result = asymptotic_sum(
        1 / k**2,
        k,
        n,
        sp.oo,
        parameter=n,
        terms=3,
        method="euler-maclaurin",
    )

    assert isinstance(result, AsymptoticSumResult)
    assert result.method == "euler-maclaurin"
    assert result.status == "CERTIFIED"
    expected = 1 / n + 1 / (2 * n**2) + 1 / (6 * n**3)
    assert sp.series(result.expression - expected, n, sp.oo, 4).removeO() == 0
    assert result.remainder is not None


def test_discrete_gaussian_lattice_saddle():
    n = sp.symbols("n", positive=True)
    k = sp.symbols("k", integer=True)

    result = asymptotic_sum(
        sp.exp(-n * (k / n) ** 2 / 2),
        k,
        -sp.oo,
        sp.oo,
        parameter=n,
        terms=2,
        method="saddle",
    )

    assert result.method == "discrete-laplace-interior-saddle"
    assert result.status == "CERTIFIED"
    assert sp.simplify(result.expression - sp.sqrt(2 * sp.pi * n)) == 0
    assert result.transformation is not None


def test_exact_geometric_sum_is_preferred():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", integer=True)
    q = sp.symbols("q", positive=True)

    result = asymptotic_sum(q**k, k, n, sp.oo, parameter=n, terms=2, method="exact")

    # Without q<1 SymPy correctly keeps convergence conditions/unevaluated
    # behavior rather than the asymptotics layer inventing a geometric tail.
    assert result.status in {"EXACT", "UNKNOWN"}


def test_cubic_turning_point_has_uniform_airy_scaling():
    n = sp.symbols("n", positive=True)
    mu = sp.symbols("mu", real=True)
    x = sp.symbols("x", real=True)

    result = airy_uniform_saddle_asymptotic(
        sp.exp(sp.I * n * (x**3 / 3 + mu * x)),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        control_parameter=mu,
    )

    expected = 2 * sp.pi * sp.airyai(mu * n ** sp.Rational(2, 3)) / n ** sp.Rational(1, 3)
    assert result.method == "oscillatory-airy-uniform-saddle"
    assert result.status == "FORMAL"
    assert sp.simplify(result.expression - expected) == 0
    assert result.expression.has(sp.airyai(mu * n ** sp.Rational(2, 3)))
