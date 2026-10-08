"""Independent mathematical regressions for the univariate corpus audit."""

import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus


def test_rotating_exponential_is_complex_infinity_on_one_side():
    x = sp.Symbol("x")
    assert one_sided_limit(sp.exp((2 - 2 * sp.I) / x), x, 0, direction="+") == sp.zoo
    assert (
        limit(sp.exp((2 - 2 * sp.I) / x), x, 0, return_result=True).status
        is LimitStatus.DOES_NOT_EXIST
    )


def test_principal_log_exponential_corpus_0033():
    x = sp.Symbol("x")
    assert (
        one_sided_limit(
            sp.exp(sp.tan(x) / sp.log(sp.cos(x))), x, sp.pi / 2, direction="+"
        )
        == sp.zoo
    )


def test_correlated_periodic_phases_do_not_prove_nonexistence():
    x = sp.Symbol("x")
    assert limit(sp.sin(x) ** 2 + sp.cos(x) ** 2, x, sp.oo) == 1
    assert limit(sp.sin(sp.sqrt(x + 1)) - sp.sin(sp.sqrt(x)), x, sp.oo) == 0


def test_undefined_sequence_does_not_acquire_a_growth_theorem():
    n = sp.Symbol("n")
    f = sp.Function("a")
    assert (
        limit(f(n) ** (1 / n), n, sp.oo, return_result=True).status
        is LimitStatus.UNKNOWN
    )


def test_root_derivative_respects_complex_half_plane_assumption():
    a, b, c = sp.symbols("a b c")
    expression = (sp.sqrt(b * b - 4 * a * c) - b) / (2 * a)
    assert limit(expression, a, 0, assumptions=sp.re(b) > 0) == -c / b


def test_real_cut_tangency_of_order_three_has_two_boundary_values():
    x = sp.Symbol("x")
    expression = sp.log(sp.sin(3 * sp.pi * sp.exp(sp.I * x) / 2))
    assert (
        limit(expression, x, 0, return_result=True).status is LimitStatus.DOES_NOT_EXIST
    )


def test_polylog_inversion_keeps_constant_quadratic_cancellation():
    x = sp.Symbol("x")
    expression = x * x / 2 - x * sp.log(1 + sp.exp(x)) - sp.polylog(2, -sp.exp(x))
    assert limit(expression, x, sp.oo) == sp.pi**2 / 6


def test_real_exponential_integral_series_cut_constant():
    x = sp.Symbol("x")
    assert (
        one_sided_limit(sp.log(x) - sp.Ei(sp.log(1 - x)), x, 0, direction="+")
        == -sp.EulerGamma
    )


def test_prime_counting_error_is_preserved_under_cancellation():
    x = sp.Symbol("x")
    assert limit((sp.primepi(x) - x / sp.log(x)) * sp.log(x) ** 2 / x, x, sp.oo) == 1
    assert (
        limit(
            (sp.primepi(x) - x / sp.log(x) - x / sp.log(x) ** 2) * sp.log(x) ** 3 / x,
            x,
            sp.oo,
        )
        == 2
    )


def test_parameter_signs_survive_radical_dispatch():
    a, x = sp.symbols("a x")
    assert (
        limit((a * a - a * x) / (a - sp.sqrt(a * x)), x, a, assumptions=a > 0) == 2 * a
    )


def test_exponentially_growing_periodic_amplitude_subsequence_conflict():
    x = sp.Symbol("x")
    r = limit(sp.exp(x) * sp.sin(x) / x, x, sp.oo, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_positive_gamma_sum_does_not_decay():
    x = sp.Symbol("x")
    assert limit(x + sp.gamma(sp.exp(-x)), x, sp.oo) == sp.oo


def test_principal_root_polynomial_tail_keeps_approached_cut_side():
    x = sp.Symbol("x")
    expression = (x - sp.I) * sp.sqrt(x + sp.I) / sp.sqrt((x - sp.I) ** 2 * (x + sp.I))
    assert limit(expression, x, -sp.oo) == 1


def test_integer_hankel_pole_survives_scalar_expansion():
    x = sp.Symbol("x")
    assert limit(x**3 * sp.hankel1(3, x), x, 0) == -16 * sp.I / sp.pi


def test_negative_infinity_polylog_uses_principal_log_of_negative_x():
    x = sp.Symbol("x")
    assert (
        limit(sp.log(x) / 6 + sp.polylog(3, x) / sp.log(x) ** 2, x, -sp.oo)
        == sp.I * sp.pi / 2
    )


def test_complex_power_norm_uses_real_part_of_parameter():
    x, s = sp.symbols("x s")
    e = x * ((x + 1) ** sp.re(s) * sp.log(x) / (sp.Abs(x**s) * sp.log(x + 1)) - 1)
    assert limit(e, x, sp.oo) == sp.re(s)


def test_real_principal_log_norm_has_two_sided_pole_conflict():
    x = sp.Symbol("x")
    e = 1 / (x - sp.Abs(sp.log(2) + sp.I * sp.pi) + sp.Abs(sp.log(-x - 2)))
    assert one_sided_limit(e, x, 0, direction="+") == sp.oo
    assert limit(e, x, 0, return_result=True).status is LimitStatus.DOES_NOT_EXIST


def test_0721_cluster_certificate_precedes_geometry():
    from asymptotic.instrumentation import symbolic_metrics

    x, y = sp.symbols("x y", real=True)
    r2 = x * x + y * y
    expr = 2 * r2 ** sp.Rational(3, 2) * sp.cos(1 / r2) - sp.sin(1 / r2)
    with symbolic_metrics() as metrics:
        result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert result.evidence[0].method == "vanishing_perturbation_cluster"
    assert metrics.radial_reduction_calls == metrics.path_conflict_calls == 0


def test_float_target_does_not_reverse_tiny_logarithm_argument():
    import mpmath as mp

    z = sp.Symbol("z")
    expression = sp.log(1 - (sp.log(z) + sp.log(-1 + sp.exp(z) / z)) / z) / z
    value = limit(expression, z, sp.Float(100))
    # Evaluate separately: symbolic subtraction can itself rewrite principal
    # logarithms through the tiny cancellation and corrupt the oracle's branch.
    with mp.workdps(120):
        expected = mp.log(-mp.log1p(-100 * mp.exp(-100)) / 100) / 100
        oracle = sp.Float(str(expected), 100)
    assert abs(sp.N(value, 100) - oracle) < sp.Float("1e-70")


def test_special_function_cut_has_attained_signed_boundary_germs():
    x = sp.Symbol("x")
    result = limit(sp.li(-sp.I * x - sp.sqrt(2)), x, 0, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert len(result.evidence) == 2 and all(
        e.method == "attained_li_signed_cut_subsequence" and e.substitutions
        for e in result.evidence
    )
