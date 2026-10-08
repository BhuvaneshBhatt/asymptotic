"""Compact, cancellation-aware regressions for profiled timeout families."""

import sympy as sp

from asymptotic import limit
from asymptotic.compact_limit_germs import (
    compact_rational_certificate,
    gamma_stirling_product_certificate,
    lambert_small_scale_certificate,
)


def test_billion_degree_power_avoids_polynomial_expansion():
    from asymptotic.instrumentation import symbolic_metrics

    x = sp.Symbol("x")
    with symbolic_metrics() as metrics:
        assert limit((x / 2 + 1) ** 3000000000, x, sp.oo) == sp.oo
        assert limit((x + 3) ** 1984, x, -4) == 1
    assert metrics.full_limit_calls == 0


def test_compact_large_rational_quotients_and_signs():
    x = sp.Symbol("x")
    assert limit((x**10000 - 1) / (x - 1), x, sp.oo) == sp.oo
    assert limit((1 - x**-10000) / (1 - 1 / x), x, sp.oo) == 1
    assert limit((-3 * x + 1) ** 1999, x, sp.oo) == -sp.oo


def test_compact_leading_cancellation_requires_another_order():
    x = sp.Symbol("x", positive=True)
    assert compact_rational_certificate((x + 1) ** 10000 - x**10000, x, sp.oo) is None


def test_lambert_small_scale_retains_quartic_cancellation():
    x = sp.Symbol("x")
    w = sp.LambertW(x**3)
    e = x**6 * (sp.cos(sp.sqrt(w / x**3)) - 1 + w / (2 * x**3)) / w**2
    assert limit(e, x, sp.oo) == sp.Rational(1, 24)


def test_lambert_shifted_and_logarithmic_argument_scales():
    x = sp.Symbol("x")
    g = x * x + 1
    w = sp.LambertW(g)
    assert limit(
        g * (sp.exp(sp.sqrt(w / g)) - 1) / w - sp.sqrt(g / w), x, sp.oo
    ) == sp.Rational(1, 2)
    g = x * sp.log(x)
    w = sp.LambertW(g)
    assert limit(g * (sp.cos(sp.sqrt(w / g)) - 1) / w, x, sp.oo) == -sp.Rational(1, 2)


def test_nonprincipal_lambert_branch_is_not_normalized_as_positive():
    x = sp.Symbol("x", positive=True)
    assert (
        lambert_small_scale_certificate(sp.exp(sp.LambertW(x, -1)) / x, x, sp.oo)
        is None
    )


def test_stirling_products_preserve_constants_and_small_root():
    x = sp.Symbol("x")
    assert limit(sp.gamma(x + 1) ** (1 / x) / x, x, sp.oo) == sp.exp(-1)
    e = (
        2 ** (4 * x + 1)
        * sp.gamma(x + 1) ** 4
        / ((2 * x + 1) * sp.gamma(2 * x + 1) ** 2)
    )
    assert limit(e, x, sp.oo) == sp.pi
    assert limit(
        sp.exp(x) * sp.gamma(x + 1) / (sp.sqrt(x) * x**x), x, sp.oo
    ) == sp.sqrt(2 * sp.pi)


def test_stirling_does_not_turn_an_oscillatory_enclosure_into_a_value():
    x = sp.Symbol("x", positive=True)
    e = sp.gamma(x + 1) * sp.exp(-x * sp.log(x) + x - sp.log(x) / 2 + sp.cos(x))
    assert gamma_stirling_product_certificate(e, x, sp.oo) is None
    assert gamma_stirling_product_certificate(sp.gamma(-x), x, sp.oo) is None


def test_exact_parameter_equality_precedes_generic_gamma_analysis():
    from asymptotic.instrumentation import symbolic_metrics

    x, z = sp.symbols("x z")
    with symbolic_metrics() as metrics:
        assert (
            limit(
                sp.gamma(x + 1) / sp.gamma(x + z + 1), x, sp.oo, assumptions=sp.Eq(z, 0)
            )
            == 1
        )
    assert metrics.univariate_calls == 0


def test_real_monotone_gamma_compositions_and_decay():
    x = sp.Symbol("x")
    assert limit(sp.log(sp.gamma(sp.gamma(x))), x, sp.oo) == sp.oo
    assert limit(sp.gamma(x + 1 / sp.gamma(x)), x, sp.oo) == sp.oo
    assert limit(sp.exp(-x * sp.log(sp.gamma(sp.gamma(x)))), x, sp.oo) == 0


def test_bounded_complex_cosh_envelope_and_floor_sides():
    from asymptotic import one_sided_limit

    x = sp.Symbol("x")
    assert limit(sp.cosh(2 * sp.pi + sp.I * sp.pi * x) / x, x, sp.oo) == 0
    assert one_sided_limit(sp.floor(x), x, 1, direction="-") == 0
    assert one_sided_limit(sp.floor(x), x, 1, direction="+") == 1


def test_real_polylog_part_cancels_exponential_polynomial_log():
    x = sp.Symbol("x")
    e = (
        -x * x / 2
        + x * sp.log((1 - sp.exp(x)) ** 2) / 2
        + sp.re(sp.polylog(2, sp.exp(x)))
    )
    assert limit(e, x, sp.oo) == sp.pi**2 / 3


def test_unknown_polylog_sign_and_nonreal_monotone_root_are_declined():
    from asymptotic.compact_limit_germs import (
        real_monotone_composition_certificate,
    )
    from asymptotic.limit_certificates import polylog_inversion_certificate

    x = sp.Symbol("x", positive=True)
    a = sp.Symbol("a", real=True, nonzero=True)
    assert polylog_inversion_certificate(sp.polylog(2, a * sp.exp(x)), x, sp.oo) is None
    assert (
        real_monotone_composition_certificate(sp.gamma(x) + sp.sqrt(-x), x, sp.oo)
        is None
    )


def test_gamma_reciprocal_shifts_keep_digamma_cancellation_orders():
    x = sp.Symbol("x")
    g = sp.gamma(x)
    assert limit((sp.gamma(x + 1 / g) - g) / sp.log(x), x, sp.oo) == 1
    assert limit(x * (sp.log(x) - g + sp.gamma(x - 1 / g)), x, sp.oo) == sp.Rational(
        1, 2
    )
    assert limit(
        x * ((sp.gamma(x + 1 / g) - g) / sp.log(x) - sp.cos(1 / x)) * sp.log(x),
        x,
        sp.oo,
    ) == -sp.Rational(1, 2)
    assert limit((sp.gamma(x + 2 / g) - g) / sp.log(x), x, sp.oo) == 2


def test_gamma_shift_requires_real_remainder_control():
    from asymptotic.compact_limit_germs import gamma_reciprocal_shift_certificate

    x = sp.Symbol("x", positive=True)
    g = sp.gamma(x)
    assert (
        gamma_reciprocal_shift_certificate(
            x**8 * (sp.gamma(x + 1 / g) - g - sp.log(x)), x, sp.oo
        )
        is None
    )
    assert (
        gamma_reciprocal_shift_certificate(sp.gamma(x + sp.I / g) - g, x, sp.oo) is None
    )


def test_minimum_with_strict_gap_selects_rational_branch():
    x = sp.Symbol("x")
    e = sp.Min(
        (x + 1) / (3 * x - 3),
        sp.cos(sp.sqrt(2) * (x * x + 1)) / 2 + 1,
        sp.Mod(x * x + 1, 2) / 2 + sp.Rational(1, 2),
        sp.cos(sp.sin(x * x + 1) ** 2),
    )
    assert limit(e, x, sp.oo) == sp.Rational(1, 3)


def test_overlap_of_cluster_bounds_does_not_prove_nonexistence():
    from asymptotic.compact_limit_germs import (
        minmax_interval_dominance_certificate,
    )

    x = sp.Symbol("x", positive=True)
    e = sp.Min(
        2 * (x + 1) / (3 * x - 3),
        sp.cos(sp.sqrt(2) * (x * x + 1)) / 2 + 1,
        sp.Mod(x * x + 1, 2) / 2 + sp.Rational(1, 2),
        sp.cos(sp.sin(x * x + 1) ** 2),
    )
    assert minmax_interval_dominance_certificate(e, x, sp.oo) is None
