"""Shared certificates must prove families and decline unsound extrapolations."""

import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.gamma_exponential_germs import (
    exponential_envelope_certificate,
    positive_gamma_tail_certificate,
    signed_gamma_germ_certificate,
)
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_normalization import scalar_reference_namespace


def test_positive_gamma_poles_and_beta_scales():
    x = sp.Symbol("x", positive=True)
    for c in (sp.Rational(1, 2), -3):
        assert limit(sp.gamma(sp.exp(c - x)), x, sp.oo) == sp.oo
    assert limit(sp.gamma(sp.exp(-x)) * sp.exp(-x), x, sp.oo) == 1
    assert limit(sp.beta(sp.exp(-x), sp.exp(-x)) * sp.exp(-x), x, sp.oo) == 2
    assert (
        limit(
            sp.sqrt(x)
            * (sp.beta(sp.exp(-x), sp.exp(-x)) - sp.beta(x, x))
            * sp.exp(x * (1 + 2 * sp.log(2))),
            x,
            sp.oo,
        )
        == sp.oo
    )
    assert (
        limit(
            sp.sqrt(x)
            * sp.beta(2 * x, x)
            * sp.exp(x * (-sp.sqrt(3) + 2 * sp.log(2 + sp.sqrt(3)))),
            x,
            sp.oo,
        )
        == 0
    )


def test_small_gamma_errors_are_or_exponents():
    x = sp.Symbol("x", positive=True)
    for e in (
        sp.exp(sp.gamma(sp.exp(-x)) - sp.exp(x)),
        sp.gamma(sp.exp(-x)) ** sp.exp(x) * sp.exp(-x * sp.exp(x)),
        sp.gamma(sp.exp(-x)) - sp.exp(x),
    ):
        assert positive_gamma_tail_certificate(e, x, sp.oo) is None


def test_fixed_complex_upper_gamma_amplification_guard():
    x = sp.Symbol("x", positive=True)
    n, a = sp.symbols("n a")
    for e in (x * sp.uppergamma(n, x), x**a * sp.uppergamma(n, 3 * x)):
        assert positive_gamma_tail_certificate(e, x, sp.oo)[1] == 0
    assert (
        positive_gamma_tail_certificate(sp.exp(x) * sp.uppergamma(n, x), x, sp.oo)
        is None
    )
    assert positive_gamma_tail_certificate(sp.uppergamma(x, x), x, sp.oo) is None


def test_gamma_bounded_shift_is_uniform_and_rejects_unbounded_shift():
    x = sp.Symbol("x", positive=True)
    for s in (sp.sin(x), 3 * sp.cos(x) + 2):
        assert limit(sp.gamma(x + s) / (sp.gamma(x) * x**s), x, sp.oo) == 1
    assert (
        positive_gamma_tail_certificate(
            sp.gamma(x + sp.sqrt(x)) / (sp.gamma(x) * x ** sp.sqrt(x)), x, sp.oo
        )
        is None
    )


def test_gamma_density_keeps_complex_fixed_parameters():
    v = sp.Symbol("v", positive=True)
    z = sp.Symbol("z")
    e = (v / (v + z * z)) ** ((v + 1) / 2) / (
        sp.sqrt(v) * sp.beta(v / 2, sp.Rational(1, 2))
    )
    assert limit(e, v, sp.oo) == sp.exp(-z * z / 2) / sp.sqrt(2 * sp.pi)


def test_imaginary_gamma_argument_has_two_attained_subsequences():
    x = sp.Symbol("x", real=True)
    r = limit(sp.arg(sp.gamma(sp.I * x)), x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert len(r.evidence) == 2
    assert {e.value for e in r.evidence} == {-sp.pi / 2, sp.pi / 2}
    assert all(e.substitutions for e in r.evidence)
    assert (
        one_sided_limit(sp.arg(sp.gamma(sp.I * x)), x, 0, direction="+") == -sp.pi / 2
    )
    assert one_sided_limit(sp.arg(sp.gamma(sp.I * x)), x, 0, direction="-") == sp.pi / 2


def test_gamma_recurrence_before_pole_substitution():
    x = sp.Symbol("x")
    assert limit(sp.factorial(x + 2) / sp.factorial(x), x, -3) == 2
    assert limit(sp.gamma(x + 4) / sp.gamma(x), x, -1) == 0


def test_polygamma_subtraction_orders_and_sides():
    x = sp.Symbol("x", real=True)
    for m in range(1, 5):
        e = sp.polygamma(m, x) - (-1) ** (m + 1) * sp.factorial(m) / x ** (m + 1)
        assert limit(e, x, 0) == sp.polygamma(m, 1)
    assert one_sided_limit(sp.polygamma(0, x), x, 0, direction="+") == -sp.oo
    assert one_sided_limit(sp.polygamma(0, x), x, 0, direction="-") == sp.oo
    assert signed_gamma_germ_certificate(sp.polygamma(0, x), x, 0) is None
    assert (
        signed_gamma_germ_certificate(
            (sp.polygamma(1, x) - 1 / x**2 - sp.zeta(2)) / x**2, x, 0
        )
        is None
    )


def test_scaled_upper_gamma_zero_both_rays_and_parameter_guard():
    t = sp.Symbol("t", real=True)
    a, b = sp.symbols("a b")
    e = a * (-sp.I * b * t) ** a * sp.uppergamma(-a, -sp.I * b * t)
    assert limit(e, t, 0, assumptions=(a > 1) & (b > 1)) == 1
    assert signed_gamma_germ_certificate(e, t, 0) is None


def test_digamma_exponential_arguments_preserve_absolute_errors():
    n = sp.Symbol("n", positive=True)
    a = sp.Symbol("a")
    e = sp.polygamma(0, a ** (n + 1)) - sp.polygamma(0, a**n)
    assert limit(e, n, sp.oo, assumptions=a > 1) == sp.log(a)
    assert signed_gamma_germ_certificate(e, n, sp.oo) is None
    assert (
        signed_gamma_germ_certificate(n * sp.polygamma(0, n) - n * sp.log(n), n, sp.oo)
        is None
    )


def test_real_oscillator_outward_bounds_do_not_prove_nonexistence():
    x = sp.Symbol("x", positive=True)
    for e in (
        x * (2 + sp.sin(x)),
        x * (1 + sp.exp(sp.sin(x))),
        sp.exp(x) - x ** (2 + sp.cos(x)),
    ):
        assert limit(e, x, sp.oo) == sp.oo
    assert limit(sp.log(x**4 + x ** (2 + sp.sin(x))) / sp.log(1 + x * x), x, sp.oo) == 2
    for e in (sp.sin(x), x * (1 + sp.sin(x)), sp.sin(sp.I * x)):
        assert exponential_envelope_certificate(e, x, sp.oo) is None


def test_parameter_expm1_and_exponential_dominance():
    x = sp.Symbol("x", positive=True)
    a, b = sp.symbols("a b")
    assert limit(x * (sp.exp(x ** (-b)) - 1), x, sp.oo, assumptions=b > 1) == 0
    assert limit(a**x / x**a, x, sp.oo, assumptions=a > 1) == sp.oo
    assert exponential_envelope_certificate(a**x / x**a, x, sp.oo) is None


def test_negative_base_phases_finite_limits_are_uniform():
    x = sp.Symbol("x", positive=True)
    for b in (2, 5):
        assert limit(
            ((-b) ** x + (b + 1) ** x) / ((-b) ** (x + 1) + (b + 1) ** (x + 1)),
            x,
            sp.oo,
        ) == sp.Rational(1, b + 1)
    assert limit((-1) ** x * x / ((-1) ** x * x + 1), x, sp.oo) == 1
    assert exponential_envelope_certificate((-1) ** x, x, sp.oo) is None
    assert exponential_envelope_certificate((-2) ** x, x, sp.oo) is None
    assert exponential_envelope_certificate(1 / (1 + (-1) ** x), x, sp.oo) is None


def test_positive_directed_infinity_requires_positive_dominant_phase():
    x = sp.Symbol("x", positive=True)
    e = (1 - sp.sqrt(13)) ** x * (13 - 5 * sp.sqrt(13)) + (1 + sp.sqrt(13)) ** x * (
        13 + 5 * sp.sqrt(13)
    )
    r = limit(e, x, sp.oo, return_result=True)
    assert r.value == sp.oo
    assert "direction_exponential" in r.evidence[0].method
    assert exponential_envelope_certificate(3**x + (-4) ** x, x, sp.oo) is None


def test_source_negative_top_binomial_integer_values():
    x = sp.Symbol("x")
    e = scalar_reference_namespace()["binomial"](-5, x - 8)
    assert e.func is sp.binomial
    assert e.subs(x, 0) == 0
    # Do not infer a finite-valued analytic germ on an identically pole slice.
    assert limit(e, x, 0, return_result=True).status is LimitStatus.UNKNOWN
