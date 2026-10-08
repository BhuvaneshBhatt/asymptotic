import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus
from asymptotic.reference_normalization import scalar_reference_namespace
from asymptotic.tail_cancellation_germs import (
    dirichlet_cancellation_certificate,
    gamma_ratio_scale_certificate,
    monomial_phase_subsequences,
    small_log_power_certificate,
    tangent_pole_subsequences,
    zeta_finite_pole_certificate,
)


def test_stirling_dispatch():
    from asymptotic.instrumentation import symbolic_metrics

    k = sp.Symbol("k", positive=True)
    expr = (
        5 ** (-2 * k - 1)
        * sp.gamma(k + 1) ** k
        * sp.gamma(k + sp.Rational(3, 2))
        * sp.gamma(k + 2) ** (-k - 1)
        / sp.gamma(k + sp.Rational(1, 2))
    )
    with symbolic_metrics() as metrics:
        result = limit(expr, k, sp.oo, return_result=True)
    assert result.value == 0
    assert metrics.recurrence_calls == 0


def test_recurrence_fallback_retries_certificates_general_analysis():
    from asymptotic.instrumentation import symbolic_metrics

    x = sp.Symbol("x", positive=True)
    order = sp.Symbol("order")
    expr = (
        sp.besselj(order, x)
        + sp.besselj(order + 2, x)
        - 2 * (order + 1) * sp.besselj(order + 1, x) / x
    )
    with symbolic_metrics() as metrics:
        result = limit(expr, x, 1, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == 0
    assert result.expression == expr
    assert any("recurrence" in item.method for item in result.evidence)
    assert metrics.recurrence_calls == 1
    assert metrics.general_limit_calls == 0


def test_known_source_functions_and_real_root_branches():
    x = sp.Symbol("x")
    real_root_function = scalar_reference_namespace()["RealRoot"]
    assert one_sided_limit(real_root_function(x - 1, 2) / x, x, 1, direction="+") == 0
    assert (
        limit(
            sp.gamma(x + sp.Rational(1, 2)) / (real_root_function(x, 2) * sp.gamma(x)),
            x,
            sp.oo,
        )
        == 1
    )
    assert (
        limit(real_root_function(x, 2), x, -1, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert (
        limit(real_root_function(x, 2), x, 0, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert (
        limit(sp.Function("f")(x), x, 0, return_result=True).status
        is LimitStatus.UNKNOWN
    )
    assert (
        limit(scalar_reference_namespace()["InverseRegularizedGamma"](1, x), x, 0)
        == sp.oo
    )
    assert (
        limit(
            scalar_reference_namespace()["InverseRegularizedGamma"](2, x),
            x,
            0,
            return_result=True,
        ).status
        is LimitStatus.UNKNOWN
    )


def test_dirichlet_projection_retains_multiple_and_bounds():
    x = sp.Symbol("x", positive=True)
    for n in (3, 4, 5):
        e = n**x * (sp.zeta(x) - sum(j ** (-x) for j in range(1, n)))
        assert limit(e, x, sp.oo) == 1
    e = 8**x * (sp.zeta(x) - sum(j ** (-x) for j in range(1, 7)))
    assert dirichlet_cancellation_certificate(e, x, sp.oo) is None
    assert dirichlet_cancellation_certificate(sp.zeta(x) ** 2, x, sp.oo) is None
    e = (sp.zeta(x) - 1 - 2 ** (-x)) ** 16

    assert dirichlet_cancellation_certificate(e, x, sp.oo) is None


def test_finite_zeta_pole_remainders_and_parameter_guards():
    x = sp.Symbol("x")
    a = sp.Symbol("a")
    assert limit(sp.zeta(x) - 1 / (x - 1), x, 1) == sp.EulerGamma
    e = sp.zeta(x, a) - 1 / (x - 1)
    assert limit(e, x, 1, assumptions=sp.re(a) > 0) == -sp.polygamma(0, a)
    assert zeta_finite_pole_certificate(e, x, 1, sp.S.true) is None
    assert (
        zeta_finite_pole_certificate(
            (sp.zeta(x) - 1 / (x - 1) - sp.EulerGamma) / (x - 1), x, 1, sp.S.true
        )
        is None
    )


def test_harmonic_relative_tails():
    x = sp.Symbol("x")
    assert limit(sp.harmonic(x, -sp.Rational(1, 3)), x, sp.oo) == sp.oo
    assert limit(sp.harmonic(x, -sp.Rational(1, 3)) / x**2, x, sp.oo) == 0
    assert limit(
        sp.harmonic(x, -sp.Rational(1, 3)) / x ** sp.Rational(4, 3), x, sp.oo
    ) == sp.Rational(3, 4)


def test_log_power_remainder_and_gamma_ratio_growth_chart():
    x = sp.Symbol("x", positive=True)
    e = (1 + 1 / x + 1 / x**3) ** x
    assert small_log_power_certificate(e, x, sp.oo)[1] == sp.E
    assert small_log_power_certificate((1 + 1 / x) ** x**2, x, sp.oo) is None
    e = sp.gamma(2 * x + sp.Rational(2, 3)) / (
        sp.gamma(2 * x) * (2 * x) ** sp.Rational(2, 3)
    )
    assert gamma_ratio_scale_certificate(e, x, sp.oo)[1] == 1
    assert (
        gamma_ratio_scale_certificate(sp.gamma(x + 1) - sp.gamma(x), x, sp.oo) is None
    )
    e = sp.atan(2**x * sp.exp(sp.sin(x) / x) / sp.log(x))
    r = limit(e, x, sp.oo, return_result=True)
    assert (
        r.value == sp.pi / 2
        and r.evidence[0].method == "real_oscillatory_outward_squeeze"
    )


def test_attained_monomial_sequences_are_interval_bounds():
    x = sp.Symbol("x")
    for e, p in (
        (sp.sin(x ** sp.Rational(1, 3)), sp.oo),
        ((4 * x * x + 1) * sp.sin(1 / x), 0),
        (sp.csc(1 / x), 0),
    ):
        r = limit(e, x, p, return_result=True)
        assert r.status is LimitStatus.DOES_NOT_EXIST and len(r.evidence) == 2
        assert r.evidence[0].value != r.evidence[1].value
        for ev in r.evidence:
            variable, sequence = ev.substitutions[0]
            assert variable == x
            n = next(iter(sequence.free_symbols))
            assert sp.limit(sequence, n, sp.oo) == p
            along = sp.simplify(e.subs(x, sequence))
            assert not along.has(sp.zoo, sp.nan)
            assert sp.limit(along, n, sp.oo) == ev.value
    assert monomial_phase_subsequences(sp.sin(x), x, sp.oo, x > 1) is None
    assert (
        monomial_phase_subsequences(sp.sec(sp.tan(sp.csc(x))), x, sp.oo, sp.S.true)
        is None
    )
    assert (
        monomial_phase_subsequences(
            sp.sin(x) + sp.exp(sp.I * sp.sqrt(2) * x), x, sp.oo, sp.S.true
        )
        is None
    )
    r = limit(sp.log(1 + sp.sin(x)), x, sp.oo, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    # A phase at log(0) is outside the expression domain, not an attained
    # negative-infinity witness. Use two finite, defined phase values instead.
    assert all(not ev.value.has(sp.oo, -sp.oo, sp.zoo, sp.nan) for ev in r.evidence)
    assert limit(sp.sin(x) / x, x, sp.oo) == 0


def test_tangent_subsequences_avoid_poles_and_retain_amplification():
    x = sp.Symbol("x", positive=True)
    for e, expected in (
        (sp.tan(x) / x, 1 / sp.pi),
        (sp.tan(x) / (1 - x * x), -1 / sp.pi**2),
    ):
        r = limit(e, x, sp.oo, return_result=True)
        assert r.status is LimitStatus.DOES_NOT_EXIST
        assert [ev.value for ev in r.evidence] == [0, expected]
        sequence = r.evidence[1].substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        for k in (1, 2, 10):
            assert sp.cos(sequence.subs(n, k)).is_zero is False
    assert tangent_pole_subsequences(sp.tan(x) / x, x, sp.oo, x > 1) is None


def test_commensurate_frequencies_record_actual_subsequences():
    x = sp.Symbol("x")
    r = limit(sp.sin(300 * x) * sp.cos(x), x, sp.oo, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert len(r.evidence) == 2 and all(ev.substitutions for ev in r.evidence)


def test_upper_gamma_cut_signed_germs_and_nonzero_jump():
    x = sp.Symbol("x")
    e = sp.uppergamma(sp.Rational(1, 4), (x + sp.I) ** 4)
    r = one_sided_limit(e, x, 1, direction="+", return_result=True)
    assert r.value == sp.uppergamma(sp.Rational(1, 4), -4)
    r = limit(e, x, 1, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert (
        len(r.evidence) == 3 and r.evidence[-1].method == "upper_gamma_nonzero_cut_jump"
    )
    result = limit(sp.li(-sp.I * x - sp.sqrt(2)), x, 0, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert all(
        e.method == "attained_li_signed_cut_subsequence" and e.substitutions
        for e in result.evidence
    )


def test_multivariate_recurrence_follows_existing_early_certificates():
    x = sp.Symbol("x", positive=True)
    y = sp.Symbol("y")
    v = sp.Symbol("v")
    e = (
        sp.besselj(v, x)
        + sp.besselj(v + 2, x)
        - 2 * (v + 1) * sp.besselj(v + 1, x) / x
        + y
    )
    r = limit(e, (x, y), (1, 0), return_result=True)
    assert r.value == 0 and r.expression == e
    assert any(ev.method == "integer_shift_recurrence_besselj" for ev in r.evidence)


def test_finite_zeta_projection_checks_error_amplification():
    x = sp.Symbol("x")
    assert zeta_finite_pole_certificate(sp.zeta(x), x, 1, sp.S.true) is None
    e = sp.Heaviside(1 - x) / (x - 1) * (sp.zeta(x) - 1 / (x - 1) - sp.EulerGamma)
    assert zeta_finite_pole_certificate(e, x, 1, sp.S.true) is None
    assert limit(sp.zeta(x) - 1 / (x - 1), x, 1) == sp.EulerGamma
    # Laurent's pole sign is negative below 1; the divergent terms cancel.
    assert limit(x + sp.zeta(1 - 1 / x), x, sp.oo) == sp.EulerGamma


def test_perturbation_routes_decline_unrestricted_accumulating_poles():
    x = sp.Symbol("x", positive=True)
    e = (1 + sp.tan(x) / x) ** x
    assert small_log_power_certificate(e, x, sp.oo) is None
    e = sp.gamma(x + sp.tan(x) + sp.Rational(1, 2)) / sp.gamma(x + sp.tan(x))
    assert gamma_ratio_scale_certificate(e, x, sp.oo) is None
    e = sp.tan(x) * 7**x * (sp.zeta(x) - sum(j ** (-x) for j in range(1, 7)))
    assert dirichlet_cancellation_certificate(e, x, sp.oo) is None


def test_quadratic_radical_uniform_perturbation_keeps_cancellation():
    from asymptotic.tail_cancellation_germs import (
        quadratic_radical_perturbation_certificate,
    )

    x = sp.Symbol("x", positive=True)
    e = x - sp.sqrt(x * x + 10 * x + sp.sin(x))
    r = limit(e, x, sp.oo, return_result=True)
    assert (
        r.value == -5 and r.evidence[0].method == "quadratic_radical_uniform_remainder"
    )
    e = 3 * sp.sqrt(4 * x * x - 7 * x + sp.sin(x)) - 6 * x + 2
    assert limit(e, x, sp.oo) == -sp.Rational(13, 4)
    e = x - sp.sqrt(x * x + x ** sp.Rational(3, 2))
    assert quadratic_radical_perturbation_certificate(e, x, sp.oo) is None
