import sympy as sp

from asymptotic import limit, one_sided_limit
from asymptotic._symbolic_policy import bounded_limit
from asymptotic.function_normalization import normalize_functions
from asymptotic.limit_models import LimitStatus
from asymptotic.rational_pullback_germs import (
    exp_coefficients,
    gamma_rational_pullback_certificate,
    local_real_sign_certificate,
    near_one_adaptive_certificate,
    real_error_tail_certificate,
    rounded_tail_certificate,
)
from asymptotic.reference_normalization import scalar_reference_namespace


def test_gamma_rational_pullback_fourth_order():
    x = sp.Symbol("x", positive=True)
    u = x * x + 1
    # Independent exact integer-shift identity, not a series oracle.
    expr = u**4 * (sp.gamma(u + 2) / (sp.gamma(u) * u**2) - 1 - 1 / u) + 7
    result = gamma_rational_pullback_certificate(expr, x, sp.oo)
    assert result is not None and result[1] == 7
    ratio = sp.gamma(u + sp.Rational(1, 2)) / (sp.gamma(u) * sp.sqrt(u))
    expr = u**4 * (ratio - 1 + 1 / (8 * u) - 1 / (128 * u**2) - 5 / (1024 * u**3))
    result = gamma_rational_pullback_certificate(expr, x, sp.oo)
    assert result is not None and result[1] == -21 / sp.Integer(32768)
    assert gamma_rational_pullback_certificate(expr.subs(x, sp.I * x), x, sp.oo) is None


def test_formal_exp_recurrence():
    assert exp_coefficients([sp.S.One, sp.S.Zero, sp.S.Zero, sp.S.Zero]) == [
        1,
        1,
        sp.Rational(1, 2),
        sp.Rational(1, 6),
        sp.Rational(1, 24),
    ]


def test_near_one_fourth_order_and_budget():
    x = sp.Symbol("x", positive=True)
    p = (1 + 1 / x) ** x
    expr = x**4 * (
        p - sp.E + sp.E / (2 * x) - 11 * sp.E / (24 * x**2) + 7 * sp.E / (16 * x**3)
    )
    r = near_one_adaptive_certificate(expr, x, sp.oo)
    assert r is not None and sp.simplify(r[1] - 2447 * sp.E / 5760) == 0
    assert near_one_adaptive_certificate(x**8 * (p - sp.E), x, sp.oo) is None
    assert near_one_adaptive_certificate(p, x, sp.oo) is None


def test_erfc_rational_tail_with_cancellation():
    x = sp.Symbol("x", positive=True)
    u = x * x + 2
    expr = u**4 * (sp.sqrt(sp.pi) * u * sp.exp(u * u) * sp.erfc(u) - 1 + 1 / (2 * u**2))
    r = real_error_tail_certificate(expr, x, sp.oo)
    assert r is not None and r[1] == sp.Rational(3, 4)
    assert real_error_tail_certificate(sp.exp(2 * x * x) * sp.erfc(x), x, sp.oo) is None
    assert real_error_tail_certificate(sp.erfc(sp.I * x), x, sp.oo) is None
    assert real_error_tail_certificate(sp.erf(-u), x, sp.oo)[1] == -1


def test_local_abs_attained_subsequences():
    x = sp.Symbol("x", real=True)
    for expr in [sp.Abs(x) / x, (x - sp.Abs(x)) / x**2]:
        r = limit(expr, x, 0, return_result=True)
        assert r.status is LimitStatus.DOES_NOT_EXIST
        assert len(r.evidence) == 2 and all(e.substitutions for e in r.evidence)
    assert one_sided_limit(sp.Abs(x) / x, x, 0, direction="+") == 1
    assert one_sided_limit(sp.Abs(x) / x, x, 0, direction="-") == -1
    assert local_real_sign_certificate(sp.Abs(x) / x, x, 0, x > 0, sp.S.true) is None
    p = sp.Symbol("p", positive=True)
    assert local_real_sign_certificate(sp.floor(p), p, -1, sp.S.true, sp.S.true) is None


def test_floor_ceiling_integer_and_noninteger_charts():
    x = sp.Symbol("x", real=True)
    r = limit(sp.ceiling(x), x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in r.evidence} == {0, 1}
    assert limit(sp.floor(2 + x * x), x, 0) == 2
    assert limit(sp.ceiling(2 - x * x), x, 0) == 2
    assert limit(sp.floor(sp.Rational(3, 2) + x), x, 0) == 1
    assert (
        local_real_sign_certificate(
            sp.floor(x), sp.Symbol("k", integer=True), 0, sp.S.true, sp.S.true
        )
        is None
    )


def test_round_scaled_error_and_adverse_amplification():
    x = sp.Symbol("x", positive=True)
    Round = scalar_reference_namespace()["NearestInteger"]
    expr = Round(sp.gamma(x + 1) / sp.E) / sp.gamma(x + 1)
    r = limit(expr, x, sp.oo, return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 1 / sp.E
    assert rounded_tail_certificate(Round(x) - x, x, sp.oo) is None
    assert rounded_tail_certificate(Round(sp.I * x) / x, x, sp.oo) is None
    assert (
        limit(Round(sp.Symbol("t")), sp.Symbol("t"), 0, return_result=True).status
        is LimitStatus.UNKNOWN
    )


def test_finite_product_definitions():
    x, j = sp.symbols("x j")
    expr = sp.Product(x + j, (j, 0, 2)) + sp.ff(x, 3) + sp.rf(x, 3)
    reduced, evidence = normalize_functions(expr, (x,), (0,))
    expected = 2 * x * (x + 1) * (x + 2) + x * (x - 1) * (x - 2)
    assert sp.expand(reduced - expected) == 0
    assert len(evidence) == 1 and evidence[0].value == reduced
    held = sp.Product(x + j, (j, 0, 100))
    assert normalize_functions(held, (x,), (0,)) == (held, ())
    unknown = sp.Function("unbound_argument")(1)
    assert normalize_functions(unknown, (x,), (0,)) == (unknown, ())


def test_undefined_function_stops_general_fallback():
    from asymptotic.instrumentation import symbolic_metrics

    x = sp.Symbol("x")
    expr = sp.Function("missing_definition")(x)
    with symbolic_metrics() as metrics:
        assert bounded_limit(expr, x, 0) is None
    assert metrics.general_limit_calls == 0
    assert metrics.undefined_limit_skips == 1


def test_entire_error_local_rational_cancellation():
    from asymptotic.rational_pullback_germs import (
        finite_error_germ_certificate,
    )

    x = sp.Symbol("x", real=True)
    delta = x / (1 + x * x)
    expr = (sp.erf(delta) - 2 / sp.sqrt(sp.pi) * (delta - delta**3 / 3)) / delta**5
    r = finite_error_germ_certificate(expr, x, sp.S.Zero, sp.S.true)
    assert r is not None and r[1] == 1 / (5 * sp.sqrt(sp.pi))
    assert limit(expr, x, 0) == r[1]
    assert (
        finite_error_germ_certificate(sp.erfi(1 / x), x, sp.S.Zero, sp.S.true) is None
    )
    assert (
        finite_error_germ_certificate(
            (sp.erf(x) - 2 * x / sp.sqrt(sp.pi)) / x**8, x, sp.S.Zero, sp.S.true
        )
        is None
    )


def test_unresolved_infinity_expression_is_not_a_certificate():
    from asymptotic.tail_cancellation_germs import clean

    x = sp.Symbol("x", positive=True)
    expr = -2 * x * sp.hyper(
        (sp.Rational(1, 2), sp.Rational(1, 2)),
        (sp.Rational(3, 2), sp.Rational(3, 2)),
        -x * x,
    ) / sp.sqrt(sp.pi) + sp.log(x) * sp.erf(x)
    assert real_error_tail_certificate(expr, x, sp.oo) is None
    assert (
        clean(
            -2
            * x
            * sp.hyper(
                (sp.Rational(1, 2), sp.Rational(1, 2)),
                (sp.Rational(3, 2), sp.Rational(3, 2)),
                -x * x,
            )
            / sp.sqrt(sp.pi)
            + sp.log(x),
            x,
            sp.oo,
        )
        is None
    )


def test_deeper_integral_tails_use_attained_sequences():
    from asymptotic.rational_pullback_germs import (
        integral_rational_adaptive_tail_certificate,
    )

    x = sp.Symbol("x", positive=True)
    u = x + 3
    expr = u**3 * (sp.Si(u) - sp.pi / 2 + sp.cos(u) / u + sp.sin(u) / u**2)
    r = integral_rational_adaptive_tail_certificate(expr, x, sp.oo)
    assert r is not None and r[0] is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in r[2] if e.substitutions} == {sp.Integer(2), sp.Integer(-2)}
    phase = sp.pi * x * x / 2
    expr = x**5 * (
        sp.fresnelc(x)
        - sp.Rational(1, 2)
        - sp.sin(phase) / (sp.pi * x)
        + sp.cos(phase) / (sp.pi**2 * x**3)
    )
    r = integral_rational_adaptive_tail_certificate(expr, x, sp.oo)
    assert r is not None and r[0] is LimitStatus.DOES_NOT_EXIST
    assert len([e for e in r[2] if e.substitutions]) >= 2
    assert (
        integral_rational_adaptive_tail_certificate(x**10 * sp.Si(x), x, sp.oo) is None
    )


def test_triangle_wave_exact_definition_and_pole_avoidance():
    from asymptotic.function_normalization import TriangleWave
    from asymptotic.rational_pullback_germs import (
        triangle_reciprocal_phase_certificate,
    )

    x = sp.Symbol("x")
    Triangle = scalar_reference_namespace()["TriangleWave"]
    assert [
        TriangleWave(q)
        for q in [0, sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4), 1]
    ] == [0, 1, 0, -1, 0]
    expr = 1 / (1 + Triangle(1 / x))
    r = limit(expr, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {
        e.value
        for e in r.evidence
        if e.substitutions and e.method == "attained_triangle_reciprocal_subsequence"
    } == {1, sp.Rational(2, 3)}
    r = one_sided_limit(expr, x, 0, direction="+", return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    p = sp.Symbol("p", negative=True)
    expr = 1 / (1 + TriangleWave(1 / p))
    r = triangle_reciprocal_phase_certificate(expr, p, 0, sp.S.true, sp.S.true)
    assert r is not None and {e.value for e in r[2]} == {1, 2}
    assert (
        triangle_reciprocal_phase_certificate(
            1 / (1 + TriangleWave(sp.I / x)), x, 0, sp.S.true, sp.S.true
        )
        is None
    )
