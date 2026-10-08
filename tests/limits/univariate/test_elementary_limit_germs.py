import sympy as sp

from asymptotic import limit
from asymptotic.elementary_limit_germs import (
    exponential_phase_subsequences,
    gamma_ratio_cancellation_certificate,
    near_one_cancellation_certificate,
    nonpositive_radical_subsequences,
    unit_phase_envelope_certificate,
    zeta_laurent_cancellation_certificate,
)
from asymptotic.limit_models import LimitStatus


def test_unit_phase_bound_needs_real_phase_and_vanishing_coefficients():
    p, a = sp.symbols("p a")
    e = sp.exp(sp.I * p * a) / p
    r = limit(e, p, sp.oo, assumptions=sp.Eq(sp.im(a), 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0
    x = sp.Symbol("x", positive=True)
    assert (
        unit_phase_envelope_certificate(
            sp.exp(sp.I * x) / x, x, sp.oo, sp.S.true, sp.S.true
        )[1]
        == 0
    )
    assert (
        unit_phase_envelope_certificate(
            sp.exp(sp.I * x), x, sp.oo, sp.S.true, sp.S.true
        )
        is None
    )
    assert (
        unit_phase_envelope_certificate(
            sp.exp((1 + sp.I) * x) / x, x, sp.oo, sp.S.true, sp.S.true
        )
        is None
    )


def test_exponential_witnesses_are_attained_with_original_parameters():
    x, a = sp.symbols("x a")
    e = sp.exp(sp.I * a * x)
    r = limit(e, x, sp.oo, assumptions=a > 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST and len(r.evidence) == 2
    assert {v.value for v in r.evidence} == {sp.S.One, -sp.S.One}
    for ev in r.evidence:
        variable, sequence = ev.substitutions[0]
        assert variable == x and a in sequence.free_symbols
        assert sp.simplify(e.subs(x, sequence) - ev.value) == 0


def test_phase_denominators_and_logarithmic_phases_are_certified():
    x = sp.Symbol("x", positive=True)
    r = limit(1 / (2 + sp.exp(sp.I * x)), x, sp.oo, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {ev.value for ev in r.evidence} == {sp.S.One, sp.Rational(1, 3)}
    for ev in r.evidence:
        assert sp.simplify(2 + sp.exp(sp.I * ev.substitutions[0][1])).is_zero is False
    for e in (x**sp.I, sp.I * x ** (-sp.I)):
        r = limit(e, x, 0, return_result=True)
        assert r.status is LimitStatus.DOES_NOT_EXIST
        for ev in r.evidence:
            assert sp.simplify(e.subs(x, ev.substitutions[0][1]) - ev.value) == 0
    assert (
        exponential_phase_subsequences(sp.exp(sp.I * x), x, sp.oo, sp.S.false) is None
    )


def test_radical_opposing_sequences_and_one_sided_guard():
    x, a = sp.symbols("x a")
    e = (a - sp.sqrt(a * a + x * x)) / x
    for assumption in (sp.Eq(a, 0), a < 0):
        r = limit(e, x, 0, assumptions=assumption, return_result=True)
        assert r.status is LimitStatus.DOES_NOT_EXIST and len(r.evidence) == 2
        assert {ev.substitutions[0][0] for ev in r.evidence} == {x}
    u = sp.Symbol("u", positive=True)
    assert (
        nonpositive_radical_subsequences(-sp.sqrt(u * u) / u, u, 0, sp.S.true) is None
    )


def test_near_one_power_keeps_amplified_error():
    x = sp.Symbol("x", positive=True)
    p = (1 + 1 / x) ** x
    assert near_one_cancellation_certificate(x * (p - sp.E), x, sp.oo)[1] == -sp.E / 2
    assert (
        near_one_cancellation_certificate(
            x * x * (p - sp.E + sp.E / (2 * x)), x, sp.oo
        )[1]
        == 11 * sp.E / 24
    )
    assert near_one_cancellation_certificate(x**3 * (p - sp.E), x, sp.oo) is None
    assert near_one_cancellation_certificate((-1 - 1 / x) ** x, x, sp.oo) is None
    assert (
        near_one_cancellation_certificate(((x - 1) / (x + 3)) ** (x + 2), x, sp.oo)
        is None
    )
    r = limit(((x - 1) / (x + 3)) ** (x + 2), x, sp.oo, return_result=True)
    assert (
        r.value == sp.exp(-4) and r.evidence[0].method == "near_one_power_log_remainder"
    )


def test_gamma_fixed_shift_bernoulli_terms_survive_subtraction():
    x = sp.Symbol("x", positive=True)
    r = sp.gamma(x + sp.Rational(1, 2)) / sp.gamma(x) / sp.sqrt(x)
    assert gamma_ratio_cancellation_certificate(
        x * x * (r - 1 + 1 / (8 * x)), x, sp.oo
    )[1] == sp.Rational(1, 128)
    assert gamma_ratio_cancellation_certificate(
        x**3 * (r - 1 + 1 / (8 * x) - 1 / (128 * x * x)), x, sp.oo
    )[1] == sp.Rational(5, 1024)
    assert gamma_ratio_cancellation_certificate(x**4 * (r - 1), x, sp.oo) is None


def test_zeta_laurent_remainder_and_complex_dirichlet_shifts():
    x = sp.Symbol("x")
    e = (sp.zeta(1 + x) - 1 / x - sp.EulerGamma) / x
    assert zeta_laurent_cancellation_certificate(e, x, 0, sp.S.true)[
        1
    ] == -sp.stieltjes(1)
    assert zeta_laurent_cancellation_certificate(e / x**3, x, 0, sp.S.true) is None
    n = sp.Symbol("n", positive=True)
    a = sp.Symbol("a")
    from asymptotic.tail_cancellation_germs import (
        dirichlet_cancellation_certificate,
    )

    assert (
        dirichlet_cancellation_certificate(
            3**n * (sp.zeta(n - a, 3) - sp.zeta(n, 3)) + 1, n, sp.oo
        )[1]
        == 3**a
    )


def test_reference_0262_correction_from_independent_dirichlet_terms():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    row = next(
        r
        for r in json.loads(
            (root / "tests/data/univariate_limit_reference_cases.json").read_text()
        )
        if r["id"].endswith("0262")
    )
    assert row["expected"] == "4**x"
    n = sp.Symbol("n", integer=True, positive=True)
    x = sp.Symbol("x")
    tail = sp.Symbol("tail")
    expression = sp.sympify(row["expression"], locals={"n": n, "x": x})
    truncation = 1 + 2 ** (x - n) + 3 ** (x - n) + 4 ** (x - n) + tail
    derived = expression.xreplace({sp.zeta(n - x): truncation})
    assert sp.simplify(sp.expand_power_exp(derived) - (4**x + 4**n * tail)) == 0
    bound = 5 ** sp.re(x) * (sp.Rational(4, 5)) ** n * (1 + 5 / (n - sp.re(x) - 1))
    assert sp.limit(bound, n, sp.oo) == 0
