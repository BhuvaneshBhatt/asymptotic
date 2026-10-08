import pytest
import sympy as s

from asymptotic import limit, one_sided_limit
from asymptotic.algebraic_limit_germs import (
    assumed_parameter_power_certificate,
    complex_near_one_power_certificate,
    inverse_trig_cut_certificate,
    parity_subsequence_certificate,
    rational_exponential_dominance_certificate,
    reciprocal_sine_amplitude_certificate,
)
from asymptotic.limit_models import LimitStatus
from asymptotic.rational_approximation import rational_approximation
from asymptotic.reference_normalization import scalar_reference_namespace


def test_attained_sine_amplitude_sequences_and_denominator_avoidance():
    x = s.Symbol("x", real=True)
    e = s.sqrt(x) * s.sin(1 / x) / s.sin(x)
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {v.value for v in r.evidence} == {s.S.Zero, s.oo}
    for v in r.evidence:
        seq = v.substitutions[0][1]
        n = next(iter(seq.free_symbols))
        z = seq.subs(n, 1000)
        assert z.is_positive is True and s.sin(z).is_zero is False
        assert s.simplify(s.sin(1 / seq)) in (0, 1)
    assert reciprocal_sine_amplitude_certificate(e, x, 0, s.Eq(x, 0), s.true) is None


@pytest.mark.parametrize(
    "make,values",
    [
        (
            lambda n: ((-1) ** n * (n + 1) + n) / (s.sqrt(n) + n + 1),
            {s.S.Zero, s.Integer(2)},
        ),
        (lambda n: n * s.log(1 + (-1) ** n / n), {s.S.One, -s.S.One}),
    ],
)
def test_parity_witnesses_are_attained_on_real_and_integer_domains(make, values):
    for n in (s.Symbol("n", real=True), s.Symbol("n", positive=True, integer=True)):
        r = limit(make(n), n, s.oo, return_result=True)
        assert (
            r.status is LimitStatus.DOES_NOT_EXIST
            and {v.value for v in r.evidence} == values
        )
        for ev in r.evidence:
            seq = ev.substitutions[0][1]
            k = next(iter(seq.free_symbols))
            assert s.simplify((-1) ** seq) in (1, -1)
            assert (
                abs(
                    complex((make(n).subs(n, seq.subs(k, 100000)) - ev.value).evalf(25))
                )
                < 0.01
            )


def test_parity_declines_uncontrolled_log_remainders_and_domains():
    x = s.Symbol("x", positive=True)
    assert (
        parity_subsequence_certificate(
            x * x * s.log(1 + (-1) ** x / x), x, s.oo, s.true, s.true
        )
        is None
    )
    assert parity_subsequence_certificate((-1) ** x, x, s.oo, x > 1, s.true) is None


@pytest.mark.parametrize("power", [1, 2, 4])
def test_exponential_decay_dominates_algebraic_poles(power):
    x = s.Symbol("x", real=True)
    r = limit(s.exp(-3 / x) / x**power, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert r.evidence[0].value == 0
    assert r.evidence[1].value == (-s.oo if power % 2 else s.oo)
    assert one_sided_limit(s.exp(-3 / x) / x**power, x, 0, direction="+") == 0


@pytest.mark.parametrize("w", [s.S.Zero, s.Integer(2), s.I, s.Rational(2, 3) + s.I])
def test_complex_near_one_power_cancellation_with_independent_samples(w):
    n = s.Symbol("n", positive=True)
    e = s.exp(-s.I * s.sqrt(n) * w) / (1 - s.I * w / s.sqrt(n)) ** n
    r = limit(e, n, s.oo, return_result=True)
    assert (
        r.status is LimitStatus.PROVED and s.simplify(r.value - s.exp(-w * w / 2)) == 0
    )
    assert (
        abs(complex((e.subs(n, s.Float(1000000, 40)) - s.exp(-w * w / 2)).evalf(35)))
        < 0.01
    )


def test_symbolic_complex_near_one_and_remainder_guard():
    n = s.Symbol("n", positive=True)
    w = s.Symbol("w")
    e = s.exp(-s.I * s.sqrt(n) * w) / (1 - s.I * w / s.sqrt(n)) ** n
    assert limit(e, n, s.oo) == s.exp(-w * w / 2)
    assert (
        complex_near_one_power_certificate((-1 + 1 / n) ** n, n, s.oo, s.true, s.true)
        is None
    )
    assert (
        complex_near_one_power_certificate(
            (1 + 1 / n) ** (n**9), n, s.oo, s.true, s.true
        )
        is None
    )


def test_identity_power_on_principal_negative_base():
    x = s.Symbol("x")
    e = (x + x ** s.exp(x) * s.exp(1 / x)) / x
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST and {v.value for v in r.evidence} == {
        s.S.One,
        s.oo,
    }


def test_principal_atan_cut_values_are_independently_attained():
    x = s.Symbol("x", real=True)
    e = s.atan(2 * s.sqrt(-s.exp(s.I * x)))
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    for ev in r.evidence:
        seq = ev.substitutions[0][1]
        n = next(iter(seq.free_symbols))
        assert (
            abs(complex((e.subs(x, seq.subs(n, 10000)) - ev.value).evalf(30))) < 0.001
        )
    assert inverse_trig_cut_certificate(e, x, 0, s.Eq(x, 0), s.true) is None


def test_real_cube_root_pullback_and_secant_radical_tail():
    x = s.Symbol("x")
    ns = scalar_reference_namespace()
    assert (
        one_sided_limit(
            s.sympify("tan(real_root(x, 3))", locals=ns), x, s.pi**3 / 8, direction="-"
        )
        == s.oo
    )
    n = s.Symbol("n")
    assert limit(s.sec(s.pi * (n - s.sqrt(n * (n - 1)))), n, s.oo) == -s.oo


def test_tangent_exponential_boundary():
    x = s.Symbol("x", real=True)
    e = s.exp(-s.tan(x)) * s.sec(x) ** 2
    r = limit(e, x, s.pi / 2, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST and {v.value for v in r.evidence} == {
        s.S.Zero,
        s.oo,
    }


def test_explicit_parameter_power_strata_and_missing_assumptions():
    x = s.Symbol("x")
    n = s.Symbol("n")
    p = s.Symbol("p")
    assert limit(x**p * s.log(x), x, 0, assumptions=s.re(p) > 0) == 0
    assert limit(s.exp(-x * x**n), x, s.oo, assumptions=n < -1) == 1
    assert limit(s.exp(-x * x**n), x, s.oo, assumptions=n > -1) == 0
    assert limit(s.exp(-x * x ** (-1)), x, s.oo) == s.exp(-1)
    assert limit(x ** (-n) * s.exp(-1 / x), x, s.oo, assumptions=n < 0) == s.oo
    assert (
        assumed_parameter_power_certificate(s.exp(-x * x**n), x, s.oo, s.true, s.true)
        is None
    )


@pytest.mark.parametrize(
    "condition,values",
    [
        (lambda n: s.And(s.Eq(s.re(n), -1), s.Ne(s.im(n), 0)), {s.exp(-1), s.E}),
        (lambda n: s.And(s.re(n) > -1, s.Ne(s.im(n), 0)), {s.S.Zero, s.oo}),
    ],
)
def test_nonreal_parameter_power_strata_use_attained_phases(condition, values):
    x = s.Symbol("x")
    n = s.Symbol("n")
    r = limit(s.exp(-x * x**n), x, s.oo, assumptions=condition(n), return_result=True)
    assert (
        r.status is LimitStatus.DOES_NOT_EXIST
        and {v.value for v in r.evidence} == values
    )
    assert all(v.substitutions for v in r.evidence)


def test_literal_decimal_offset_is_a_pole_not_a_derivative():
    x = s.Symbol("h")
    e = (s.sin(x + s.Float("2.0")) - s.Float("0.909297426825682")) / x
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST and {v.value for v in r.evidence} == {
        s.oo,
        -s.oo,
    }


def test_rationalize_interval_definition_and_known_source_target():
    z = s.N(s.sqrt(s.pi), 100)
    tol = s.Rational(1, 10**100)
    expected = s.Rational(
        163058077416802338133084818274433213090751841141227,
        91995668791883515387450257467653926464520949863410,
    )
    assert rational_approximation(z, tol) == expected
    assert abs(expected - s.Rational(z)) <= tol
    for q in range(1, 15):
        result = rational_approximation(s.Rational(7, 13), s.Rational(1, q))
        assert abs(result - s.Rational(7, 13)) <= s.Rational(1, q)
        assert all(
            not any(
                abs(s.Rational(p, d) - s.Rational(7, 13)) <= s.Rational(1, q)
                for p in range(-2 * d, 3 * d)
            )
            for d in range(1, int(s.denom(result)))
        )
    with pytest.raises(ValueError):
        rational_approximation(s.Rational(1, 2), -s.S.One)
    assert (
        rational_approximation(s.sqrt(2), tol).func.__name__ == "rational_approximation"
    )


def test_symbolic_atan_parameter_boundary_includes_imaginary_axis():
    eps = s.Symbol("eps")
    p = s.Symbol("p")
    e = s.atan(p / eps)
    r = one_sided_limit(
        e, eps, 0, direction="+", assumptions=s.Ne(p, 0), return_result=True
    )
    assert r.status is LimitStatus.PROVED and r.value == s.pi * s.sqrt(p * p) / (2 * p)
    for value in (s.I, -s.I, 2 + s.I, -2 + s.I):
        expected = r.value.subs(p, value)
        actual = s.atan(value * s.Integer(100000)).evalf(35)
        assert abs(complex(actual - expected)) < 0.001
    two = limit(e, eps, 0, assumptions=s.Ne(p, 0), return_result=True)
    assert two.status is LimitStatus.DOES_NOT_EXIST
    assert all(v.substitutions for v in two.evidence)


def test_exponential_dominance_declines_potentially_zero_denominator():
    x = s.Symbol("x", real=True)
    a, b = s.symbols("a b", real=True)
    assert (
        rational_exponential_dominance_certificate(
            s.exp(-1 / x) / (a * x + b), x, 0, s.true, s.true
        )
        is None
    )


def test_large_denominator_fractional_powers_meet_budget():
    import json
    import subprocess
    import sys
    from pathlib import Path

    u = s.Symbol("u", positive=True)
    e = (
        (u ** s.Rational(38966551, 183370156) - 1)
        * (u ** s.Rational(34592177, 122508763) - 1)
        / (u - 1)
    )
    assert assumed_parameter_power_certificate(e, u, s.S.Zero, s.true, s.true) is None
    root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_univariate_corpus.py",
            "--worker",
            "3504",
            "--budget",
            "5",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=12,
    )
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result["classification"] == "pass" and result["actual"] == "-1"
    assert result["soft_budget_fired"] is False


@pytest.mark.parametrize(
    "variable,point",
    [
        (s.Symbol("nonreal_coordinate", real=False), s.oo),
        (s.Symbol("negative_coordinate", negative=True), s.oo),
        (s.Symbol("positive_coordinate", positive=True), -s.oo),
    ],
)
def test_real_infinity_witnesses_respect_original_variable_domain(variable, point):
    result = limit((-1) ** variable, variable, point, return_result=True)
    assert result.status is LimitStatus.UNKNOWN
    assert result.evidence[0].method == "incompatible_real_infinity_chart"
    assert not any(ev.substitutions for ev in result.evidence)


def test_explicit_nonreal_complex_ray_remains_supported():
    from asymptotic import complex_ray_limit

    z = s.Symbol("z", real=False)
    result = complex_ray_limit(z * z, z, 0, ray=s.I, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == 0


def test_regular_principal_inverse_trig_root_comparison():
    from asymptotic.reference_normalization import scalar_reference_equal

    x = s.Symbol("x", real=True)
    a = (1 - s.sqrt(3) + (-1) ** s.Rational(1, 3)) / (
        1 + s.sqrt(3) + (-1) ** s.Rational(1, 3)
    )
    b = (3 - s.sqrt(3) * (2 - s.I)) / (3 + s.sqrt(3) * (2 + s.I))
    assert scalar_reference_equal(s.acos(a), s.acos(b))
    e = s.acos(
        (1 - s.sqrt(3))
        * (x + 1) ** 2
        / ((1 + s.sqrt(3)) * (x + 1) ** 2 + (-1) ** s.Rational(1, 3))
        + (-1) ** s.Rational(1, 3)
        / ((1 + s.sqrt(3)) * (x + 1) ** 2 + (-1) ** s.Rational(1, 3))
    )
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.PROVED and scalar_reference_equal(r.value, s.acos(b))


def test_coupled_modulo_limits():
    x = s.Symbol("x", positive=True)
    y = x**3 - x
    e = (
        (x + 1)
        / (x - 1)
        * (1 + s.cos(s.sqrt(2) * y) / 2)
        * (s.Rational(1, 2) + s.Mod(y, 2) / 2)
        * s.cos(s.sin(y) ** 2)
    )
    r = limit(e, x, s.oo, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {v.value for v in r.evidence} == {s.Rational(1, 4), s.Rational(3, 2)}
    assert all(
        v.method == "attained_joint_modular_phase" and v.substitutions
        for v in r.evidence
    )
    assert limit(s.log(x) + e, x, s.oo) == s.oo
    assert limit(e / x, x, s.oo) == 0
    # On x>3 all factors are positive, and e<2*(3/2)*(3/2).
    for n in (4, 9, 100):
        assert 0 < float(e.subs(x, n).evalf()) < 4.5
