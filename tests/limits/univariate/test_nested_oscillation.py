"""Attained nested oscillations include every original denominator check."""

import pytest
import sympy as s

from asymptotic import limit
from asymptotic.attained_oscillatory_germs import nested_secant_certificate
from asymptotic.limit_models import LimitStatus


def nested(phase):
    return s.sec(s.tan(s.csc(phase)))


@pytest.mark.parametrize(
    "phase",
    [
        lambda x: x,
        lambda x: x * x + 1,
        lambda x: s.sin(x * x + 1),
        lambda x: 7 * s.sin(x * x + 1),
        lambda x: s.log(x),
        lambda x: s.sin(x),
        lambda x: 2 * x * x + 3 * x + 4,
    ],
)
def test_attained_bounded_sequences(phase):
    x = s.Symbol("x")
    expression = nested(phase(x))
    result = limit(expression, x, s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    witnesses = [item for item in result.evidence if item.substitutions]
    assert {item.value for item in witnesses} == {-1, 1}
    for item in witnesses:
        sequence = item.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        value = sequence.subs(n, 50)
        angle = phase(x).subs(x, value)
        denominators = [
            s.sin(angle),
            s.cos(1 / s.sin(angle)),
            s.cos(s.tan(1 / s.sin(angle))),
        ]
        for denominator in denominators:
            assert abs(complex(denominator.evalf(40))) > 0.1
        actual = expression.subs(x, value).evalf(40)
        assert abs(complex(actual) - int(item.value)) < 1e-25


def test_sine_correlation_sequences():
    x = s.Symbol("x")
    expression = s.sin(x) + nested(x * x + 1)
    result = limit(expression, x, s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    for item in result.evidence:
        sequence = item.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        point = sequence.subs(n, 1000)
        assert abs(float(s.sin(point).evalf(30))) < 0.001
        assert abs(float(expression.subs(x, point).evalf(30)) - int(item.value)) < 0.001


def test_growing_offset_sequences():
    x = s.Symbol("x")
    result = limit(x + nested(x * x + 1), x, s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {item.value for item in result.evidence} == {s.oo, -s.oo}
    for item in result.evidence:
        sequence = item.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        point = sequence.subs(n, 100)
        actual = (x + nested(x * x + 1)).subs(x, point).evalf(50)
        assert float(actual) > 10 if item.value == s.oo else float(actual) < -10


def test_exponential_offset_pole_sequence():
    x = s.Symbol("x")
    expression = s.exp(x) + (2 + s.sin(x)) * nested(s.log(x))
    result = limit(expression, x, s.oo, return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    assert {item.value for item in result.evidence} == {s.oo, -s.oo}
    negative = next(item for item in result.evidence if item.value == -s.oo)
    sequence = negative.substitutions[0][1]
    n = next(iter(sequence.free_symbols))
    # Work at the first index with enough precision to resolve the very
    # small nonzero cosine; the exact construction supplies late-tail proof.
    point = sequence.subs(n, 1)
    value = nested(s.log(x)).subs(x, point).evalf(1200)
    assert value.is_real is True
    assert value < -s.exp(1000)
    assert expression.subs(x, point).evalf(1200) < -s.exp(1000)


def test_continuous_domain_guard():
    n = s.Symbol("n", integer=True)
    x = s.Symbol("x")
    assert nested_secant_certificate(nested(n), n, s.oo, s.S.true) is None
    assert nested_secant_certificate(nested(x), x, s.oo, x > 5) is None
    assert nested_secant_certificate(nested(s.sin(1 / x)), x, s.oo, s.S.true) is None


def test_uniform_secant_quotient():
    x = s.Symbol("x")
    assert limit((x + s.sec(x)) / (x * x * s.sec(x) + 1), x, s.oo) == 0
    cosine = s.Symbol("c", real=True)
    original = (x + 1 / cosine) / (x * x / cosine + 1)
    cleared = (x * cosine + 1) / (x * x + cosine)
    assert s.cancel(original - cleared) == 0


def test_compact_phase_budget():
    x = s.Symbol("x")
    assert (
        nested_secant_certificate(nested((x + 1) ** 1000000), x, s.oo, s.S.true) is None
    )
