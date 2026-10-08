import pytest
import sympy as s

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus
from asymptotic.local_nonexistence_families import (
    attained_tangent_pole_certificate,
    local_rational_pole_certificate,
    reciprocal_exponential_sides_certificate,
)


@pytest.mark.parametrize(
    "expression,point",
    [
        ("(x**2+x-2)/(x-1)**2", 1),
        ("(x-2)**-3", 2),
        ("1/(x**3-6*x**2+11*x-6)", 2),
        ("(x**2-5*x+10)/(x*x-25)", 5),
        ("(x*x-2*x)/(x*x-4*x+4)", 2),
        ("x**-3", 0),
    ],
)
def test_attained_rational_pole_sides(expression, point):
    x = s.Symbol("x", real=True)
    e = s.sympify(expression, locals={"x": x})
    r = limit(e, x, point, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {v.value for v in r.evidence} == {s.oo, -s.oo}
    assert all(v.substitutions[0][0] == x for v in r.evidence)


@pytest.mark.parametrize(
    "expression,values",
    [
        ("1/(2**(1/x)+3)", {s.S.Zero, s.Rational(1, 3)}),
        ("(2**(1/x)+1)/(2**(1/x)+3)", {s.S.One, s.Rational(1, 3)}),
        ("2/(1+exp(-1/x))", {s.S.Zero, s.Integer(2)}),
        ("(1+10**(-1/x))/(2-10**(-1/x))", {s.Rational(1, 2), -s.S.One}),
    ],
)
def test_attained_reciprocal_exponential_sides(expression, values):
    x = s.Symbol("x", real=True)
    e = s.sympify(expression, locals={"x": x})
    r = limit(e, x, 0, return_result=True)
    assert (
        r.status is LimitStatus.DOES_NOT_EXIST
        and {v.value for v in r.evidence} == values
    )
    assert all(v.substitutions for v in r.evidence)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_tangent_pole_sequences_retain_nonzero_limit_and_avoid_poles(k):
    x = s.Symbol("x", real=True)
    e = x**k * s.tan(2 / x)
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {v.value for v in r.evidence} == {s.S.Zero, (2 / s.pi) ** k}
    # Independent numerical check of the nonzero attained branch.
    sequence = r.evidence[1].substitutions[0][1]
    n = next(iter(sequence.free_symbols))
    z = sequence.subs(n, 100)
    assert abs(float((z**k * s.tan(2 / z)).evalf(35)) - (2 / float(s.pi)) ** k) < 0.02
    assert s.cos(2 / z).evalf(35) != 0


def test_domains_coefficients_and_unsupported_cases_decline():
    x = s.Symbol("x", real=True)
    a = s.Symbol("a")
    assert local_rational_pole_certificate(a / x, x, 0, s.true, s.true) is None
    assert local_rational_pole_certificate(s.I / x, x, 0, s.true, s.true) is None
    assert local_rational_pole_certificate(1 / x**2, x, 0, s.true, s.true) is None
    assert (
        reciprocal_exponential_sides_certificate(s.exp(1 / x) / x, x, 0, s.true, s.true)
        is None
    )
    assert (
        attained_tangent_pole_certificate(s.tan(a / x) * x, x, 0, s.true, s.true)
        is None
    )
    assert (
        attained_tangent_pole_certificate(s.tan(1 / x) * x, x, 0, s.Eq(x, 0), s.true)
        is None
    )
    assert (
        attained_tangent_pole_certificate(s.tan(1 / x) * x, x, 0, s.true, x > 1) is None
    )


def test_one_sided_results_and_negative_domain_tangent_sequences():
    x = s.Symbol("x", real=True)
    assert one_sided_limit(1 / x, x, 0, direction="+") == s.oo
    assert one_sided_limit(1 / x, x, 0, direction="-") == -s.oo
    assert one_sided_limit(1 / (2 ** (1 / x) + 3), x, 0, direction="+") == 0
    y = s.Symbol("y", negative=True)
    r = limit(y * s.tan(1 / y), y, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    for ev in r.evidence:
        sequence = ev.substitutions[0][1]
        n = next(iter(sequence.free_symbols))
        assert sequence.subs(n, 1).is_negative is True
        assert sequence.subs(n, 1000).is_negative is True
        assert s.limit(sequence, n, s.oo) == 0
    # pi*n+pi/2-1/n**k >= pi+pi/2-1 > 0 for integer n>=1.
    assert (3 * s.pi / 2 - 1).is_positive is True
