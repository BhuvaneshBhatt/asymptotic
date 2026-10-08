import pytest
import sympy as s

from asymptotic import limit, one_sided_limit
from asymptotic.exponential_integral_germs import (
    bounded_ei_composition_certificate,
)
from asymptotic.lambert_small_germs import reciprocal_lambert_germ_certificate
from asymptotic.limit_models import LimitStatus


def test_ei_attained_sequences_and_uniform_bounds():
    x = s.Symbol("x", real=True)
    e = s.Ei(2 + s.sin(1 / x))
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert set(v.value for v in r.evidence) == {s.Ei(1), s.Ei(3)}
    for v in r.evidence:
        seq = v.substitutions[0][1]
        assert s.simplify(e.subs(x, seq)) == v.value
    assert limit(x * e, x, 0) == 0
    assert one_sided_limit(e / x, x, 0, direction="+") == s.oo
    assert one_sided_limit(e / x, x, 0, direction="-") == -s.oo
    assert (
        bounded_ei_composition_certificate(s.Ei(s.sin(1 / x)), x, 0, s.true, s.true)
        is None
    )


@pytest.mark.parametrize("rate", [1, 3])
@pytest.mark.parametrize("kind", [1, 2, 3])
def test_expint_retains_cancelled_constants_and_side_values(rate, kind):
    x = s.Symbol("x", real=True)
    z = s.exp(-rate / x)
    e = (
        s.expint(1, z) - rate / x
        if kind == 1
        else (s.expint(2, z) - 1) / z + (rate / x if kind == 3 else 0)
    )
    right, left = {
        1: (-s.EulerGamma, s.oo),
        2: (-s.oo, s.S.Zero),
        3: (s.EulerGamma - 1, -s.oo),
    }[kind]
    r = limit(e, x, 0, return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST and {v.value for v in r.evidence} == {
        right,
        left,
    }
    assert one_sided_limit(e, x, 0, direction="+") == right
    assert one_sided_limit(e, x, 0, direction="-") == left
    if right.is_finite:
        assert abs(complex((e.subs(x, s.Rational(rate, 20)) - right).evalf(40))) < 1e-7


@pytest.mark.parametrize("ray", [s.I, 1 - s.I, -1 - s.I])
@pytest.mark.parametrize(
    "make,expected",
    [
        (lambda w: (s.sinh(1 / w) - 1 / w) * w**2, 0),
        (lambda w: (s.cosh(1 / w) - 1 - 1 / (2 * w**2)) * w**3, 0),
        (lambda w: (s.exp(1 / w) - 1) * w, 1),
        (
            lambda w: (s.log(1 + 1 / w) - 1 / w + 1 / (2 * w**2)) * w**3,
            s.Rational(1, 3),
        ),
        (lambda w: (s.sqrt(1 + 1 / w) - 1) * w, s.Rational(1, 2)),
    ],
)
def test_lambert_reciprocal_germs_and_independent_taylor_coefficients(
    ray, make, expected
):
    x = s.Symbol("x")
    p = s.Function("DirectionalInfinity")(ray)
    w = s.LambertW(x)
    r = limit(make(w), x, p, return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == expected
    u = s.Symbol("u")
    assert s.series(make(1 / u), u, 0, 1).removeO() == expected
    t = s.Symbol("t", positive=True)
    assert limit(make(s.LambertW(ray * t)), t, s.oo) == expected


def test_lambert_declines_uncertified_branch_direction_and_pole():
    x = s.Symbol("x", positive=True)
    w = s.LambertW(x)
    assert reciprocal_lambert_germ_certificate(w, x, s.oo, s.true, s.true) is None
    assert (
        reciprocal_lambert_germ_certificate(1 / s.LambertW(-x), x, s.oo, s.true, s.true)
        is None
    )
    assert (
        reciprocal_lambert_germ_certificate(
            1 / s.LambertW(x, 1), x, s.oo, s.true, s.true
        )
        is None
    )
    p = s.Function("DirectionalInfinity")(s.I)
    assert reciprocal_lambert_germ_certificate(1 / w, x, p, s.true, s.true) is None
    assert (
        reciprocal_lambert_germ_certificate(
            s.sin(1 / w) * w**8, x, s.oo, s.true, s.true
        )
        is None
    )


def test_lambert_dispatch_accepts_native_integer_local_targets():
    x = s.Symbol("x", positive=True)
    assert (
        reciprocal_lambert_germ_certificate(s.sin(x) / x, x, 0, s.true, s.true) is None
    )
    assert one_sided_limit(s.sin(x) / x, x, 0, direction="+") == 1
