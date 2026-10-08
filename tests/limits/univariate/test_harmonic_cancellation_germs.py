import pytest
import sympy as s

from asymptotic import limit
from asymptotic.harmonic_cancellation_germs import (
    circular_segment_endpoint_certificate,
    harmonic_convergent_tail_certificate,
)
from asymptotic.limit_models import LimitStatus

x = s.Symbol("x")


@pytest.mark.parametrize("order", [2, 3, 11, 41])
def test_harmonic_tail_is_bounded_and_returns_exact_zeta(order):
    expr = s.harmonic(x, order)
    result = limit(expr, x, s.oo, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == s.zeta(order)
    partial = s.harmonic(30, order)
    remainder = s.zeta(order) - partial
    bound = s.Rational(31) ** (-order) + s.Rational(31) ** (1 - order) / (order - 1)
    assert s.N(remainder, 70) > 0 and s.N(bound - remainder, 70) > 0


def test_opposite_logarithmic_poles_have_common_scale_on_both_sides():
    expr = -s.log(2 * s.cot(x) + s.I) / s.log(-2 * s.cot(x) - s.I)
    result = limit(expr, x, 0, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == -1
    for side in (-1, 1):
        sample = expr.subs(x, side * s.Rational(1, 10**30))
        assert abs(complex(s.N(sample + 1, 50))) < 0.05


def test_ordered_circular_endpoint_retains_cancellation_and_domain():
    R, w, z = s.symbols("R w z")
    q = R**2 - w * w - z * z
    y = s.sqrt(q) - 1 / x
    expr = y * s.sqrt(q - y * y) / 2 + q * s.atan(y / s.sqrt(q - y * y)) / 2
    bounds = (
        (s.sqrt(R * R - w * w) > 0)
        & (s.sqrt(R * R - w * w) > z)
        & (z + s.sqrt(R * R - w * w) > 0)
    )
    result = limit(expr, x, s.oo, assumptions=bounds, return_result=True)
    assert result.status is LimitStatus.PROVED and result.value == s.pi * q / 4
    assert (
        abs(float(s.N((expr - result.value).subs({R: 3, w: 1, z: 1, x: 10000}))))
        < 0.001
    )
    xp = s.Symbol("xp", positive=True)
    assert (
        circular_segment_endpoint_certificate(
            expr.subs(x, xp), xp, s.oo, s.S.true, s.S.true
        )
        is None
    )


def test_harmonic_bound_declines_nonconvergent_order():
    xp = s.Symbol("xp", positive=True)
    assert (
        harmonic_convergent_tail_certificate(
            s.harmonic(xp, s.Rational(1, 2)), xp, s.oo, s.S.true, s.S.true
        )
        is None
    )
