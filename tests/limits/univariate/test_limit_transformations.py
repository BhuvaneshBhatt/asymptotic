"""Limit relations with explicit chart, direction and domain hypotheses."""

import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from asymptotic import limit, one_sided_limit
from asymptotic.limit_models import LimitStatus

CHECKS = settings(max_examples=8, deadline=None, derandomize=True)
SMALL = st.integers(-3, 3)
POSITIVE = st.integers(1, 3)
x, t = sp.symbols("x t", real=True)


@pytest.mark.parametrize(
    "expression,expected",
    [
        (sp.sin(x) / x, sp.S.One),
        (sp.log(1 + x) / x, sp.S.One),
        ((1 - sp.cos(x)) / x**2, sp.Rational(1, 2)),
        ((sp.sqrt(1 + x) - 1) / x, sp.Rational(1, 2)),
    ],
)
@CHECKS
@given(scale=POSITIVE, curvature=st.integers(0, 3))
def test_nonlinear_chart(expression, expected, scale, curvature):
    # A nonzero derivative gives a local real inverse and preserves both sides.
    chart = scale * t * (1 + curvature * t**2)
    baseline = limit(expression, x, 0, return_result=True)
    changed = limit(expression.subs(x, chart), t, 0, return_result=True)
    assert baseline.status is changed.status is LimitStatus.PROVED
    assert baseline.value == changed.value == expected


@CHECKS
@given(multiplier=SMALL, offset=SMALL, perturbation=SMALL)
def test_output_and_small_error(multiplier, offset, perturbation):
    expression = sp.sin(x) / x
    changed = multiplier * expression + offset + perturbation * x * sp.sin(1 / x)
    baseline = limit(expression, x, 0, return_result=True)
    result = limit(changed, x, 0, return_result=True)
    assert baseline.status is result.status is LimitStatus.PROVED
    assert baseline.value == 1
    assert result.value == multiplier * baseline.value + offset


@CHECKS
@given(coefficient=SMALL)
def test_reciprocal_chart(coefficient):
    expression = (2 * x**2 + coefficient * x + 1) / (x**2 + 1)
    tail = limit(expression, x, sp.oo, return_result=True)
    local = one_sided_limit(
        expression.subs(x, 1 / t), t, 0, direction="+", return_result=True
    )
    assert tail.status is local.status is LimitStatus.PROVED
    assert tail.value == local.value == 2


@CHECKS
@given(coefficient=SMALL)
def test_nonzero_reciprocal(coefficient):
    expression = (2 + coefficient * x) / (1 - x)
    baseline = limit(expression, x, 0, return_result=True)
    reciprocal = limit(1 / expression, x, 0, return_result=True)
    assert baseline.status is reciprocal.status is LimitStatus.PROVED
    assert baseline.value == 2
    assert reciprocal.value == 1 / baseline.value


@pytest.mark.parametrize(
    "expression,right,left",
    [
        (sp.Abs(x) / x, sp.S.One, -sp.S.One),
        (1 / x, sp.oo, -sp.oo),
    ],
)
@CHECKS
@given(scale=POSITIVE)
def test_reflection(expression, right, left, scale):
    reflected = expression.subs(x, -scale * t)
    assert one_sided_limit(expression, x, 0, direction="+") == right
    assert one_sided_limit(expression, x, 0, direction="-") == left
    assert one_sided_limit(reflected, t, 0, direction="+") == left
    assert one_sided_limit(reflected, t, 0, direction="-") == right


@CHECKS
@given(
    scale=st.sampled_from([1, 3]),
    shift=st.sampled_from([0, 1, sp.pi / 3]),
    multiplier=st.sampled_from([-2, -1, 1, 2]),
    offset=SMALL,
)
def test_attained_nonexistence(scale, shift, multiplier, offset):
    original = sp.Mod(x, 2) + sp.cos(sp.sqrt(2) * x)
    chart = scale * t + shift
    expression = multiplier * original.subs(x, chart) + offset
    baseline = limit(original, x, sp.oo, return_result=True)
    changed = limit(expression, t, sp.oo, return_result=True)
    assert baseline.status is changed.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in baseline.evidence} == {-1, 2}
    assert {e.value for e in changed.evidence} == {
        offset - multiplier,
        offset + 2 * multiplier,
    }
    assert all(e.substitutions for e in changed.evidence)


def test_one_side_image():
    # Squaring omits the negative side, so a local inverse hypothesis is essential.
    expression = sp.Abs(x) / x
    original = limit(expression, x, 0, return_result=True)
    restricted = limit(expression.subs(x, t**2), t, 0, return_result=True)
    assert original.status is LimitStatus.DOES_NOT_EXIST
    assert restricted.status is LimitStatus.PROVED and restricted.value == 1


def test_noninjective_output():
    expression = sp.Abs(x) / x
    original = limit(expression, x, 0, return_result=True)
    squared = limit(expression**2, x, 0, return_result=True)
    assert original.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in original.evidence} == {-1, 1}
    assert squared.status is LimitStatus.PROVED and squared.value == 1
