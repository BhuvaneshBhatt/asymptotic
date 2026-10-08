"""Native analytic series relations compared modulo a fixed power."""

import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from asymptotic import limit, multiseries

CHECKS = settings(max_examples=8, deadline=None, derandomize=True)
SMALL = st.integers(-3, 3)
h, t = sp.symbols("h t", positive=True)


def _jet(expression, variable, order):
    expansion = multiseries(
        expression,
        variable,
        scale=[variable],
        point=0,
        terms=order,
        allow_series_fallback=False,
        return_result=True,
    )
    # Term budgets count nonzero terms, whereas this contract fixes the exponent.
    terms = expansion.terms(order)
    assert all(term.exponent.is_Integer and term.exponent >= 0 for term in terms)
    return sp.expand(
        sum(
            term.coefficient * variable**term.exponent
            for term in terms
            if term.exponent < order
        )
    )


def _below(expression, variable, order):
    polynomial = sp.Poly(sp.expand(expression), variable)
    return sum(polynomial.nth(k) * variable**k for k in range(order))


@CHECKS
@given(order=st.integers(2, 6))
def test_exponential_anchor(order):
    expected = sum(h**k / sp.factorial(k) for k in range(order))
    assert _jet(sp.exp(h), h, order) == expected


@CHECKS
@given(a=SMALL, b=SMALL, order=st.integers(3, 6))
def test_addition_and_cancellation(a, b, order):
    f = sp.exp(h)
    g = -1 - h - h**2 / 2 + a * h**3 + b * h**4
    assert _jet(f + g, h, order) == sp.expand(_jet(f, h, order) + _jet(g, h, order))
    if order >= 4:
        assert _jet(f + g, h, order).coeff(h, 3) == a + sp.Rational(1, 6)


@CHECKS
@given(a=SMALL, b=SMALL, order=st.integers(2, 6))
def test_product(a, b, order):
    f, g = sp.exp(h) + a * h, sp.log(1 + h) + b * h**2
    direct = _jet(f * g, h, order)
    assembled = _below(_jet(f, h, order) * _jet(g, h, order), h, order)
    assert sp.expand(direct - assembled) == 0


@CHECKS
@given(a=SMALL, order=st.integers(2, 6))
def test_reciprocal(a, order):
    f = sp.exp(h) + 1 + a * h
    product = _below(_jet(f, h, order) * _jet(1 / f, h, order), h, order)
    assert sp.expand(product - 1) == 0


@pytest.mark.parametrize(
    "outer", [sp.exp(h), sp.log(1 + h), (1 + h) ** sp.Rational(1, 2)]
)
@CHECKS
@given(a=SMALL, order=st.integers(2, 6))
def test_composition(outer, a, order):
    chart = t + a * t**2
    direct = _jet(outer.subs(h, chart), t, order)
    composed = _below(_jet(outer, h, order).subs(h, chart), t, order)
    assert sp.expand(direct - composed) == 0


@pytest.mark.parametrize(
    "expression", [sp.exp(h), sp.log(1 + h), (1 + h) ** sp.Rational(1, 2)]
)
@CHECKS
@given(order=st.integers(3, 6))
def test_derivative_order(expression, order):
    direct = _jet(sp.diff(expression, h), h, order - 1)
    differentiated = sp.diff(_jet(expression, h, order), h)
    assert sp.expand(direct - differentiated) == 0


@CHECKS
@given(order=st.integers(2, 5), extra=st.integers(1, 3))
def test_exponent_prefix(order, extra):
    expression = sp.exp(h) - 1 - h
    short = _jet(expression, h, order)
    long = _jet(expression, h, order + extra)
    assert short == _below(long, h, order)


@CHECKS
@given(order=st.integers(2, 4))
def test_limit_of_remainder(order):
    polynomial = _jet(sp.exp(h), h, order)
    assert limit((sp.exp(h) - polynomial) / h**order, h, 0) == 1 / sp.factorial(order)


def test_sparse_exponent_cutoff():
    expression = 1 + h**3 + h**7
    expansion = multiseries(
        expression,
        h,
        scale=[h],
        point=0,
        terms=3,
        allow_series_fallback=False,
        return_result=True,
    )
    assert expansion.truncate(3) == expression
    assert _jet(expression, h, 3) == 1


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_puiseux_pullback(degree):
    # Positive coordinates preserve the chosen real root under h=t**degree.
    expression = h ** sp.Rational(1, degree) * sp.exp(h)
    original = multiseries(
        expression,
        h,
        scale=[h],
        point=0,
        terms=4,
        allow_series_fallback=False,
        return_result=True,
    )
    changed = multiseries(
        expression.subs(h, t**degree),
        t,
        scale=[t],
        point=0,
        terms=4,
        allow_series_fallback=False,
        return_result=True,
    )
    assert [term.exponent for term in original.terms(4)] == [
        sp.Rational(1, degree) + k for k in range(4)
    ]
    assert [term.exponent for term in changed.terms(4)] == [
        1 + degree * k for k in range(4)
    ]
    assert [term.coefficient for term in original.terms(4)] == [
        1 / sp.factorial(k) for k in range(4)
    ]
    assert sp.expand(original.truncate(4).subs(h, t**degree) - changed.truncate(4)) == 0
