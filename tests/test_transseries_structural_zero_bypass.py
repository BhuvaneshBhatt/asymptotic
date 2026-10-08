import sympy as sp

from asymptotic.context import AsymptoticContext
from asymptotic.transseries import transseries_valuation


def test_structural_valuation_does_not_call_general_zero_oracle():
    h = sp.symbols("h", positive=True)
    calls = []

    def oracle(expr, **kwargs):
        calls.append(expr)

    context = AsymptoticContext(h, 0, zero_oracle=oracle)
    expr = (h**2 / 4 - h - 2) * sp.exp(1 / h)
    value = transseries_valuation(expr, h, context=context)
    assert value is not None
    assert value.leading_coefficient == -2
    assert value.monomial == sp.exp(1 / h)
    assert calls == []


def test_laurent_exponential_valuation_uses_leading_coefficient():
    h = sp.symbols("h", positive=True)
    calls = []

    def oracle(expr, **kwargs):
        calls.append(expr)

    context = AsymptoticContext(h, 0, zero_oracle=oracle)
    expr = h**3 * (h / 64 - sp.Rational(1, 8)) * sp.exp(2 / h)
    value = transseries_valuation(expr, h, context=context)
    assert value is not None
    assert value.leading_coefficient == -sp.Rational(1, 8)
    assert calls == []
