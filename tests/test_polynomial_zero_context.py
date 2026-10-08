"""Polynomial identity checks preserve parameter and assumption semantics."""

import sympy as sp

from asymptotic.context import AsymptoticContext


def test_polynomial_identity_at_root():
    x = sp.Symbol("x")
    context = AsymptoticContext(x, point=1)
    assert context.is_zero((x - 1) * (x**8 + 1)) is False
    assert context.is_zero((x + 1) ** 2 - x * x - 2 * x - 1) is True


def test_custom_oracle_is_preserved():
    x = sp.Symbol("x")
    calls = []

    def oracle(expression, **kwargs):
        calls.append(expression)
        return None

    context = AsymptoticContext(x, zero_oracle=oracle, use_sympy_zero_fallback=False)
    assert context.is_zero(x - 1) is None
    assert calls == [x - 1]


def test_parameter_specialization():
    x, a = sp.symbols("x a")
    context = AsymptoticContext(x, assumptions=sp.Eq(a, 0))
    assert context.is_zero(a * (x + 1)) is True
    constrained = AsymptoticContext(x, assumptions=sp.Eq(x, 1))
    assert constrained.is_zero(x - 1) is True
