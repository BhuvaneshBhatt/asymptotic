"""Order-targeted completeness checks must obey the symbolic limit policy."""

from __future__ import annotations

import pytest
import sympy as sp

from asymptotic import order_expansion


def test_order_remainder_uses_bounded_limit(monkeypatch):
    epsilon = sp.Symbol("epsilon", positive=True)
    calls = []

    def limit_policy(expr, variable, point, *, direction):
        calls.append((expr, variable, point, direction))
        return sp.S.Zero

    monkeypatch.setattr(order_expansion, "bounded_limit", limit_policy)
    result = order_expansion.expand_to_order(sp.exp(epsilon), epsilon, point=0, order=0)

    assert result.is_complete_through(0)
    assert result.truncate() == 1
    assert len(calls) == 1
    quotient, variable, point, direction = calls[0]
    assert point == 0
    assert direction == "+"
    assert sp.simplify(quotient - variable) == 0


def test_order_remainder_fails_closed_when_limit_budget_declines(monkeypatch):
    epsilon = sp.Symbol("epsilon", positive=True)
    monkeypatch.setattr(order_expansion, "bounded_limit", lambda *args, **kwargs: None)

    result = order_expansion.expand_to_order(sp.exp(epsilon), epsilon, point=0, order=0)

    assert result.status == "unknown"
    assert result.expression is None
    assert result.remainder is None
    assert "symbolic policy" in (result.reason or "")
    with pytest.raises(NotImplementedError):
        result.truncate()


def test_order_remainder_rejects_nonzero_limit(monkeypatch):
    epsilon = sp.Symbol("epsilon", positive=True)
    monkeypatch.setattr(
        order_expansion, "bounded_limit", lambda *args, **kwargs: sp.S.One
    )

    result = order_expansion.expand_to_order(sp.exp(epsilon), epsilon, point=0, order=0)

    assert result.status == "unknown"
    assert "remainder does not vanish" in (result.reason or "")
