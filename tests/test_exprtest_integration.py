from __future__ import annotations

import pytest
import sympy as sp

from asymptotic import AsymptoticContext


def test_configured_zero_oracle_is_cached_locally():
    x = sp.symbols("x")
    calls = []

    def oracle(
        expr, assumptions=True, use_cache=True, *, rng=None, seed=None, confidence="probable"
    ):
        calls.append((expr, use_cache, confidence))
        return True

    ctx = AsymptoticContext(
        x,
        use_sympy_zero_fallback=False,
        zero_oracle=oracle,
    )
    opaque = sp.Function("opaque")(x)
    assert ctx.is_zero(opaque) is True
    assert ctx.is_zero(opaque) is True
    assert calls == [(opaque, False, "certified")]


def test_probable_zero_policy_is_forwarded_to_oracle():
    x = sp.symbols("x")
    seen = []

    def oracle(
        expr, assumptions=True, use_cache=True, *, rng=None, seed=None, confidence="probable"
    ):
        seen.append(confidence)
        return False

    ctx = AsymptoticContext(
        x,
        zero_confidence="probable",
        use_sympy_zero_fallback=False,
        zero_oracle=oracle,
    )
    assert ctx.is_zero(sp.Function("opaque")(x)) is False
    assert seen == ["probable"]


def test_unknown_zero_oracle_can_fall_back_to_sympy():
    x = sp.symbols("x")
    ctx = AsymptoticContext(
        x,
        use_sympy_zero_fallback=True,
        zero_oracle=lambda *args, **kwargs: None,
    )
    expr = sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1
    assert ctx.is_zero(expr) is True


def test_invalid_zero_confidence_is_rejected_before_oracle_call():
    x = sp.symbols("x")
    with pytest.raises(ValueError, match="zero_confidence"):
        AsymptoticContext(
            x,
            zero_confidence="guess",
            zero_oracle=lambda *args, **kwargs: None,
        )
