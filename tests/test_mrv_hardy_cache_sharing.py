import sympy as sp

from asymptotic.context import AsymptoticContext
from asymptotic.transseries import transseries_valuation


def test_transseries_valuation_reuses_context_analysis_cache():
    h = sp.symbols("h", positive=True)
    context = AsymptoticContext(h, 0)
    expr = (1 + h) * sp.exp(1 / h)
    first = transseries_valuation(expr, h, context=context)
    before = context.cache_metrics()
    second = transseries_valuation(expr, h, context=context)
    after = context.cache_metrics()
    assert second == first
    assert after["hits"].get("transseries_valuation", 0) == (
        before["hits"].get("transseries_valuation", 0) + 1
    )


def test_context_allows_exprtest_proof_cache():
    x = sp.symbols("x")
    calls = []

    def oracle(expr, **kwargs):
        calls.append(kwargs)
        return False

    context = AsymptoticContext(x, zero_oracle=oracle)
    assert context.is_zero(x + 1) is False
    assert calls[0]["use_cache"] is True
