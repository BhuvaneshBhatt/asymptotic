import sympy as sp

from asymptotic import (
    GrowthComparison,
    multiseries,
)
from asymptotic.obligations import (
    CoefficientExpansionObligation,
    ComparabilityFactorObligation,
    GrowthComparisonObligation,
    ObligationKind,
    ZeroTestObligation,
)


def test_zero_fact_obligation_resolves_without_changing_scale():
    x = sp.symbols("x", positive=True)
    ms = multiseries(x + 1, x, scale=[1 / sp.log(x), 1 / x])
    _, syms = ms._formal()
    before = ms.scale.exprs
    obligation = ZeroTestObligation(ms.expr, syms[0] * sp.log(x) - 1)

    assert ms.resolve_obligation(obligation)
    assert ms.knowledge.get(obligation) is True
    assert ms.scale.exprs == before
    assert ms.obligation_history[-1].kind is ObligationKind.ZERO_TEST


def test_growth_comparison_obligation_returns_shared_fact():
    x = sp.symbols("x", positive=True)
    ms = multiseries(x + 1, x, scale=[1 / sp.log(x), 1 / x])
    _, syms = ms._formal()
    obligation = GrowthComparisonObligation(ms.expr, syms[0], syms[1])

    assert ms.resolve_obligation(obligation)
    comparison, ratio = ms.knowledge.get(obligation)
    assert comparison is GrowthComparison.LARGER
    assert ratio is sp.oo


def test_coefficient_expansion_obligation_recurses_into_lower_scale():
    x = sp.symbols("x", positive=True)
    ms = multiseries(sp.exp(1 / sp.log(x)) + 1 / x, x, scale=[1 / sp.log(x), 1 / x])
    _, syms = ms._formal()
    obligation = CoefficientExpansionObligation(ms.expr, sp.exp(syms[0]), lower_level=0, terms=3)

    assert ms.resolve_obligation(obligation)
    answer = ms.knowledge.get(obligation)
    expected = 1 + 1 / sp.log(x) + 1 / (2 * sp.log(x) ** 2)
    assert sp.simplify(answer - expected) == 0
    assert ms.scale.exprs == (1 / sp.log(x), 1 / x)


def test_comparability_factor_obligation_returns_factor_and_residual():
    x = sp.symbols("x", positive=True)
    ms = multiseries(sp.exp(-2 * x), x, scale=[1 / x, sp.exp(-x)])
    obligation = ComparabilityFactorObligation(ms.expr, -2 * x, (-x,))

    assert ms.resolve_obligation(obligation)
    candidate, ratio, residual = ms.knowledge.get(obligation)
    assert candidate == -x
    assert ratio == 2
    assert sp.simplify(residual) == 0
