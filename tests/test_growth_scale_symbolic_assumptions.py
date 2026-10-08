import sympy as sp

from asymptotic.relative_growth import GrowthScaleComparison, prove_growth_comparison


def test_symbolic_power_order_uses_assumptions():
    x = sp.symbols("x", positive=True)
    N, M = sp.symbols("N M", real=True)
    p = prove_growth_comparison(x**N, x**M, x, assumptions=sp.Gt(N, M))
    assert p.certified and p.relation is GrowthScaleComparison.LESS
    assert p.proof.rule == "symbolic_power_order"


def test_symbolic_power_order_without_relation_stays_unknown():
    x = sp.symbols("x", positive=True)
    N, M = sp.symbols("N M", real=True)
    assert (
        prove_growth_comparison(x**N, x**M, x).relation is GrowthScaleComparison.UNKNOWN
    )


def test_symbolic_inverse_log_order_uses_assumptions():
    x = sp.symbols("x", positive=True)
    N, M = sp.symbols("N M", positive=True)
    p = prove_growth_comparison(
        sp.log(1 / x) ** (-N), sp.log(1 / x) ** (-M), x, assumptions=sp.Gt(N, M)
    )
    assert p.certified and p.relation is GrowthScaleComparison.LESS


def test_flat_exponential_symbolic_positive_exponents():
    x = sp.symbols("x", positive=True)
    q, N = sp.symbols("q N", positive=True)
    p = prove_growth_comparison(sp.exp(-1 / x**q), x**N, x)
    assert p.certified and p.relation is GrowthScaleComparison.LESS
    assert p.proof.rule == "flat_exponential_vs_power"


def test_power_vs_inverse_log_symbolic_positive_exponents():
    x = sp.symbols("x", positive=True)
    N, M = sp.symbols("N M", positive=True)
    p = prove_growth_comparison(x**N, sp.log(1 / x) ** (-M), x)
    assert p.certified and p.relation is GrowthScaleComparison.LESS


def test_positive_constants_are_normalized_not_order_changing():
    x = sp.symbols("x", positive=True)
    p = prove_growth_comparison(7 * x**3, x**3, x)
    assert p.certified and p.relation is GrowthScaleComparison.EQUIVALENT


def test_nested_negative_exponential_layers_are_structural():
    x = sp.symbols("x", positive=True)
    p = prove_growth_comparison(sp.exp(-sp.exp(1 / x)), sp.exp(-1 / x), x)
    assert p.certified and p.relation is GrowthScaleComparison.LESS
