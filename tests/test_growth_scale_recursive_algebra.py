import sympy as sp

from asymptotic.relative_growth import (
    GrowthScaleComparison,
    growth_scale,
    prove_growth_comparison,
)


def test_normalized_tree_preserves_product_power_exp_log_structure():
    x = sp.symbols("x", positive=True)
    s = growth_scale(3 * sp.exp(-1 / x) * sp.log(x) ** -2, x)
    assert s.kind == "product"
    assert any(a.kind in {"exp", "power", "product"} for a in s.args)


def test_structural_proof_carries_rule_certificate():
    x = sp.symbols("x", positive=True)
    p = prove_growth_comparison(sp.exp(-1 / x), x**3, x)
    assert (
        p.certified and p.relation == GrowthScaleComparison.LESS and p.proof is not None
    )


def test_nested_exp_comparison():
    x = sp.symbols("x", positive=True)
    p = prove_growth_comparison(sp.exp(-sp.exp(1 / x)), sp.exp(-1 / x), x)
    assert p.certified and p.relation == GrowthScaleComparison.LESS
