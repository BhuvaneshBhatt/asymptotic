import sympy as sp

from asymptotic.nscale_stratification import (
    nscale_stratified_expand,
    relevant_scale_comparisons,
)
from asymptotic.proof_obligations import ObligationKind
from asymptotic.stratified_expansion import stratified_expand


def test_three_competing_scales_have_13_total_preorders():
    x, e, d = sp.symbols("x e d", positive=True)
    result = nscale_stratified_expand(x / (x + e + d), (x, e, d))
    assert result.certified
    assert len(result.branches) == 13
    assert all(b.approximation == x / (x + e + d) for b in result.branches)


def test_comparable_blocks_retain_ratio_coordinates():
    x, e, d = sp.symbols("x e d", positive=True)
    result = nscale_stratified_expand(x / (x + e + d), (x, e, d))
    fully_comparable = [
        b
        for b in result.branches
        if "~" in b.regime.statement and "<o<" not in b.regime.statement
    ]
    assert len(fully_comparable) == 1
    assert len(fully_comparable[0].regime.scaled_coordinates) == 2


def test_budget_declines_without_partial_coverage_claim():
    x, e, d = sp.symbols("x e d", positive=True)
    result = nscale_stratified_expand(x / (x + e + d), (x, e, d), branch_budget=5)
    assert not result.certified
    assert result.branches == ()
    assert result.obligations[0].kind is ObligationKind.SYMBOLIC_BUDGET


def test_irrelevant_variable_not_added_to_graph():
    x, e, d = sp.symbols("x e d", positive=True)
    comparisons = relevant_scale_comparisons(x / (x + e), (x, e, d))
    assert {(c.left, c.right) for c in comparisons} == {(e, x)} or {
        (c.left, c.right) for c in comparisons
    } == {(x, e)}


def test_public_stratifier_routes_three_scale_problem():
    x, e, d = sp.symbols("x e d", positive=True)
    result = stratified_expand(x / (x + e + d), (x, e, d))
    assert result.certified
    assert len(result.branches) == 13
