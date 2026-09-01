import sympy as sp

from asymptotic import (
    TransseriesExpansion,
    implicit_asymptotic,
)
from asymptotic.dominant import transseries_dominant_balance_candidates
from asymptotic.transseries import transseries_valuation


def test_transseries_valuation_recognizes_exponential_small_monomial():
    x = sp.symbols("x", positive=True)
    valuation = transseries_valuation(sp.exp(-1 / x) / x, x)
    assert valuation is not None
    assert sp.simplify(valuation.leading_coefficient - 1) == 0
    assert sp.simplify(valuation.monomial - sp.exp(-1 / x) / x) == 0


def test_general_balance_finds_exponential_correction_scale():
    x, delta = sp.symbols("x delta", positive=True)
    candidates = transseries_dominant_balance_candidates(
        delta**2 + 2 * delta / x - sp.exp(-1 / x),
        delta,
        x,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert sp.simplify(candidate.monomial - x * sp.exp(-1 / x)) == 0
    assert candidate.coefficients == (sp.Rational(1, 2),)


def test_implicit_solver_adds_beyond_all_orders_correction_at_infinity():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(
        y**2 - x**2 - sp.exp(-x),
        y,
        x,
        point=sp.oo,
        terms=2,
    )
    assert len(branches) == 2
    assert all(isinstance(branch.series, TransseriesExpansion) for branch in branches)
    got = {sp.expand(branch.series.truncate()) for branch in branches}
    assert got == {
        x + sp.exp(-x) / (2 * x),
        -x - sp.exp(-x) / (2 * x),
    }


def test_implicit_solver_handles_logarithmic_corrections():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(
        y**2 - x**2 - sp.log(x),
        y,
        x,
        point=sp.oo,
        terms=3,
    )
    positive = next(
        branch
        for branch in branches
        if branch.series.truncate().could_extract_minus_sign() is False
    )
    got = sp.expand(positive.series.truncate())
    expected = x + sp.log(x) / (2 * x) - sp.log(x) ** 2 / (8 * x**3)
    assert sp.simplify(got - expected) == 0


def test_puiseux_branches_remain_puiseux_for_backward_compatibility():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(y**2 - x - x**2, y, x, terms=3)
    assert all(not branch.is_transseries for branch in branches)
    assert {branch.series.leading_term.exponent for branch in branches} == {sp.Rational(1, 2)}
