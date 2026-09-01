import sympy as sp

from asymptotic import puiseux_series
from asymptotic.puiseux import (
    algebraic_branches,
    newton_polygon_candidates,
)


def test_puiseux_explicit_ramification():
    x = sp.symbols("x", positive=True)
    s = puiseux_series(sp.sqrt(x) * (1 + x), x, terms=4)
    assert s.ramification_index == 2
    assert s.leading_term.exponent == sp.Rational(1, 2)
    assert s.leading_term.coefficient == 1


def test_newton_polygon_and_square_root_branches():
    x, y = sp.symbols("x y")
    cand = newton_polygon_candidates(y**2 - x, y, x)
    assert len(cand) == 1
    assert cand[0].exponent == sp.Rational(1, 2)
    assert set(cand[0].coefficients) == {-1, 1}

    branches = algebraic_branches(y**2 - x, y, x, terms=4)
    assert len(branches) == 2
    assert {b.series.ramification_index for b in branches} == {2}
    assert {b.newton_exponent for b in branches} == {sp.Rational(1, 2)}
    assert {sp.simplify(b.newton_coefficient) for b in branches} == {-1, 1}


def test_cubic_puiseux_branches_have_ramification_three():
    x, y = sp.symbols("x y")
    branches = algebraic_branches(y**3 - x**2, y, x, terms=4)
    assert len(branches) == 3
    assert all(b.series.leading_term.exponent == sp.Rational(2, 3) for b in branches)
    assert all(b.series.ramification_index == 3 for b in branches)


def test_newton_puiseux_lifts_quintic_without_radical_formula():
    x, y = sp.symbols("x y")
    # Generic Bring-type quintic: SymPy does not need an explicit radical root
    # for the branch through y=0 to be expanded.
    branches = algebraic_branches(y**5 + y - x, y, x, terms=3)
    analytic = [b for b in branches if b.newton_exponent == 1 and b.newton_coefficient == 1]
    assert len(analytic) == 1
    assert sp.expand(analytic[0].series.truncate() - (x - x**5 + 5 * x**9)) == 0
    assert analytic[0].exact_root is None
