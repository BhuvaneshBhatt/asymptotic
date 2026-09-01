import sympy as sp

from asymptotic import (
    dominant_balance_candidates,
    implicit_asymptotic,
    inverse_asymptotic,
    series_reversion,
)
from asymptotic.puiseux import newton_polygon_candidates


def test_series_reversion_regular_analytic():
    x, y = sp.symbols("x y", positive=True)
    branch = series_reversion(x + x**2, x, y, terms=4, branch=0)
    got = sp.expand(branch.truncate())
    assert sp.expand(got - (y - y**2 + 2 * y**3 - 5 * y**4)) == 0
    residual = sp.series((x + x**2).subs(x, got) - y, y, 0, 5).removeO()
    assert sp.expand(residual) == 0


def test_series_reversion_puiseux_leading_branch():
    x, y = sp.symbols("x y", positive=True)
    branch = series_reversion(x**2 + x**3, x, y, terms=3, branch=0)
    assert branch.series.ramification_index == 2
    assert branch.leading_exponent == sp.Rational(1, 2)
    assert sp.simplify(abs(branch.series.leading_term.coefficient) - 1) == 0
    assert branch.series.leading_term.exponent == sp.Rational(1, 2)


def test_inverse_asymptotic_at_infinity():
    x, y = sp.symbols("x y", positive=True)
    branch = inverse_asymptotic(x + 1 / x, x, y, point=sp.oo, terms=3)
    # Reciprocal coordinates reduce the inverse at infinity to local reversion.
    assert branch.series.leading_term.exponent == 1
    assert sp.simplify(branch.series.leading_term.coefficient - 1) == 0


def test_shared_dominant_balance_matches_newton_polygon():
    x, y = sp.symbols("x y")
    shared = dominant_balance_candidates(y**3 + x * y - x**2, y, x)
    old = newton_polygon_candidates(y**3 + x * y - x**2, y, x)
    assert [c.exponent for c in shared] == [c.exponent for c in old]
    assert all(
        sp.simplify(
            a.coefficient_equation
            - b.coefficient_equation.xreplace(
                {
                    next(
                        iter(b.coefficient_equation.free_symbols - {x, y}), sp.Symbol("dummy")
                    ): next(iter(a.coefficient_equation.free_symbols - {x, y}), sp.Symbol("dummy"))
                }
            )
        )
        == 0
        for a, b in zip(shared, old)
    )


def test_implicit_asymptotic_square_root_branches():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(y**2 - x - x**2, y, x, terms=3)
    assert len(branches) == 2
    leads = {sp.simplify(b.series.leading_term.coefficient) for b in branches}
    assert leads == {-1, 1}
    assert all(b.series.leading_term.exponent == sp.Rational(1, 2) for b in branches)


def test_implicit_asymptotic_quintic_without_explicit_root():
    x, y = sp.symbols("x y")
    branches = implicit_asymptotic(y**5 + y - x, y, x, terms=3)
    analytic = [b for b in branches if b.balance.exponent == 1 and b.leading_coefficient == 1]
    assert len(analytic) == 1
    got = analytic[0].series.truncate()
    assert sp.expand(got - (x - x**5 + 5 * x**9)) == 0


def test_implicit_asymptotic_at_infinity():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(y**2 - x**3 - x, y, x, point=sp.oo, terms=2)
    assert len(branches) >= 2
    assert {b.series.leading_term.exponent for b in branches} == {sp.Rational(3, 2)}


def test_repeated_newton_root_splits_after_translation():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic((y - x) ** 2 - x**3, y, x, terms=2)
    assert len(branches) == 2
    got = {sp.expand(branch.series.truncate()) for branch in branches}
    assert got == {x - x ** sp.Rational(3, 2), x + x ** sp.Rational(3, 2)}
    assert all(len(branch.balance_path) >= 2 for branch in branches)
    assert {branch.balance_path[1].exponent for branch in branches} == {sp.Rational(3, 2)}


def test_repeated_cubic_newton_root_recursively_splits():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic((y - x) ** 3 - x**5, y, x, terms=2)
    assert len(branches) == 3
    assert all(branch.balance_path[0].exponent == 1 for branch in branches)
    assert all(branch.balance_path[1].exponent == sp.Rational(5, 3) for branch in branches)
    coeffs = {sp.simplify(branch.series.terms[1].coefficient ** 3) for branch in branches}
    assert coeffs == {1}


def test_transcendental_implicit_sine_inverse_without_explicit_solve():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(sp.sin(y) - x, y, x, terms=3)
    assert len(branches) == 1
    got = sp.expand(branches[0].series.truncate())
    expected = x + x**3 / 6 + 3 * x**5 / 40
    assert sp.expand(got - expected) == 0


def test_transcendental_implicit_exponential_inverse():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(sp.exp(y) - 1 - x, y, x, terms=4)
    assert len(branches) == 1
    got = sp.expand(branches[0].series.truncate())
    expected = x - x**2 / 2 + x**3 / 3 - x**4 / 4
    assert sp.expand(got - expected) == 0


def test_transcendental_branch_can_use_nonzero_dependent_limit():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(
        sp.sin(y) - x,
        y,
        x,
        dependent_limit=sp.pi,
        terms=2,
    )
    assert len(branches) == 1
    got = sp.expand(branches[0].series.truncate())
    assert sp.expand(got - (sp.pi - x)) == 0


def test_transcendental_logarithmic_branch_at_nonzero_center():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(
        sp.log(y) - x,
        y,
        x,
        dependent_limit=1,
        terms=4,
    )
    assert len(branches) == 1
    got = sp.expand(branches[0].series.truncate())
    expected = 1 + x + x**2 / 2 + x**3 / 6
    assert sp.expand(got - expected) == 0
