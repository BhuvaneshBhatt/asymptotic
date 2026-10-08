import sympy as sp

from asymptotic.remainder import Remainder, RemainderKind


def test_germ_preserving_substitution_keeps_certified_order():
    x, t = sp.symbols("x t", positive=True)
    r = Remainder.big_o(x**3, x, 0, exact_expression=x**4)
    q = r.substitute(t**2, t, 0)
    assert q.kind is RemainderKind.BIG_O and q.scale == t**6


def test_differentiation_replays_instead_of_assuming_big_o_rule():
    x = sp.symbols("x", positive=True)
    r = Remainder.big_o(x**3, x, 0, exact_expression=x**4)
    q = r.differentiate()
    assert q.is_certified and sp.simplify(q.scale - 3 * x**2) == 0


def test_analytic_composition_replays_bound():
    x = sp.symbols("x", positive=True)
    r = Remainder.big_o(x**3, x, 0, exact_expression=x**4)
    q = r.compose_analytic(sp.exp, x)
    assert q.is_certified
