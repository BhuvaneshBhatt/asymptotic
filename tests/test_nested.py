import sympy as sp

from asymptotic import nested_expansion
from asymptotic.nested import nested_form


def test_power_nested_form():
    x = sp.symbols("x", positive=True)
    nf = nested_form(x**2, x)
    assert nf.constant is None
    assert nf.outer_sign == 1
    level = nf.levels[0]
    assert level.epsilon == 1
    assert level.exp_depth == 0
    assert level.log_depth == 0
    assert sp.simplify(level.power - 2) == 0
    assert sp.limit(level.remainder, x, sp.oo) == 1
    assert sp.simplify(nf.reconstruct() - x**2) == 0


def test_exponential_nested_form():
    x = sp.symbols("x", positive=True)
    nf = nested_form(sp.exp(x**2), x)
    level = nf.levels[0]
    assert level.exp_depth == 1
    assert level.log_depth == 0
    assert sp.simplify(level.power - 2) == 0
    assert sp.limit(level.remainder, x, sp.oo) == 1
    assert sp.simplify(nf.reconstruct() - sp.exp(x**2)) == 0


def test_vanishing_nested_form():
    x = sp.symbols("x", positive=True)
    nf = nested_form(x**-2, x)
    level = nf.levels[0]
    assert level.epsilon == -1
    assert level.exp_depth == 0
    assert level.log_depth == 0
    assert sp.simplify(level.power - 2) == 0
    assert sp.simplify(nf.reconstruct() - x**-2) == 0


def test_finite_limit_peeling():
    x = sp.symbols("x", positive=True)
    nf = nested_form(3 + 1 / x, x)
    assert nf.constant == 3
    assert nf.outer_sign == 1
    assert nf.levels[0].epsilon == -1
    assert sp.simplify(nf.reconstruct() - (3 + 1 / x)) == 0


def test_nested_expansion_refines_remainder():
    x = sp.symbols("x", positive=True)
    expr = x**2 * (1 + 1 / sp.log(x))
    ne = nested_expansion(expr, x, depth=3)
    assert len(ne.forms) >= 2
    assert sp.simplify(ne.forms[0].reconstruct() - expr) == 0
    assert ne.forms[1].constant == 1
