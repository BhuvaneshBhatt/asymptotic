import sympy as sp

from asymptotic import compose, inverse, leading_term, series, truncate


def test_generic_representation_queries():
    x = sp.symbols("x", positive=True)
    expansion = series(1 / x + 1 / x**2, x, point=sp.oo, terms=3, return_result=True)
    assert truncate(expansion) == 1 / x + 1 / x**2
    assert leading_term(expansion) == 1 / x


def test_compose_accepts_callable_outer_function():
    x = sp.symbols("x", positive=True)
    expansion = series(1 / x, x, point=sp.oo, terms=3, return_result=True)
    composed = compose(sp.exp, expansion, terms=3, return_result=True)
    assert composed.variable == x
    assert sp.simplify(composed.truncate() - sp.exp(1 / x)) == 0


def test_inverse_dispatches_from_representation():
    x = sp.symbols("x", positive=True)
    expansion = series(1 / x, x, point=sp.oo, terms=3, return_result=True)
    result = inverse(expansion, terms=3, return_result=True)
    assert sp.simplify(result.series.truncate() - 1 / result.inverse_variable) == 0
