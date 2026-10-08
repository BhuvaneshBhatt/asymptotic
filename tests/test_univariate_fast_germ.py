import sympy as sp

from asymptotic.limits import limit

x = sp.symbols("x", real=True)


def test_univariate_singular_cancellation_uses_local_germ():
    expr = (sp.exp(x) - 1 - x) / x**2
    result = limit(expr, x, 0, return_result=True)
    assert result.status.name == "PROVED"
    assert result.value == sp.Rational(1, 2)
