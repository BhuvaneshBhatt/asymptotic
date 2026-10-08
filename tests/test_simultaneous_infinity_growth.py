import pytest
import sympy as sp

from asymptotic.limits import LimitStatus, limit

x, y = sp.symbols("x y", real=True)


@pytest.mark.parametrize(
    ("expr", "target", "expected"),
    [
        (sp.exp(-y) + sp.exp(-x), (-sp.oo, -sp.oo), sp.oo),
        (sp.exp(-(x**2) - y**2), (sp.oo, sp.oo), 0),
        (x**2 * sp.exp(x * y), (-sp.oo, -sp.oo), sp.oo),
    ],
)
def test_product_end_growth(expr, target, expected):
    result = limit(expr, (x, y), target, return_result=True)
    assert result.status.name == "PROVED"
    assert result.value == expected
    assert result.evidence[0].method == "simultaneous_infinity_growth"


def test_noncoercive_decay_is_not_certified():
    expression = sp.exp(-((x - y) ** 2))
    result = limit(expression, (x, y), (sp.oo, sp.oo), return_result=True)
    assert result.status is LimitStatus.DOES_NOT_EXIST
    t = sp.Symbol("t", positive=True)
    assert expression.subs({x: t, y: t}) == 1
    assert sp.limit(expression.subs({x: 2 * t, y: t}), t, sp.oo) == 0


def test_signed_product_end():
    result = limit(sp.exp(-x * y), (x, y), (sp.oo, -sp.oo), return_result=True)
    assert result.status.name == "PROVED"
    assert result.value is sp.oo


def test_positive_definite_quadratic_exponential_decay():
    expr = sp.exp(-(x**2 - x * y + y**2))
    result = limit(expr, (x, y), (sp.oo, sp.oo), return_result=True)
    assert result.status.name == "PROVED"
    assert result.value == 0
