"""Mathematical behavior through the public product and regime gateways."""

import sympy as sp

from asymptotic import ProductResult, product, stratified_series


def test_constant_product():
    k = sp.Symbol("k", integer=True, positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    result = product(sp.S.One, k, 1, n, parameter=n, return_result=True)
    assert isinstance(result, ProductResult)
    assert result.certified
    assert result.expression == 1


def test_stratified_rational_series():
    x, epsilon = sp.symbols("x epsilon", positive=True)
    result = stratified_series(x / (x + epsilon), (x, epsilon), order=3)
    assert result.certified
    assert len(result.branches) == 3
    assert (
        sp.simplify(
            result.branches[0].approximation - (x / epsilon - x * x / epsilon**2)
        )
        == 0
    )
    assert (
        sp.simplify(
            result.branches[2].approximation - (1 - epsilon / x + epsilon**2 / x**2)
        )
        == 0
    )
