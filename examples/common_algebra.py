"""Cross-representation arithmetic through the common asymptotic algebra."""

import sympy as sp

from asymptotic import multiseries, nested_series
from asymptotic.algebra import AsymptoticAlgebra


def main() -> None:
    x = sp.symbols("x", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo, terms=4)
    left = multiseries(sp.exp(1 / x), x, terms=5, return_result=True)
    right = nested_series(1 + 1 / x, x, depth=1, return_result=True)
    product = algebra.multiply(left, right)
    prefix = sp.expand(product.truncate())
    assert [prefix.coeff(x, power) for power in (0, -1, -2, -3)] == [
        1,
        2,
        sp.Rational(3, 2),
        sp.Rational(2, 3),
    ]
    assert (
        sp.simplify(
            product.remainder.exact_expression - ((1 + 1 / x) * sp.exp(1 / x) - prefix)
        )
        == 0
    )
    assert product.remainder.check() is True
    print("product:", product.truncate())
    print("remainder:", product.remainder)
    print("certificate verifies:", product.remainder.check())


if __name__ == "__main__":
    main()
