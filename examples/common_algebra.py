"""Cross-representation arithmetic through the common asymptotic algebra."""

import sympy as sp

from asymptotic import AsymptoticAlgebra, multiseries, nested_expansion


def main() -> None:
    x = sp.symbols("x", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo, terms=4)
    left = multiseries(sp.exp(1 / x), x, terms=5)
    right = nested_expansion(1 + 1 / x, x, depth=1)
    product = algebra.multiply(left, right)
    print("product:", product.truncate())
    print("remainder:", product.remainder)
    print("certificate verifies:", product.remainder.check())


if __name__ == "__main__":
    main()
