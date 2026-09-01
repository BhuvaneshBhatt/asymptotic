"""Demonstrate structural complexity of finite transseries multiplication."""

import sympy as sp

from asymptotic import transseries_from_expression
from asymptotic.instrumentation import symbolic_metrics


def multiplication_pair_count(size: int) -> tuple[int, int]:
    """Return retained input size and raw term-pair count for equal-size products.

    The multiplication kernel forms every pair of nonconstant retained terms
    before canonical collection.  For two inputs with ``size`` terms, this
    stage therefore has exactly ``size**2`` candidate products.
    """

    if size < 1:
        raise ValueError("size must be positive")

    x = sp.symbols("x", positive=True)
    left = transseries_from_expression(
        sum(sp.Rational(1, i + 1) / x**i for i in range(1, size + 1)),
        x,
        point=sp.oo,
        complete=True,
    )
    right = transseries_from_expression(
        sum(sp.Rational(1, i + 2) / x ** (2 * i - 1) for i in range(1, size + 1)),
        x,
        point=sp.oo,
        complete=True,
    )

    with symbolic_metrics() as metrics:
        product = left * right

    retained = len(left.terms)
    expected = len(left.terms) * len(right.terms)
    if metrics.term_products != expected:
        raise RuntimeError("term-product instrumentation disagrees with multiplication")
    if product.truncate() == 0:
        raise RuntimeError("nonzero inputs unexpectedly produced a zero product")
    return retained, metrics.term_products


def main() -> None:
    """Print exact candidate counts for a small doubling sequence."""

    print("terms per input | raw term products")
    for size in (2, 4, 8):
        retained, products = multiplication_pair_count(size)
        print(f"{retained:15d} | {products:17d}")
    print("candidate-generation complexity: Theta(n*m), Theta(n^2) when n=m")


if __name__ == "__main__":
    main()
