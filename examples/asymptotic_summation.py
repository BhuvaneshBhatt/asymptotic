"""Creative telescoping, Mellin, Poisson, and multidimensional sum examples."""

import sympy as sp

from asymptotic import asymptotic_sum


def main() -> None:
    n = sp.symbols("n", nonnegative=True, integer=True)
    k = sp.symbols("k", integer=True)

    telescoped = asymptotic_sum(
        sp.binomial(n, k),
        k,
        0,
        sp.oo,
        parameter=n,
        method="zeilberger",
        terms=3,
    )
    print("Zeilberger:", telescoped.expression, telescoped.status)
    print("certificate verifies:", telescoped.certificate.replay())

    x = sp.symbols("x", positive=True)
    gaussian = asymptotic_sum(
        sp.exp(-x * k**2),
        k,
        -sp.oo,
        sp.oo,
        parameter=x,
        point=0,
        method="poisson",
        terms=2,
    )
    print("Poisson:", gaussian.expression, gaussian.remainder)

    positive_k = sp.symbols("positive_k", positive=True, integer=True)
    bessel = asymptotic_sum(
        sp.besselk(0, x * positive_k),
        positive_k,
        1,
        sp.oo,
        parameter=x,
        point=0,
        method="mellin",
        terms=3,
    )
    print("Mellin:", bessel.expression, bessel.status)

    i, j = sp.symbols("i j", integer=True)
    multi = asymptotic_sum(
        (1 + x * i) * (1 + x * j),
        (i, j),
        (0, 0),
        (2, 3),
        parameter=x,
        point=0,
        method="series",
        terms=3,
    )
    print("multidimensional:", multi.expression, multi.status)


if __name__ == "__main__":
    main()
