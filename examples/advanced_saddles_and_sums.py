"""Degenerate/coalescing saddle and discrete summation examples."""

import sympy as sp

from asymptotic import (
    asymptotic_sum,
    coalescing_saddle_asymptotic,
    laplace_asymptotic_integral,
)


def main() -> None:
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)
    quartic = laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )
    print("quartic saddle:", quartic.expression)
    print("quartic certified:", quartic.certified)

    mu = sp.symbols("mu", real=True)
    transition = coalescing_saddle_asymptotic(
        sp.exp(-n * (x**4 / 4 + mu * x**2 / 2)),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        control_parameter=mu,
        terms=1,
    )
    print("coalescing saddle:", transition.expression)

    k = sp.symbols("k", integer=True)
    lattice = asymptotic_sum(
        sp.exp(-n * (k / n) ** 2 / 2),
        k,
        -sp.oo,
        sp.oo,
        parameter=n,
        method="saddle",
        terms=2,
    )
    print("lattice saddle:", lattice.expression)
    print("lattice certified:", lattice.certified)


if __name__ == "__main__":
    main()
