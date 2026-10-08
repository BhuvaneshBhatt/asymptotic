"""Use positive harmonic tails and a fixed-shift gamma ratio."""

import sympy as sp

from asymptotic import limit


def main():
    x = sp.Symbol("x", positive=True)
    assert limit(sp.harmonic(x, 3), x, sp.oo) == sp.zeta(3)
    assert limit(sp.gamma(x + 1) / (x * sp.gamma(x)), x, sp.oo) == 1
    # For positive x the Hurwitz tail has an explicit vanishing upper bound.
    bound = (x + 1) ** -3 + (x + 1) ** -2 / 2
    assert limit(bound, x, sp.oo) == 0
    print("Harmonic zeta limit and gamma shift ratio verified.")


if __name__ == "__main__":
    main()
