"""Ordinary small-parameter expansion with a certified omitted-term scale."""

import sympy as sp

from asymptotic import multiseries


def main() -> None:
    x = sp.symbols("x", positive=True)
    expansion = multiseries(sp.exp(1 / x), x, scale=[1 / x], terms=5)
    truncation = expansion.asymptotic_element().truncation(3)
    print("prefix:", truncation.prefix)
    print("remainder:", truncation.remainder)
    print("certificate verifies:", truncation.remainder.check())


if __name__ == "__main__":
    main()
