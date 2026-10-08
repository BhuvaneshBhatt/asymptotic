"""Ordinary small-parameter expansion with a certified omitted-term scale."""

import sympy as sp

from asymptotic import multiseries


def main() -> None:
    x = sp.symbols("x", positive=True)
    expansion = multiseries(
        sp.exp(1 / x), x, scale=[1 / x], terms=5, return_result=True
    )
    truncation = expansion.as_element().truncation(3)
    assert truncation.prefix == 1 + 1 / x + 1 / (2 * x**2)
    assert (
        sp.simplify(
            truncation.remainder.exact_expression - (sp.exp(1 / x) - truncation.prefix)
        )
        == 0
    )
    assert truncation.remainder.check() is True
    print("prefix:", truncation.prefix)
    print("remainder:", truncation.remainder)
    print("certificate verifies:", truncation.remainder.check())


if __name__ == "__main__":
    main()
