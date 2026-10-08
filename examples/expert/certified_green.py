"""Asymptotically constant Green/Frechet certification."""

import sympy as sp

from asymptotic.remainder_theorems import certify_green_inverse_operator_remainder


def main() -> None:
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.diff(delta(x), x) / x - delta(x)
    certificate, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    print("certified:", certificate.certified)
    print("theorem:", certificate.theorem)
    print("conclusion:", certificate.conclusion)
    if green is not None:
        print("certificate verifies:", green.replay_asymptotic(x))


if __name__ == "__main__":
    main()
