"""Probability and expectation asymptotics with moving-domain Laplace analysis."""

import sympy as sp
from sympy.stats import Normal

from asymptotic import asymptotic_expectation, asymptotic_probability


def main():
    """Run one moving-tail and one interior-saddle example."""

    n, a = sp.symbols("n a", positive=True)

    # A moving Gaussian tail. The implementation discovers x = n*y and then
    # applies the lower-endpoint Laplace expansion on y in (a, infinity).
    x = Normal("X_probability_example", 0, sp.sqrt(n))
    tail = asymptotic_probability(x > a * n, x, parameter=n, terms=3)

    # A concentrating Gaussian expectation. Forcing the Laplace route makes
    # the interior saddle visible even though this example has an exact MGF.
    y = Normal("Y_expectation_example", 0, 1 / sp.sqrt(n))
    moment = asymptotic_expectation(sp.exp(y), y, parameter=n, terms=3, method="laplace")

    print("tail method:", tail.method)
    print("tail:", tail.expression)
    print("expectation method:", moment.method)
    print("expectation:", moment.expression)
    return tail, moment


if __name__ == "__main__":
    main()
