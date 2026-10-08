"""Explicit-tolerance rational approximation on finite represented real numbers."""

import sympy as sp


def rational_approximation(value, tolerance):
    """Smallest denominator rational in the closed tolerance interval.

    The approximation interval is closed. Only rational
    and Float inputs, whose represented values are exact rationals, are accepted.
    Exact irrational inputs and the implicit-tolerance overload remain gaps.
    """
    value, tolerance = map(sp.sympify, (value, tolerance))
    if not isinstance(value, (sp.Rational, sp.Float)) or not isinstance(
        tolerance, (sp.Rational, sp.Float)
    ):
        return sp.Function("rational_approximation")(value, tolerance)
    value, tolerance = sp.Rational(value), sp.Rational(tolerance)
    if tolerance < 0:
        raise ValueError("rational_approximation tolerance must be nonnegative")
    if tolerance == 0:
        return value
    low, high = value - tolerance, value + tolerance
    # Continued-fraction interval descent, with exact endpoints and bounded work.
    offsets = []
    for _ in range(512):
        k = sp.floor(low)
        if low == k:
            result = k
            break
        if sp.floor(high) > k:
            result = k + 1
            break
        offsets.append(k)
        low, high = 1 / (high - k), 1 / (low - k)
    else:
        return sp.Function("rational_approximation")(value, tolerance)
    for k in reversed(offsets):
        result = k + 1 / result
    return result
