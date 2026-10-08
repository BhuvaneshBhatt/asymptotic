"""Structural polynomial bounds that avoid expansion during recognition."""

from functools import lru_cache


def bounded_degree(expr, variables, maximum):
    """Return a syntactic degree upper bound, or None outside the budget.

    Product degrees add and power degrees multiply. Checking the expression
    tree first prevents polynomial conversion from expanding oversized powers.
    Cancellation may lower the true degree; declining such a form is safe.
    """
    if not expr.has(*variables):
        return 0
    if expr in variables:
        return 1
    if expr.is_Add or expr.is_Mul:
        degrees = [bounded_degree(a, variables, maximum) for a in expr.args]
        if any(d is None for d in degrees):
            return None
        degree = max(degrees) if expr.is_Add else sum(degrees)
    elif expr.is_Pow and expr.exp.is_Integer and 0 <= expr.exp <= maximum:
        base_degree = bounded_degree(expr.base, variables, maximum)
        if base_degree is None:
            return None
        degree = int(expr.exp) * base_degree
    else:
        return None
    return degree if degree <= maximum else None


@lru_cache(maxsize=256)
def bounded_expansion_width(expr, maximum):
    """Bound rational expansion size before exposing products or integer powers.

    Function heads and fractional powers are atomic for this estimate.
    Negative integer powers contribute the expansion size of their denominator.
    The estimate can overcount cancellations. An oversized expression returns
    maximum + 1, never None. A 256-entry cache shares immutable structural
    estimates without caching assumptions or mathematical proof decisions.
    """
    if expr.is_Add:
        width = 0
        for term in expr.args:
            width += bounded_expansion_width(term, maximum)
            if width > maximum:
                return maximum + 1
        return width
    if expr.is_Mul:
        width = 1
        for factor in expr.args:
            width *= bounded_expansion_width(factor, maximum)
            if width > maximum:
                return maximum + 1
        return width
    if expr.is_Pow and expr.exp.is_Integer is True:
        base_width = bounded_expansion_width(expr.base, maximum)
        if base_width == 1:
            return 1
        exponent = abs(expr.exp)
        if exponent > maximum or base_width > maximum:
            return maximum + 1
        return min(maximum + 1, base_width ** int(exponent))
    return 1
