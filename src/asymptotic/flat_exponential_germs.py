"""Real punctured-origin domination by a flat exponential."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus
from .local_tail_germs import fixed_finite


def flat_exponential_certificate(expr, x, point, domain, assumptions):
    """Bound fixed complex powers times exp(-c/x**m), for even m and c>0."""
    if (
        point != 0
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_real is False
        or x.is_integer is True
        or sp.count_ops(expr) > 40
    ):
        return None
    exponentials = expr.atoms(sp.exp)
    if len(exponentials) != 1:
        return None
    exponential = next(iter(exponentials))
    power = exponential.args[0].as_powers_dict().get(x, sp.S.Zero)
    if not (power.is_Integer and power < 0 and power >= -16 and power % 2 == 0):
        return None
    rate = -exponential.args[0] / x**power
    if rate.has(x) or rate.is_positive is not True or rate.is_finite is not True:
        return None
    # Extract only a multiplicative factor. Replacing all occurrences could
    # mistake an exponentially small exponent for a decaying amplitude.
    if expr == exponential:
        remainder = sp.S.One
    elif expr.is_Mul and exponential in expr.args:
        remainder = sp.Mul(*(factor for factor in expr.args if factor != exponential))
    else:
        return None
    powers = remainder.as_powers_dict()
    order = powers.pop(x, sp.S.Zero)
    coefficient = sp.Mul(*(base**degree for base, degree in powers.items()))
    if (
        order.has(x)
        or coefficient.has(x)
        or not fixed_finite(order)
        or not fixed_finite(coefficient)
    ):
        return None
    return (
        LimitStatus.PROVED,
        sp.S.Zero,
        LimitEvidence(
            "flat_exponential_power_domination",
            "On each real punctured side, the principal complex power has modulus a fixed side-dependent constant times |x|**Re(order). The positive exponential exp(-c/|x|**m), with even m, dominates every fixed algebraic order. The expression is defined away from zero and tends to zero on both sides.",
            value=sp.S.Zero,
        ),
    )
