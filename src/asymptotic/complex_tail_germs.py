"""Spherical polynomial growth and fixed imaginary-axis special-function tails."""

import sympy as sp

from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence, LimitStatus
from .local_tail_germs import fixed_finite


def spherical_growth_certificate(expr, x):
    """Prove modulus divergence of a positive rational power of a polynomial."""
    base, exponent = (expr.base, expr.exp) if expr.is_Pow else (expr, sp.S.One)
    if not exponent.is_Rational or not 0 < exponent <= 8:
        return None
    if expr.free_symbols - {x} or bounded_degree(base, (x,), 8) is None:
        return None
    polynomial = sp.Poly(base, x)
    if (
        polynomial.degree() < 1
        or polynomial.LC().is_zero is not False
        or not all(c.is_finite is True for c in polynomial.all_coeffs())
    ):
        return None
    return (
        LimitStatus.PROVED,
        sp.zoo,
        LimitEvidence(
            "spherical_polynomial_power_growth",
            "Outside a sufficiently large disk the polynomial leading term dominates every lower term uniformly in angle. The principal power has modulus Abs(polynomial)**exponent for positive real exponent, so its modulus tends to infinity on every complex approach.",
            value=sp.zoo,
        ),
    )


def imaginary_tail_certificate(expr, x, direction):
    """Use attained circular phases or fixed-order Bessel decay on one axis."""
    height = sp.im(direction)
    if sp.re(direction).is_zero is not True or not fixed_finite(height):
        return None
    side = 1 if height.is_positive is True else -1 if height.is_negative is True else 0
    if not side:
        return None
    if expr == sp.cosh(x):
        n = sp.Dummy("imaginary_tail_index", positive=True, integer=True)
        items = tuple(
            LimitEvidence(
                "attained_imaginary_cosh_phases",
                "cosh(I*t)=cos(t). The two entire-domain sequences have unbounded imaginary modulus in the declared direction and attain exactly +1 and -1; no poles or excluded domain points occur.",
                ((x, side * sp.I * (2 * sp.pi * n + offset)),),
                value,
            )
            for offset, value in ((0, sp.S.One), (sp.pi, -sp.S.One))
        )
        return LimitStatus.DOES_NOT_EXIST, None, items
    if expr.func is sp.besseli and expr.args[1] == x:
        order = expr.args[0]
        if not order.has(x) and fixed_finite(order):
            return (
                LimitStatus.PROVED,
                sp.S.Zero,
                (
                    LimitEvidence(
                        "fixed_order_imaginary_bessel_decay",
                        "DLMF 10.27.6 rotates I_nu(+/-I*r) into a constant phase times J_nu(r). For every fixed finite complex order, DLMF 10.17.3 gives J_nu(r)=O(r**(-1/2)) on the positive real tail. The original imaginary ray avoids the argument cut and zero eventually; order is fixed, not uniform or growing.",
                        value=sp.S.Zero,
                    ),
                ),
            )
    return None
