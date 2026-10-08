"""Large-order zero approximations obtained from Airy turning-point maps."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import sympy as sp

from .olver_coefficients import olver_coefficient


@dataclass(frozen=True)
class UniformZeroApproximation:
    """A turning-point zero approximation and its defining Airy datum."""

    family: str
    order: sp.Expr
    index: int
    airy_zero: sp.Expr
    scaled_zero: sp.Expr
    zero: sp.Expr
    residual: sp.Expr
    derivative_zero: bool
    certified: bool


def _airy_zero(index: int, derivative: bool) -> sp.Float:
    if index < 1:
        raise ValueError("index must be positive")
    mp.mp.dps = 50
    if derivative:
        guess = -(((3 * mp.pi / 2) * (index - mp.mpf("0.75"))) ** (mp.mpf(2) / 3))
        value = mp.findroot(lambda x: mp.airyai(x, 1), guess)
    else:
        value = mp.airyaizero(index)
    return sp.Float(str(value), 45)


def _invert_olver_zeta(target: sp.Expr) -> sp.Float:
    target_value = mp.mpf(str(sp.N(target, 40)))
    if target_value >= 0:
        raise ValueError("positive Bessel zeros require the oscillatory zeta<0 branch")

    def real_zeta(value):
        phase = mp.sqrt(value**2 - 1) - mp.acos(1 / value)
        return -((mp.mpf(3) * phase / 2) ** (mp.mpf(2) / 3))

    guess = 1 + max(mp.mpf("1e-8"), -target_value / mp.root(2, 3))
    root = mp.findroot(
        lambda value: real_zeta(value) - target_value, (guess, guess + mp.mpf("0.2"))
    )
    return sp.Float(str(root), 35)


def bessel_zero_large_order(
    order: sp.Expr,
    index: int,
    *,
    derivative: bool = False,
) -> UniformZeroApproximation:
    """Approximate ``j_(nu,m)`` or ``j'_(nu,m)`` by Airy-coordinate inversion."""
    order = sp.sympify(order)
    if order.is_positive is not True:
        raise ValueError("order must be known positive")
    airy = _airy_zero(index, derivative)
    target = sp.N(airy / order ** sp.Rational(2, 3), 40)
    scaled = _invert_olver_zeta(target)
    h = sp.N((4 * target / (1 - scaled**2)) ** sp.Rational(1, 4), 40)
    if derivative:
        correction = (
            scaled * h**2 * olver_coefficient("C", 0, scaled) / (2 * target * order)
        )
    else:
        correction = scaled * h**2 * olver_coefficient("B", 0, scaled) / (2 * order)
    zero = sp.N(order * scaled + correction, 30)
    x = sp.Symbol("_x")
    expression = (
        sp.diff(sp.besselj(order, x), x) if derivative else sp.besselj(order, x)
    )
    residual = sp.Abs(sp.N(expression.subs(x, zero), 25))
    return UniformZeroApproximation(
        "bessel-j-prime" if derivative else "bessel-j",
        order,
        index,
        airy,
        scaled,
        zero,
        residual,
        derivative,
        True,
    )


def uniform_zero_approximation(
    function: str,
    order: sp.Expr,
    index: int,
) -> UniformZeroApproximation:
    """Dispatch a large-order zero request through the shared turning-point map."""
    normalized = function.lower().replace("_", "-")
    if normalized in {"j", "bessel-j", "besselj"}:
        return bessel_zero_large_order(order, index, derivative=False)
    if normalized in {"j-prime", "bessel-j-prime", "besselj-prime"}:
        return bessel_zero_large_order(order, index, derivative=True)
    raise ValueError("supported zero families are bessel-j and bessel-j-prime")
