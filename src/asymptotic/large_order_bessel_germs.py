"""Proportional large-order Bessel germs on a positive real tail."""

import sympy as sp

from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence, LimitStatus


def _constant_sign(expression):
    """Prove a real constant sign using exact rational or outward-rounded balls."""
    expression = sp.sympify(expression)
    if expression == 0:
        return 0
    if expression.is_Rational:
        return 1 if expression > 0 else -1
    if (
        expression.free_symbols
        or expression.has(sp.Float)
        or sp.count_ops(expression) > 80
    ):
        return None
    try:
        from flint import arb, ctx
    except ImportError:
        return None

    def enclosure(e):
        if e.is_Rational:
            return arb(int(e.p)) / int(e.q)
        if e is sp.pi:
            return arb.pi()
        if e is sp.E:
            return arb(1).exp()
        if e.is_Add or e.is_Mul:
            result = arb(0 if e.is_Add else 1)
            for a in e.args:
                value = enclosure(a)
                if value is None:
                    return None
                result = result + value if e.is_Add else result * value
            return result
        if e.is_Pow and e.exp.is_Rational and abs(e.exp) <= 64:
            value = enclosure(e.base)
            if value is None:
                return None
            if e.exp.is_Integer:
                if e.exp < 0 and not (value > 0 or value < 0):
                    return None
                return value ** int(e.exp)
            if not value > 0:
                return None
            return (value.log() * enclosure(e.exp)).exp()
        if e.func in (sp.log, sp.exp):
            value = enclosure(e.args[0])
            if value is None:
                return None
            if e.func is sp.log:
                return value.log() if value > 0 else None
            return value.exp()
        return None

    with ctx.workprec(128):
        value = enclosure(expression)
        if value is not None and value.is_finite():
            if value > 0:
                return 1
            if value < 0:
                return -1
    return None


def proportional_bessel_certificate(expr, x, point, domain, assumptions):
    """Use Debye's relative remainder for one multiplicatively scaled J or Y.

    A fixed rational ratio nu/z>1 stays outside the turning-point regime.
    The 1+O(1/nu) relative remainder survives multiplication by the admitted
    monomial/exponential weight, but does not license subtractive cancellation.
    """
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or x.is_real is False
    ):
        return None
    if not expr.has(sp.besselj, sp.bessely) or sp.count_ops(expr) > 120:
        return None
    atoms = expr.atoms(sp.besselj, sp.bessely)
    if len(atoms) != 1 or expr.has(sp.Float) or expr.free_symbols - {x}:
        return None
    atom = next(iter(atoms))
    if expr.as_powers_dict().get(atom) != 1:
        return None
    order, argument = atom.args
    a, m = argument.as_coeff_exponent(x)
    if not a.is_Rational or a <= 0 or not m.is_Integer or not 1 <= m <= 8:
        return None
    order_coefficient, order_power = order.as_coeff_exponent(x)
    if order_power != m or order_coefficient.is_Rational is not True:
        return None
    ratio = order_coefficient / a
    if not ratio.is_Rational or not 1 < ratio <= 16:
        return None
    factors = list(sp.Mul.make_args(expr))
    if atom not in factors:
        return None
    factors.remove(atom)
    weight = sp.Mul(*factors)
    if weight.has(atom):
        return None
    phase, power, coefficient = sp.S.Zero, sp.S.Zero, sp.S.One
    for factor in sp.Mul.make_args(weight):
        if factor.func is sp.exp:
            if bounded_degree(factor.args[0], (x,), 8) is None:
                return None
            phase += factor.args[0]
        elif factor == x:
            power += 1
        elif (
            factor.is_Pow
            and factor.base == x
            and factor.exp.is_Rational
            and abs(factor.exp) <= 8
        ):
            power += factor.exp
        elif not factor.has(x):
            coefficient *= factor
        else:
            return None
    sign = _constant_sign(coefficient)
    if sign not in (-1, 1):
        return None
    root = sp.sqrt(ratio**2 - 1)
    rate = ratio * sp.log(ratio + root) - root
    phase += (-1 if atom.func is sp.besselj else 1) * rate * argument
    try:
        polynomial = sp.Poly(phase, x, expand=False)
    except sp.PolynomialError:
        return None
    signs = {degree: _constant_sign(c) for (degree,), c in polynomial.terms()}
    if any(s is None for s in signs.values()):
        return None
    for (degree,), c in polynomial.terms():
        if degree == 0:
            continue
        direction = signs[degree]
        if direction is None:
            return None
        if direction:
            result = (
                sp.S.Zero
                if direction < 0
                else sign * (1 if atom.func is sp.besselj else -1) * sp.oo
            )
            break
    else:
        remaining_power = power - m / 2
        if remaining_power < 0:
            result = sp.S.Zero
        elif remaining_power > 0:
            result = sign * (1 if atom.func is sp.besselj else -1) * sp.oo
        else:
            result = (
                coefficient
                * (1 if atom.func is sp.besselj else -2)
                * sp.exp(polynomial.nth(0))
                / sp.sqrt(2 * sp.pi * a * root)
            )
    return (
        LimitStatus.PROVED,
        result,
        LimitEvidence(
            "proportional_large_bessel_order",
            "DLMF 10.19.3: nu/z is a fixed rational constant >1, z=a*x^m tends to positive infinity, and the Debye leading term has relative error O(1/nu). Exact polynomial scale comparison and certified constant signs determine the multiplied limit. No turning-point, complex-sector or subtractive cancellation theorem is inferred.",
            value=result,
        ),
    )
