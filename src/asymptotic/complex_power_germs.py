"""Magnitude certificates for decaying principal complex powers."""

import sympy as sp

from .compact_limit_germs import answer


def _decaying_power(atom, x):
    """Check fixed exponential modulus decay, including principal positive roots."""
    if not atom.is_Pow:
        return False
    outer_order = sp.S.One
    if atom.exp.is_Rational and atom.exp.is_positive and atom.base.is_Pow:
        outer_order, atom = atom.exp, atom.base
    base, exponent = atom.args
    if base.has(x) or base.is_finite is not True or base.is_zero is not False:
        return False
    magnitude = sp.Abs(base)
    if (magnitude - 1).is_positive is not True:
        return False
    rate, offset = sp.diff(exponent, x), exponent.subs(x, 0)
    return (
        not rate.has(x)
        and rate.is_negative is True
        and rate.is_finite is True
        and offset.is_real is True
        and offset.is_finite is True
        and sp.expand(exponent - rate * x - offset) == 0
        and outer_order.is_positive is True
    )


def _polynomial_bound(expr, x):
    if not expr.is_rational_function(x):
        return False
    numerator, denominator = sp.fraction(sp.cancel(expr))
    try:
        n, d = sp.Poly(numerator, x), sp.Poly(denominator, x)
    except sp.PolynomialError:
        return False
    return (
        not d.is_zero
        and max(n.degree(), d.degree()) <= 8
        and all(c.is_finite is True for p in (n, d) for c in p.all_coeffs())
        and d.LC().is_zero is False
    )


def decaying_power_log_certificate(expr, x, point):
    """Evaluate a principal logarithm whose power exponent decays exponentially."""
    if point is not sp.oo or x.is_positive is not True or sp.count_ops(expr) > 65:
        return None
    logs = expr.atoms(sp.log)
    if len(logs) != 1 or expr.free_symbols - {x}:
        return None
    atom = next(iter(logs))
    power = atom.args[0]
    if not power.is_Pow or not _decaying_power(power.exp, x):
        return None
    base = power.base
    if base.has(x) or base.is_finite is not True or base.is_zero is not False:
        return None
    coefficient = sp.cancel(expr / atom)
    if not _polynomial_bound(coefficient, x):
        return None
    return answer(
        sp.S.Zero,
        "decaying_complex_power_log",
        "For fixed nonzero c with Abs(c)>1 and real affine exponent "
        "a*x+b with a<0, the principal power has modulus "
        "exp((a*x+b)*log(Abs(c))). Its product with any fixed rational "
        "coefficient of polynomial growth tends to zero. Writing "
        "d**w=exp(w*log(d)), the imaginary part of w*log(d) is "
        "eventually in (-pi,pi), so principal log(d**w)=w*log(d) "
        "exactly on the tail. Rational denominator poles are "
        "eventually avoided.",
    )


def exponentially_small_denominator_certificate(expr, x, point):
    """Bound complex exponential corrections before expanding a reciprocal."""
    if (
        point is not sp.oo
        or x.is_positive is not True
        or expr.func is not sp.exp
        or sp.count_ops(expr) > 90
        or expr.free_symbols - {x}
    ):
        return None
    exponent = expr.args[0]
    delta = sp.cancel(1 - x / exponent)
    if delta == 0 or not delta.has(x):
        return None
    for term in sp.Add.make_args(sp.expand_mul(delta)):
        decays = [
            factor for factor in sp.Mul.make_args(term) if _decaying_power(factor, x)
        ]
        if len(decays) != 1:
            return None
        coefficient = sp.cancel(term / decays[0])
        replacements = {}
        for atom in coefficient.atoms(sp.sin, sp.cos):
            if atom.args[0].is_real is not True:
                return None
            # Only products of nonnegative integer powers are bounded by one;
            # reciprocal trigonometric factors can have accumulating poles.
            marker = sp.Dummy("bounded_trigonometric_factor")
            replacements[atom] = marker
        reduced = coefficient.xreplace(replacements)
        if replacements:
            try:
                polynomial = sp.Poly(reduced, *replacements.values())
            except sp.PolynomialError:
                return None
            if polynomial.total_degree() > 8 or any(
                not _polynomial_bound(c, x) for c in polynomial.coeffs()
            ):
                return None
        elif not _polynomial_bound(reduced, x):
            return None
    value = sp.oo if delta.is_real is True else sp.zoo
    return answer(
        value,
        "exponentially_small_complex_denominator",
        "Every correction is a fixed principal power with "
        "exponentially decaying modulus, multiplied by a rational "
        "coefficient of polynomial growth and bounded real sine/cosine "
        "factors. Thus delta and x*delta tend to zero and "
        "1-delta is eventually nonzero. The original exponent is "
        "x/(1-delta)=x+o(1), so the exponential has diverging modulus "
        "and phase tending to zero. This certifies complex projective "
        "infinity without separating principal square-root factors.",
    )
