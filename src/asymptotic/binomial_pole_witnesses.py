"""Attained real curves around rational and signed-root binomial poles."""

import sympy as sp

from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence, LimitStatus


def binomial_pole_conflict(expr, variables, target, domain, assumptions):
    """Separate an axis from a root numerator or denominator near a pole curve.

    For P=a*x**m+b*y**n, let y=s*t and x=c*t**(n/m)*(1+t**k),
    where c**m=-b*s**n/a. Then P=-b*s**n*t**n*((1+t**k)**m-1)
    is nonzero for every t>0. An exact exponent comparison selects k so the
    quotient diverges, while a numerator-zero axis is defined and identically
    zero. Only displayed monomials and fixed real rational data are admitted.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) != 2
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 45
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(p.is_Rational is not True for p in target)
    ):
        return None
    center = dict(
        zip(
            variables,
            (v + p for v, p in zip(variables, target, strict=True)),
            strict=True,
        )
    )
    shifted = expr.subs(center, simultaneous=True)
    condition = None
    if shifted.func is sp.Piecewise:
        shifted, condition = shifted.args[0]
        if condition.func is not sp.Ne:
            return None
    signs = shifted.atoms(sp.sign)
    plain = not signs
    if plain:
        sign = inner = magnitude_power = None
        sign_power = sp.S.Zero
    else:
        if len(signs) != 1:
            return None
        sign = next(iter(signs))
        inner = sign.args[0]
        factors = shifted.as_powers_dict()
        sign_power = factors.get(sign)
        magnitude_power = factors.get(sp.Abs(inner))
        if (
            sign_power is None
            or sign_power.is_Integer is not True
            or abs(sign_power) > 4
            or magnitude_power is None
            or magnitude_power.is_Rational is not True
        ):
            return None
    for x, y in (variables, variables[::-1]):
        if plain:
            root_numerator = False
            num, polynomial = shifted.as_numer_denom()
            exponent = sp.S.One
        elif inner == x and 0 < magnitude_power < 1:
            root_numerator = True
            remaining = shifted / (sign**sign_power * sp.Abs(inner) ** magnitude_power)
            num, polynomial = sp.fraction(remaining)
            exponent = magnitude_power
        elif magnitude_power < 0 and -magnitude_power < 1:
            root_numerator = False
            polynomial = inner
            exponent = -magnitude_power
            num = shifted / (sign**sign_power * sp.Abs(inner) ** magnitude_power)
        else:
            continue
        if (
            bounded_degree(polynomial, (x, y), 6) is None
            or bounded_degree(num, (x, y), 12) is None
        ):
            continue
        try:
            pole = sp.Poly(polynomial, x, y, domain=sp.QQ)
            if condition is not None:
                condition_pole = sp.Poly(
                    condition.lhs - condition.rhs, x, y, domain=sp.QQ
                )
                if condition_pole.is_zero or condition_pole.monic() != pole.monic():
                    continue
            terms = pole.terms()
            numerator = sp.Poly(num, x, y, domain=sp.QQ).terms()
        except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
            continue
        if len(terms) != 2 or len(numerator) != 1:
            continue
        (first, a), (second, b) = terms
        m, n = first[0], second[1]
        if first != (m, 0) or second != (0, n) or not 1 <= m <= 6 or not 1 <= n <= 6:
            continue
        (u, v), coefficient = numerator[0]
        if not (0 <= u <= 8 and 0 <= v <= 8) or coefficient == 0 or u + v == 0:
            continue
        s = sp.S.One
        ratio = -b / a
        if m % 2 == 0 and ratio < 0:
            if n % 2 == 0:
                continue
            s = -sp.S.One
            ratio = -b * s**n / a
        c = sp.real_root(ratio, m)
        q = sp.Rational(n, m)
        order = q * (u + exponent) + v if root_numerator else q * u + v
        k = (
            max(1, int(sp.floor(order - n)) + 1)
            if root_numerator
            else max(1, int(sp.floor(order / exponent - n)) + 1)
        )
        if k > 32:
            continue
        j = sp.Dummy("attained_binomial_index", positive=True, integer=True)
        px = target[variables.index(x)]
        py = target[variables.index(y)]
        curve = ((x, px + c * j ** (-q) * (1 + j ** (-k))), (y, py + s / j))
        leading = coefficient * c**u * s**v
        if root_numerator:
            leading *= sp.Abs(c) ** exponent * sp.sign(c) ** sign_power
            leading /= -b * s**n * m
        else:
            d = -b * s**n * m
            leading = (
                leading / d
                if plain
                else leading * sp.Abs(d) ** (-exponent) * sp.sign(d) ** sign_power
            )
        if leading.is_positive is True:
            value = sp.oo
        elif leading.is_negative is True:
            value = -sp.oo
        else:
            continue
        if v > 0:
            axis = ((x, px + 1 / j), (y, py))
        elif not root_numerator or sign_power >= 0:
            axis = ((x, px), (y, py + 1 / j))
        else:
            continue
        return (
            LimitStatus.DOES_NOT_EXIST,
            None,
            (
                LimitEvidence(
                    "attained_binomial_axis",
                    "The displayed numerator vanishes on this axis and the binomial denominator, including any signed root, is nonzero. An admitted first piecewise condition is exactly the nonvanishing of this polynomial, so that branch is selected. Every point is defined and the value is exactly zero.",
                    axis,
                    sp.S.Zero,
                ),
                LimitEvidence(
                    "attained_binomial_pole_curve",
                    f"For t=1/j, the original pole polynomial is -b*s**n*t**n*((1+t**({k}))**m-1), nonzero for every positive t. Both coordinates approach the target. Any root and sign are evaluated on their attained real branches. An admitted first piecewise condition is equivalent to the nonvanishing of this polynomial, so both sequences select that branch. The exact leading numerator/denominator exponent comparison gives the stated signed infinity.",
                    curve,
                    value,
                ),
            ),
        )
    return None
