"""Uniform real trigonometric poles with attained polynomial approaches."""

from itertools import product

import sympy as sp

from ._polynomial_bounds import bounded_degree
from .limit_models import LimitEvidence, LimitStatus
from .multivariate_pole_bounds import _denominator_constraints


def trigonometric_pole_certificate(expr, variables, target, domain, assumptions):
    """Lift the quadratic cosine germ through a real polynomial phase.

    Near zero, 1-cos(h) ~ h**2/2 and 1-sqrt(cos(h)) ~ h**2/4.
    Their zeros are isolated in the phase. A nonzero polynomial ray therefore
    avoids every original pole eventually. Signed infinities require a uniform
    phase sign; agreement of a finite set of rays is never used to prove one.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or expr.free_symbols - set(variables)
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    exponential = expr.func is sp.exp
    if exponential:
        coefficient, denominator = sp.fraction(expr.args[0])
        atom = None
    else:
        numerator, denominator = sp.fraction(expr)
        coefficient, atom = numerator.as_coeff_Mul()
        if atom.func is not sp.sin:
            return None
    if (
        coefficient.is_number is not True
        or coefficient.is_real is not True
        or coefficient.is_finite is not True
    ):
        return None
    base, power = denominator.as_base_exp()
    if power.is_Rational is not True or not 0 < power <= 6:
        return None
    cosine = tuple(base.atoms(sp.cos))
    if len(cosine) != 1:
        return None
    cosine = cosine[0]
    if base == 1 - cosine:
        quadratic = sp.S.Half
    elif base == 1 - sp.sqrt(cosine):
        quadratic = sp.Rational(1, 4)
    else:
        return None
    phase = cosine.args[0]
    if not exponential and atom.args[0] != phase:
        return None
    if (
        bounded_degree(phase, variables, 12) is None
        or phase.subs(dict.fromkeys(variables, 0)) != 0
    ):
        return None
    try:
        polynomial = sp.Poly(phase, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    if polynomial.is_zero or len(polynomial.terms()) > 24:
        return None
    sign = (
        1
        if coefficient.is_positive is True
        else (-1 if coefficient.is_negative is True else None)
    )
    if sign is None:
        return None
    if exponential:
        value = sp.oo if sign == 1 else sp.S.Zero
    elif power < sp.S.Half:
        value = sp.S.Zero
    else:
        if _denominator_constraints(phase, variables) is not None:
            phase_sign = 1
        elif _denominator_constraints(-phase, variables) is not None:
            phase_sign = -1
        else:
            return None
        value = (
            coefficient * phase_sign / sp.sqrt(quadratic)
            if power == sp.S.Half
            else sign * phase_sign * sp.oo
        )
    t = sp.Dummy("trigonometric_ray_parameter", positive=True)
    direction = None
    for ray in product((0, 1, 2), repeat=len(variables)):
        along = polynomial.as_expr().subs(
            dict(zip(variables, (c * t for c in ray), strict=True))
        )
        if along != 0:
            direction = ray
            break
    if direction is None:
        return None
    index = sp.Dummy("attained_trigonometric_index", positive=True, integer=True)
    evidence = LimitEvidence(
        "real_trigonometric_pole_germ",
        f"The real polynomial phase tends uniformly to zero, and the denominator "
        f"is asymptotic to {quadratic} times its square. It is positive on the "
        "original defined germ. The elementary sine or exponential germ supplies "
        "the stated limit; a signed sine pole additionally uses a uniform phase "
        "sign. The attained ray has a nonzero polynomial phase, so it avoids the "
        "isolated denominator zeros eventually, with cos(h)>0 on the root branch.",
        tuple(
            (v, sp.Integer(c) / index)
            for v, c in zip(variables, direction, strict=True)
        ),
        value,
    )
    return LimitStatus.PROVED, value, evidence
