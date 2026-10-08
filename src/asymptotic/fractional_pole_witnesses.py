"""Attained fractional phases on exact rational pole charts."""

import sympy as sp

from ._polynomial_bounds import bounded_degree, bounded_expansion_width
from .attained_ray_germs import ray_leading_term
from .discontinuous_functions import SignedFractionalPart
from .limit_models import LimitEvidence, LimitStatus


def fractional_pole_conflict(expr, variables, target, domain, assumptions):
    """Construct two signed quarter-period values without touching jumps or poles.

    On an admitted affine ray the rational phase must be exactly c*t**(-m).
    Substituting t=(abs(c)/(j+q))**(1/m), q=1/4 or 3/4, attains
    sign(c)*(j+q). Every original denominator has a nonzero leading ray
    term, so the displayed late integer subsequences stay defined.
    """
    if (
        expr.func is not SignedFractionalPart
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 60
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(p.is_Rational is not True for p in target)
    ):
        return None
    phase = expr.args[0]
    if bounded_expansion_width(phase, 64) > 64:
        return None
    num, den = phase.as_numer_denom()
    if any(bounded_degree(part, variables, 8) is None for part in (num, den)):
        return None
    try:
        sp.Poly(num, *variables, domain=sp.QQ)
        sp.Poly(den, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    reduced = sp.cancel(phase)
    numerator, _ = reduced.as_numer_denom()
    directions = [
        tuple(sp.S.One if i == k else sp.S.Zero for i in range(len(variables)))
        for k in range(len(variables))
    ]
    directions.append((sp.S.One,) * len(variables))
    if len(variables) == 2 and bounded_degree(numerator, variables, 1) is not None:
        a, b = (sp.diff(numerator, v) for v in variables)
        if a.is_Rational is True and b.is_Rational is True and (a != 0 or b != 0):
            directions.append((b, -a))
    t = sp.Dummy("fractional_ray_parameter", positive=True)
    j = sp.Dummy("attained_fractional_index", positive=True, integer=True)
    guards = [
        power.base for power in expr.atoms(sp.Pow) if power.exp.is_negative is True
    ]
    for direction in directions:
        mapping = dict(
            zip(
                variables,
                (p + c * t for p, c in zip(target, direction, strict=True)),
                strict=True,
            )
        )
        along = sp.cancel(reduced.subs(mapping, simultaneous=True))
        c, power = along.as_coeff_exponent(t)
        if (
            c.is_Rational is not True
            or c == 0
            or power.is_Integer is not True
            or not -8 <= power < 0
        ):
            continue
        terms = [
            ray_leading_term(g.subs(mapping, simultaneous=True), t) for g in guards
        ]
        if any(term is None or term[0] == 0 for term in terms):
            continue
        evidence = []
        for q in (sp.Rational(1, 4), sp.Rational(3, 4)):
            radius = (sp.Abs(c) / (j + q)) ** (-1 / power)
            sequence = tuple((v, mapping[v].subs(t, radius)) for v in variables)
            evidence.append(
                LimitEvidence(
                    "attained_signed_fractional_phase",
                    "The exact rational phase on this affine ray is c*t**(-m). The displayed positive radius approaches zero and attains sign(c)*(j+q), with q=1/4 or 3/4, away from every integer jump. Nonzero leading original denominator terms exclude poles eventually. Truncation toward zero gives the displayed signed quarter-phase value.",
                    sequence,
                    sp.sign(c) * q,
                )
            )
        return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence)
    return None
