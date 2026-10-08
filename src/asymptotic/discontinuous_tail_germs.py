"""Bounded periodic tails with interior attained subsequences."""

import sympy as sp

from .analytic_limits import SquareWave
from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .special_function_limit_germs import linear_in


def _finite_tail(expr, x):
    if not expr.has(x):
        return expr if expr.is_finite is True else None
    if expr.is_rational_function(x):
        value = rational_value(expr, x, sp.oo)
        return value if value is not None and value.is_finite is True else None
    if expr.func is sp.atan and expr.args[0].is_rational_function(x):
        value = rational_value(expr.args[0], x, sp.oo)
        return sp.atan(value) if value is not None else None
    if expr.is_Pow and expr.exp.is_Integer and expr.exp < 0:
        base = expr.base
        if base.func is sp.log and base.args[0].is_rational_function(x):
            if rational_value(base.args[0], x, sp.oo) is sp.oo:
                return sp.S.Zero
    if expr.is_Add or expr.is_Mul:
        values = [_finite_tail(arg, x) for arg in expr.args]
        if all(v is not None for v in values):
            value = expr.func(*values)
            return value if value.is_finite is True else None
    return None


def discontinuous_periodic_tail_certificate(expr, x, point, domain, assumptions):
    """Certify bounded wave decay or distinct limits along affine phase sequences.

    Only interior quarter-period phases are used. Rational coefficients have
    finitely many poles, and reciprocal logarithms are defined on a late tail.
    The coefficient grammar excludes other discontinuities and oscillations.
    """
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_real is False
        or x.is_integer is True
        or sp.count_ops(expr) > 55
    ):
        return None
    waves = list(expr.atoms(SquareWave))
    parities = [
        atom
        for atom in expr.atoms(sp.Pow)
        if atom.base == -1 and atom.exp.func is sp.floor
    ]
    if len(waves) + len(parities) != 1:
        return None
    wave = (waves + parities)[0]
    phase = wave.args[0] if waves else wave.exp.args[0]
    try:
        polynomial = sp.Poly(phase, x)
    except sp.PolynomialError:
        return None
    if polynomial.degree() != 1:
        return None
    slope, offset = polynomial.nth(1), polynomial.nth(0)
    if (
        slope.is_finite is not True
        or offset.is_finite is not True
        or offset.is_real is not True
        or not (slope.is_positive is True or slope.is_negative is True)
    ):
        return None
    coefficients = linear_in(expr, wave)
    if coefficients is None:
        return None
    amplitude, baseline = map(lambda e: _finite_tail(e, x), coefficients)
    if amplitude is None or baseline is None:
        return None
    if amplitude == 0:
        return (
            LimitStatus.PROVED,
            baseline,
            LimitEvidence(
                "bounded_discontinuous_wave_tail",
                "The real wave has modulus at most one. Its coefficient tends to zero and the remaining term has a finite limit. Rational poles and logarithmic zeros are eventually avoided.",
                value=baseline,
            ),
        )
    if amplitude.is_zero is not False:
        return None
    n = sp.Dummy("periodic_tail_n", positive=True, integer=True)
    orientation = 1 if slope.is_positive is True else -1
    period = 1 if waves else 2
    phases = (sp.Rational(1, 4), sp.Rational(3, 4) if waves else sp.Rational(5, 4))
    witnesses = tuple(
        LimitEvidence(
            "attained_discontinuous_periodic_tail",
            "The affine phase equals an integer number of periods plus an interior quarter-period offset. The sequence tends to +infinity, avoids every wave jump, and attains the stated sign exactly. Coefficient limits exist; rational poles and logarithmic zeros are eventually avoided.",
            ((x, (orientation * period * n + theta - offset) / slope),),
            baseline + sign * amplitude,
        )
        for theta, sign in zip(phases, (1, -1))
    )
    return LimitStatus.DOES_NOT_EXIST, None, witnesses
