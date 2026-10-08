"""Attained periodic approaches for affine waves and hyperbolic poles."""

import sympy as sp

from ._polynomial_bounds import bounded_degree
from .analytic_limits import SquareWave
from .limit_models import LimitEvidence, LimitStatus


def square_wave_infinite_conflict(expr, variables, target, domain, assumptions):
    """Construct two open-half-period subsequences at real coordinate infinities."""
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or any(p not in (sp.oo, -sp.oo) for p in target)
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or expr.free_symbols - set(variables)
        or sp.count_ops(expr) > 30
    ):
        return None
    atoms = expr.atoms(SquareWave)
    if len(atoms) != 1:
        return None
    wave = next(iter(atoms))
    if len(wave.args) != 1:
        return None
    try:
        phase = sp.Poly(wave.args[0], *variables, domain=sp.QQ)
        z = sp.Dummy("wave_value")
        outer = sp.Poly(expr.xreplace({wave: z}), z, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    if phase.total_degree() != 1 or outer.degree() != 1:
        return None
    a, b = outer.all_coeffs()
    if a == 0:
        return None
    signs = tuple(1 if p is sp.oo else -1 for p in target)
    # These positive rays span coordinate space. A nonconstant affine phase
    # therefore has a nonzero slope on at least one of them.
    weights_to_try = [(1,) * len(variables)] + [
        tuple(1 + (i == j) for i in range(len(variables)))
        for j in range(len(variables))
    ]
    for weights in weights_to_try:
        ray = tuple(s * w for s, w in zip(signs, weights, strict=True))
        slope = sum(
            phase.coeff_monomial(v) * c for v, c in zip(variables, ray, strict=True)
        )
        if slope == 0:
            continue
        orientation = sp.sign(slope)
        j = sp.Dummy("attained_wave_index", positive=True, integer=True)
        items = []
        for fraction, value in [
            (sp.Rational(1, 4), a + b),
            (sp.Rational(3, 4), -a + b),
        ]:
            t = (orientation * j + fraction - phase.TC()) / slope
            sequence = tuple((v, c * t) for v, c in zip(variables, ray, strict=True))
            items.append(
                LimitEvidence(
                    "attained_wave_half_period",
                    "The affine phase is an integer plus 1/4 or 3/4. The exact period-one definition gives opposite attained values. Each coordinate tends to its declared real infinity, and the phases avoid every jump.",
                    sequence,
                    value,
                )
            )
        return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
    return None


def hyperbolic_pole_conflict(expr, variables, target, domain, assumptions):
    """Separate a real exponential approach from an attained imaginary period.

    For sech(b+a/(u+i*v))**k, a is a nonzero real constant and k is a
    positive integer. A real ray gives zero; u=0,v=a/(2*pi*j) gives
    sech(b)**k exactly. Cosh is nonzero on both sequences.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) != 2
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 30
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(p.is_Rational is not True for p in target)
    ):
        return None
    base, power = expr.as_base_exp()
    if base.func is not sp.sech or power.is_Integer is not True or not 1 <= power <= 4:
        return None
    x, y = variables
    try:
        num, den = base.args[0].as_numer_denom()
        if (
            bounded_degree(num, variables, 1) is None
            or bounded_degree(den, variables, 1) is None
        ):
            return None
        n = sp.Poly(num, x, y, domain=sp.QQ_I)
        d = sp.Poly(den, x, y, domain=sp.QQ_I)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    if n.total_degree() > 1 or d.total_degree() != 1:
        return None
    dx = d.coeff_monomial(x)
    nx = n.coeff_monomial(x)
    if (
        dx.is_Rational is not True
        or dx == 0
        or d.coeff_monomial(y) != sp.I * dx
        or n.coeff_monomial(y) != sp.I * nx
    ):
        return None
    point = dict(zip(variables, target, strict=True))
    if d.as_expr().subs(point) != 0:
        return None
    a = n.as_expr().subs(point) / dx
    b = nx / dx
    if a.is_Rational is not True or a == 0 or b.is_Rational is not True:
        return None
    j = sp.Dummy("attained_hyperbolic_index", positive=True, integer=True)
    real_sequence = ((x, target[0] + 1 / j), (y, target[1]))
    periodic_sequence = ((x, target[0]), (y, target[1] + a / (2 * sp.pi * j)))
    return (
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "attained_hyperbolic_real_ray",
                "The phase is b+a*j, real and unbounded. Cosh is strictly positive on this ray, and the original affine denominator is nonzero. The hyperbolic reciprocal tends to zero.",
                real_sequence,
                sp.S.Zero,
            ),
            LimitEvidence(
                "attained_hyperbolic_period",
                "The phase is b-2*pi*I*j. Periodicity gives sech(b)**k exactly; cosh(b) is strictly positive for real b, so every point avoids the accumulating pole set and the original affine denominator.",
                periodic_sequence,
                sp.sech(b) ** power,
            ),
        ),
    )
