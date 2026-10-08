"""Checked real-side Gaussian germs and vertical exponential-integral tails."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .local_tail_germs import budget


def gaussian_root_side_certificate(expr, x, point, domain, assumptions):
    """Separate the real and imaginary Gaussian scales at a square-root pole."""
    if point != 0 or not expr.has(sp.erfi) or not budget(expr, 60):
        return None
    atoms = expr.atoms(sp.erfi)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    argument = atom.args[0]
    coefficient = sp.cancel(argument * sp.sqrt(x))
    if coefficient.has(x) or coefficient.is_finite is not True:
        return None
    if expr != sp.sqrt(sp.pi) * argument * sp.exp(-(argument**2)) * atom:
        return None
    real = coefficient.is_real is True and coefficient.is_zero is False
    imaginary = coefficient.is_imaginary is True and coefficient.is_zero is False
    if not real and not imaginary:
        return None
    sides = _sides(x, point, domain, assumptions)
    if not sides:
        return None
    n = sp.Dummy("gaussian_index", integer=True, positive=True)
    items = []
    for side in sides:
        value = sp.S.One if (side > 0) == real else -sp.oo
        items.append(
            LimitEvidence(
                "attained_gaussian_root_sides",
                "On x=+/-1/n**2 the argument is exactly a nonzero real or imaginary Gaussian scale. For real y->+/-infinity, sqrt(pi)*y*exp(-y**2)*erfi(y)->1 with error O(y**-2). For y=i*t, erfi(i*t)=i*erf(t), so the expression is -sqrt(pi)*t*exp(t**2)*erf(t)->-infinity on either imaginary orientation. The principal square root fixes which real side has each scale; the attained sequences avoid x=0.",
                ((x, side / n**2),),
                value,
            )
        )
    if len({item.value for item in items}) > 1:
        return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
    return LimitStatus.PROVED, items[0].value, tuple(items)


def vertical_expint_certificate(expr, x, point, domain, assumptions):
    """Use a fixed-order sector expansion on an affine vertical argument."""
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or x.is_real is False
        or assumptions is sp.S.false
        or assumptions.has(x)
        or expr.func is not sp.expint
        or not budget(expr, 40)
    ):
        return None
    replacements = {}
    for clause in sp.And.make_args(assumptions):
        if (
            isinstance(clause, sp.StrictGreaterThan)
            and clause.lhs.is_Symbol
            and clause.rhs == 0
        ):
            replacements[clause.lhs] = sp.Dummy(
                "positive_integral_parameter", positive=True, finite=True
            )
    order, argument = expr.xreplace(replacements).args
    if order.is_Integer is not True or not 1 <= order <= 16:
        return None
    slope = sp.diff(argument, x)
    offset = argument.subs(x, 0)
    if (
        slope.has(x)
        or sp.expand(argument - offset - slope * x) != 0
        or slope.is_imaginary is not True
        or slope.is_zero is not False
        or slope.is_finite is not True
        or offset.is_finite is not True
    ):
        return None
    n = sp.Dummy("vertical_index", integer=True, positive=True)
    return (
        LimitStatus.PROVED,
        sp.S.Zero,
        LimitEvidence(
            "vertical_exponential_integral_tail",
            "For fixed integer order and z=b+i*a*x with fixed finite b and nonzero real a, the principal E_p sector expansion is exp(-z)/z*(1+O(1/z)). The tail lies in a closed sector around +/-pi/2, away from the cut and zero. Re(z)=Re(b) is fixed, so its modulus is O(1/x) and the limit is zero. The sequence x=n is attained and eventually avoids the cut. Explicit positive parameter assumptions certify the slope and offset used here.",
            ((x, n),),
            sp.S.Zero,
        ),
    )
