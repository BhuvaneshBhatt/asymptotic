"""Bounded positive-real exponential-integral germs and attained compositions.

Proofs use DLMF 6.6.1/2 and 8.19's positive integral and recurrence.
No complex-cut or accumulating-pole inference is made here.
"""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .special_function_limit_germs import linear_in


def bounded_ei_composition_certificate(expr, x, point, domain, assumptions):
    if point != 0 or sp.count_ops(expr) > 70 or expr.has(sp.Float):
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    atoms = expr.atoms(sp.Ei)
    if not sides or len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    sines = atom.args[0].atoms(sp.sin)
    if len(sines) != 1:
        return None
    sine = next(iter(sines))
    rate = sp.cancel(sine.args[0] * x)
    if (
        rate.has(x)
        or rate.is_finite is not True
        or (rate.is_positive is not True and rate.is_negative is not True)
    ):
        return None
    argument = linear_in(atom.args[0], sine)
    outer = linear_in(expr, atom)
    if argument is None or outer is None:
        return None
    amplitude, center = argument
    a, b = outer
    if (
        center.has(x)
        or amplitude.has(x)
        or center.is_real is not True
        or amplitude.is_real is not True
    ):
        return None
    if (center - sp.Abs(amplitude) - 1).is_nonnegative is not True:
        return None
    if not a.is_rational_function(x) or not b.is_rational_function(x):
        return None
    n = sp.Dummy("attained_ei_n", positive=True, integer=True)
    t = sp.Dummy("positive_ei_chart", positive=True)
    items = []
    for side in sides:
        av = rational_value(a.subs(x, side * t), t, 0)
        bv = rational_value(b.subs(x, side * t), t, 0)
        if (
            bv is None
            or bv.is_real is not True
            or bv.is_finite is not True
            or av is None
        ):
            return None
        if av == 0:
            values = (bv,)
        elif av in (sp.oo, -sp.oo):
            values = (av,)
        elif (
            av.is_real is True
            and av.is_finite is True
            and av.is_zero is False
            and amplitude.is_zero is False
        ):
            values = (
                av * sp.Ei(center + amplitude) + bv,
                av * sp.Ei(center - amplitude) + bv,
            )
        else:
            return None
        for index, value in enumerate(values):
            # Choose the sine phase exactly, including a negative original side.
            phase_sign = side * (1 if rate.is_positive is True else -1)
            phase = sp.pi / 2 if (index == 0) == (phase_sign == 1) else 3 * sp.pi / 2
            seq = side * sp.Abs(rate) / (2 * sp.pi * n + phase)
            items.append(
                LimitEvidence(
                    "attained_bounded_positive_ei_composition",
                    "The argument stays in a compact interval [center-|amplitude|,center+|amplitude|] contained in [1,infinity). Ei is continuous, positive there, and strictly increasing because Ei'(u)=exp(u)/u>0. Thus vanishing rational amplitudes vanish uniformly and signed divergent amplitudes retain their sign. The explicit nonzero sequences attain sine +1 or -1 exactly, avoid x=0 and eventually avoid every rational denominator pole. Finite nonzero amplitudes give distinct attained limits by strict monotonicity.",
                    ((x, seq),),
                    value,
                )
            )
    if len(items) == 1 or all(v.value == items[0].value for v in items):
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)


def expint_exponential_inner_certificate(expr, x, point, domain, assumptions):
    if point != 0 or sp.count_ops(expr) > 70 or expr.has(sp.Float):
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    atoms = expr.atoms(sp.expint)
    if not sides or len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    order, z = atom.args
    if order not in (1, 2) or z.func is not sp.exp:
        return None
    rate = sp.cancel(-z.args[0] * x)
    if (
        rate.has(x)
        or rate.is_finite is not True
        or (rate.is_positive is not True and rate.is_negative is not True)
    ):
        return None
    # Match a fixed affine cancellation family before any algebraic expansion.
    if order == 1:
        if expr != atom - rate / x:
            return None
        values = {1: -sp.EulerGamma, -1: sp.oo}
    else:
        candidates = ((atom - 1) / z, (atom - 1) / z + rate / x)
        matched = None
        for k, candidate in enumerate(candidates):
            if expr == candidate or sp.cancel(expr - candidate) == 0:
                matched = k
                break
        if matched is None:
            return None
        values = (
            {1: -sp.oo, -1: sp.S.Zero}
            if matched == 0
            else {1: sp.EulerGamma - 1, -1: -sp.oo}
        )
    if rate.is_negative is True:
        values = {side: values[-side] for side in (1, -1)}
    n = sp.Dummy("attained_expint_n", positive=True, integer=True)
    items = tuple(
        LimitEvidence(
            "positive_expint_exponential_inner_germ",
            "For positive z, E1(z)=-EulerGamma-log(z)+O(z) at zero, and 0<E1(z)<=exp(-z)/z at infinity. The exact recurrence E2(z)=exp(-z)-z*E1(z) and (exp(-z)-1)/z=-1+O(z) retain the constant after cancellation. Here z=exp(-rate/x)>0, so log(z)=-rate/x exactly. Along x=+/-1/n the inner variable tends monotonically to zero/infinity, stays away from the principal cut, and never vanishes; all original denominators are nonzero.",
            ((x, side / n),),
            values[side],
        )
        for side in sides
    )
    if len(items) == 1 or items[0].value == items[1].value:
        return LimitStatus.PROVED, items[0].value, items
    return LimitStatus.DOES_NOT_EXIST, None, items
