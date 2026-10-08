"""Principal-root charts and finite attained argument boundaries."""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides


def signed_radical_chart_certificate(expr, x, point, domain, assumptions):
    p = sp.sympify(point)
    sides = _sides(x, p, domain, assumptions)
    roots = [
        q
        for q in expr.atoms(sp.Pow)
        if q.exp.is_Rational and q.exp.is_Integer is not True
    ]
    if (
        not sides
        or not roots
        or len(roots) > 3
        or sp.count_ops(expr) > 65
        or expr.has(sp.Float)
    ):
        return None
    if any(q.exp.q > 4 or abs(q.exp) > 8 for q in roots):
        return None
    t = sp.Dummy("positive_root_chart", positive=True)
    n = sp.Dummy("attained_root_n", positive=True, integer=True)
    items = []
    for side in sides:
        chart = expr.subs(x, p + side * t)
        if not chart.is_rational_function(t):
            return None
        value = rational_value(chart, t, 0)
        if value is None or value.is_finite is not True:
            return None
        items.append(
            LimitEvidence(
                "attained_principal_root_charts",
                "Substitution x=p+/-t with t>0 preserves the principal powers and reduces this expression exactly to a rational chart. Finite rational side limits are attained at x=p+/-1/n. Rational denominators have finitely many zeros and original removable zeros are eventually avoided; no forced power denesting is used.",
                ((x, p + side / n),),
                value,
            )
        )
    if len(items) == 1 or items[0].value == items[1].value:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    if sp.simplify(items[0].value - items[1].value).is_zero is False:
        return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
    return None


def reciprocal_root_argument_certificate(expr, x, point, domain, assumptions):
    if point != 0 or expr.func is not sp.arg or sp.count_ops(expr) > 35:
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    if not sides:
        return None
    z = expr.args[0]
    roots = [q for q in z.atoms(sp.Pow) if q.exp == sp.Rational(1, 2) and q.base.has(x)]
    if len(roots) != 1:
        return None
    root = roots[0]
    n = sp.Dummy("attained_argument_n", positive=True, integer=True)
    items = []
    c = sp.cancel(root.base * x)
    d = sp.cancel(sp.I * (z + root))
    if (
        not c.has(x)
        and c.is_positive is True
        and c.is_finite is True
        and not d.has(x)
        and d.is_real is True
        and d.is_finite is True
        and (d.is_positive is True or d.is_negative is True or d == 0)
    ):
        for side in sides:
            value = (
                (-sp.pi if d.is_positive is True else sp.pi)
                if side == 1
                else -sp.pi / 2
            )
            items.append(
                LimitEvidence(
                    "attained_reciprocal_root_argument",
                    "Along x=1/n the principal root sqrt(c/x) is positive real and unbounded, selecting the negative-real argument boundary according to the fixed imaginary offset. Along x=-1/n the root is +i*sqrt(c*n) and the argument is eventually exactly -pi/2. These attained sequences avoid x=0 and their arguments are eventually nonzero.",
                    ((x, side / n),),
                    value,
                )
            )
    else:
        center = root.base.subs(x, 0)
        coefficient = sp.cancel((root.base - center) / x**2)
        if (
            center.is_positive is not True
            or center.is_finite is not True
            or coefficient.has(x)
            or coefficient.is_positive is not True
            or coefficient.is_finite is not True
        ):
            return None
        if sp.expand(z - sp.I * (root - sp.sqrt(center))) != 0:
            return None
        for side in sides:
            items.append(
                LimitEvidence(
                    "attained_positive_imaginary_root_increment",
                    "At x=+/-1/n the positive real root sqrt(c+a*x^2) strictly exceeds sqrt(c), so its imaginary increment is nonzero and has principal argument pi/2 exactly. Both sequences belong to the original domain.",
                    ((x, side / n),),
                    sp.pi / 2,
                )
            )
    if len(items) == 1 or items[0].value == items[1].value:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
