"""Small reciprocal-principal-Lambert-W germs on fixed off-cut rays."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus


def reciprocal_lambert_germ_certificate(expr, x, point, domain, assumptions):
    point = sp.sympify(point)
    if point.func.__name__ == "DirectionalInfinity" and len(point.args) == 1:
        ray = point.args[0]
        if (
            domain is not sp.S.true
            or ray.has(x)
            or ray.free_symbols
            or ray.is_finite is not True
            or ray.is_zero is not False
        ):
            return None
        if (
            x.is_integer is True
            or x.is_finite is False
            or (x.is_real is True and ray.is_real is not True)
        ):
            return None
        if (x.is_real is False and ray.is_real is True) or (
            x.is_imaginary is True and ray.is_imaginary is not True
        ):
            return None
        if (x.is_nonnegative is True and ray.is_positive is not True) or (
            x.is_nonpositive is True and ray.is_negative is not True
        ):
            return None
        t = sp.Dummy("positive_lambert_ray_tail", positive=True)
        return reciprocal_lambert_germ_certificate(
            expr.subs(x, ray * t), t, sp.oo, sp.S.true, assumptions.subs(x, ray * t)
        )
    atoms = expr.atoms(sp.LambertW)
    if not atoms or len(atoms) != 1 or point is not sp.oo or x.is_positive is not True:
        return None
    if (
        domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or sp.count_ops(expr) > 60
        or expr.has(sp.Float)
    ):
        return None
    atom = next(iter(atoms))
    if len(atom.args) > 1 and atom.args[1] != 0:
        return None
    c = sp.cancel(atom.args[0] / x)
    if c.has(x) or c.free_symbols or c.is_finite is not True or c.is_zero is not False:
        return None
    if c.is_real is True and c.is_positive is not True:
        return None
    if c.is_real not in (True, False):
        return None
    u = sp.Dummy("small_reciprocal_lambert")
    chart = expr.xreplace({atom: 1 / u})
    if chart.has(x):
        return None
    pole_bound = 0
    for p in chart.atoms(sp.Pow):
        if p.base == u and p.exp.is_Integer:
            pole_bound += max(0, -int(p.exp))
            continue
        if p.exp.has(u) or not p.exp.is_Rational or abs(p.exp) > 8:
            return None
        if p.base.has(u) and p.base.subs(u, 0) != 1:
            return None
    if pole_bound > 6:
        return None
    for a in chart.atoms(sp.Function):
        if a.func not in (sp.sin, sp.cos, sp.sinh, sp.cosh, sp.exp, sp.log):
            return None
        arg = a.args[0]
        if not arg.is_polynomial(u) or sp.Poly(arg, u).degree() > 4:
            return None
        if arg.subs(u, 0) != (1 if a.func is sp.log else 0):
            return None
    try:
        series = sp.series(chart, u, 0, 2).removeO().expand()
    except (ValueError, NotImplementedError, sp.PoleError):
        return None
    if any(p.base == u and p.exp.is_negative for p in series.atoms(sp.Pow)):
        return None
    value = series.subs(u, 0)
    if value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or value.is_finite is not True:
        return None
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "reciprocal_principal_lambert_analytic_germ",
            "On the fixed nonzero ray z=c*x away from the negative-real principal cut, W0(z)=Log(z)-Log(Log(z))+o(1), so |W0(z)| tends to infinity and u=1/W0(z) tends to zero. The outer expression is meromorphic in u with pole budget at most six; its bounded Taylor computation removes all negative powers and retains the constant with an O(u) remainder. Log and rational-power bases equal one at u=0, fixing their principal analytic germs. W is eventually nonzero and all remaining analytic denominators stay nonzero.",
            value=value,
        ),
    )
