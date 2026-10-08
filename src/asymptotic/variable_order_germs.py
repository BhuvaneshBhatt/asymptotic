"""Bounded joint-continuity certificates; no large-order asymptotics."""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus


def regular_variable_bessel_certificate(expr, x, point, domain, assumptions):
    atoms = tuple(
        sorted(
            expr.atoms(sp.besselj, sp.bessely, sp.besseli, sp.besselk),
            key=sp.default_sort_key,
        )
    )
    if not atoms or len(atoms) > 3 or sp.count_ops(expr) > 90:
        return None
    if not any(a.args[0].has(x) for a in atoms):
        return None
    if assumptions is sp.S.false or domain is sp.S.false:
        return None
    if x.is_integer is True or x.is_finite is False:
        return None
    if domain is not sp.S.true and domain != sp.Gt(x, point):
        return None
    p = sp.sympify(point)
    if p.is_finite is True and (
        (x.is_real is True and p.is_real is False)
        or (x.is_nonnegative is True and p.is_negative is True)
        or (x.is_nonpositive is True and p.is_positive is True)
    ):
        return None
    t = sp.Dummy("regular_bessel_chart", positive=True)
    if point is sp.oo:
        chart = 1 / t
    elif point is -sp.oo:
        chart = -1 / t
    elif sp.sympify(point).is_finite is True:
        chart = point + t
    else:
        return None

    def value(e):
        e = e.subs(x, chart)
        if not e.is_rational_function(t) or sp.count_ops(e) > 24:
            return None
        if any(p.exp.is_Integer and abs(p.exp) > 8 for p in e.atoms(sp.Pow)):
            return None
        return rational_value(e, t, 0)

    # The fixed-argument large-order theorem does not license a varying
    # argument or cancellation/amplification of its remainder.
    if len(atoms) == 1 and expr == atoms[0]:
        atom = atoms[0]
        nu, z = atom.args
        on_positive_side = (
            point is sp.oo
            or (point == 0 and x.is_nonnegative is True)
            or domain == sp.Gt(x, point)
        )
        if (
            on_positive_side
            and not z.has(x)
            and z.is_positive is True
            and z.is_finite is True
            and value(nu) is sp.oo
        ):
            chart_order = nu.subs(x, chart)
            if chart_order.is_real is True:
                result = {
                    sp.besselj: sp.S.Zero,
                    sp.besseli: sp.S.Zero,
                    sp.bessely: -sp.oo,
                    sp.besselk: sp.oo,
                }[atom.func]
                return (
                    LimitStatus.PROVED,
                    result,
                    LimitEvidence(
                        "fixed_argument_positive_large_bessel_order",
                        "DLMF 10.19(i) and 10.41(i): fixed positive finite argument, real order tending to +infinity. The fixed-argument leading forms give J,I->0 and Y->-infinity, K->+infinity. No uniformity for a varying argument or amplified remainder is inferred.",
                        value=result,
                    ),
                )
    replacements = {}
    for atom in atoms:
        order, argument = map(value, atom.args)
        if order is None or argument is None:
            return None
        if (
            order.is_finite is not True
            or argument.is_positive is not True
            or argument.is_finite is not True
        ):
            return None
        replacements[atom] = atom.func(order, argument)
    # Polynomial outer compositions are continuous; singular denominators
    # depending on a Bessel value require a separate cancellation theorem.
    markers = [sp.Dummy("regular_bessel_value") for _ in atoms]
    transformed = expr.xreplace(dict(zip(atoms, markers)))
    try:
        polynomial = sp.Poly(transformed, *markers)
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() > 4:
        return None
    result = 0
    for powers, coefficient in polynomial.terms():
        c = value(coefficient)
        if c is None or c.is_finite is not True:
            return None
        result += c * sp.Mul(*(replacements[a] ** k for a, k in zip(atoms, powers)))
    if result.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return None
    return (
        LimitStatus.PROVED,
        result,
        LimitEvidence(
            "regular_variable_order_bessel_continuity",
            "DLMF 10.2 and 10.25: at fixed positive nonzero argument J,Y,I,K are jointly continuous in finite order and argument, including removable integer-order definitions of Y and K. Rational order/argument germs converge into a neighborhood disjoint from zero and the negative-real cut. Polynomial outer compositions and finite rational coefficients preserve continuity; their polynomial denominator zeros are eventually avoided. The same rational limits hold on both finite-target sides.",
            value=result,
        ),
    )
