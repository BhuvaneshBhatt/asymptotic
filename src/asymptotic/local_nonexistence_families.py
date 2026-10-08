"""Real local poles, reciprocal exponentials and attained tangent tails."""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .special_function_limit_germs import linear_in


def _sides(x, p, domain, assumptions):
    if x.is_integer is True or x.is_real is False or x.is_finite is False:
        return ()
    if p.is_real is not True or p.is_finite is not True or p.has(x):
        return ()
    if assumptions is sp.S.false or assumptions.has(x):
        return ()
    if domain not in (sp.S.true, sp.Gt(x, p)):
        return ()
    if (x.is_nonnegative is True and p.is_negative is True) or (
        x.is_nonpositive is True and p.is_positive is True
    ):
        return ()
    if domain == sp.Gt(x, p):
        return () if p == 0 and x.is_nonpositive is True else (1,)
    if p == 0 and x.is_nonnegative is True:
        return (1,)
    if p == 0 and x.is_nonpositive is True:
        return (-1,)
    return (1, -1)


def local_rational_pole_certificate(expr, x, point, domain, assumptions):
    p = sp.sympify(point)
    sides = _sides(x, p, domain, assumptions)
    if not sides or sp.count_ops(expr) > 65 or not expr.is_rational_function(x):
        return None
    if any(v.exp.is_Integer and abs(v.exp) > 8 for v in expr.atoms(sp.Pow)):
        return None
    t = sp.Dummy("real_pole_chart", positive=True)
    try:
        num, den = (sp.Poly(v, t) for v in sp.fraction(sp.cancel(expr.subs(x, p + t))))
    except sp.PolynomialError:
        return None
    if num.is_zero or den.is_zero or max(num.degree(), den.degree()) > 8:
        return None
    ni, nc = min(num.terms())
    di, dc = min(den.terms())
    power = ni[0] - di[0]
    c = nc / dc
    if (
        power >= 0
        or c.is_finite is not True
        or (c.is_positive is not True and c.is_negative is not True)
    ):
        return None
    # Even two-sided poles are left to the existing infinity contract.
    if len(sides) == 2 and power % 2 == 0:
        return None
    n = sp.Dummy("attained_pole_n", positive=True, integer=True)
    items = []
    for side in sides:
        value = sp.oo if c * side**power > 0 else -sp.oo
        items.append(
            LimitEvidence(
                "attained_rational_pole_sides",
                "The rational Laurent leading term has certified real nonzero coefficient and odd pole order. x=p+/-1/n attains the signed divergent side limits. Nonzero polynomial denominators have finitely many zeros and these sequences eventually avoid every pole.",
                ((x, p + side / n),),
                value,
            )
        )
    if len(items) == 1:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)


def reciprocal_exponential_sides_certificate(expr, x, point, domain, assumptions):
    if point != 0 or sp.count_ops(expr) > 80:
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    if not sides:
        return None
    candidates = []
    for atom in expr.atoms(sp.exp, sp.Pow):
        if atom.func is sp.exp:
            exponent = atom.args[0]
            base = sp.E
        else:
            base, exponent = atom.args
        if (
            base.has(x)
            or base.is_positive is not True
            or base.is_finite is not True
            or base == 1
        ):
            continue
        c = sp.cancel(exponent * x)
        if c.has(x) or c.is_finite is not True:
            continue
        rate = c * sp.log(base)
        if rate.is_positive is True or rate.is_negative is True:
            candidates.append((atom, rate))
    if len(candidates) != 1:
        return None
    atom, rate = candidates[0]
    w = sp.Dummy("positive_exponential_value", positive=True)
    reduced = expr.xreplace({atom: w})
    if reduced.has(x) or not reduced.is_rational_function(w):
        return None
    if any(p.exp.is_Integer and abs(p.exp) > 8 for p in reduced.atoms(sp.Pow)):
        return None
    n = sp.Dummy("attained_exponential_n", positive=True, integer=True)
    items = []
    for side in sides:
        endpoint = sp.oo if side * rate > 0 else sp.S.Zero
        value = rational_value(reduced, w, endpoint)
        if value is None or value.is_real is not True or value.is_finite is not True:
            return None
        items.append(
            LimitEvidence(
                "attained_reciprocal_exponential_sides",
                "For a positive real constant base, the principal power equals exp(rate/x). Along x=+/-1/n it attains exp(+/-rate*n), tending to infinity or zero. The rational outer denominator is a nonzero polynomial; the monotone attained values eventually avoid its finitely many roots.",
                ((x, side / n),),
                value,
            )
        )
    if len(items) == 1 or items[0].value == items[1].value:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    if (items[0].value - items[1].value).is_zero is False:
        return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
    return None


def attained_tangent_pole_certificate(expr, x, point, domain, assumptions):
    if point != 0 or sp.count_ops(expr) > 65:
        return None
    sides = _sides(x, sp.S.Zero, domain, assumptions)
    atoms = expr.atoms(sp.tan)
    if not sides or len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    c = sp.cancel(atom.args[0] * x)
    if (
        c.has(x)
        or c.is_finite is not True
        or (c.is_positive is not True and c.is_negative is not True)
    ):
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    constant = None
    for k in range(1, 5):
        coefficient = sp.cancel(a / x**k)
        if (
            not coefficient.has(x)
            and coefficient.is_finite is True
            and coefficient.is_zero is False
        ):
            constant = coefficient
            break
    if constant is None:
        return None
    t = sp.Dummy("positive_tangent_chart", positive=True)
    side = sides[0]
    baseline = rational_value(b.subs(x, side * t), t, 0)
    if baseline is None or baseline.is_finite is not True:
        return None
    n = sp.Dummy("attained_tangent_n", positive=True, integer=True)
    sequences = (
        side * sp.Abs(c) / (sp.pi * n),
        side * sp.Abs(c) / (sp.pi * n + sp.pi / 2 - 1 / n**k),
    )
    sign = 1 if c.is_positive is True else -1
    values = (
        baseline,
        baseline + constant * side ** (k + 1) * sign * (sp.Abs(c) / sp.pi) ** k,
    )
    items = tuple(
        LimitEvidence(
            "attained_tangent_pole_avoidance",
            "x=side*|c|/(pi*n) attains tangent zero. The second sequence attains phase +/- (pi*n+pi/2-1/n^k); tan is +/-cot(1/n^k), so the scaled term has the stated nonzero limit. Its cosine denominator is +/-sin(1/n^k), strictly nonzero for every positive integer n. Other rational denominators have finitely many roots and are eventually avoided.",
            ((x, sequence),),
            value,
        )
        for sequence, value in zip(sequences, values)
    )
    return LimitStatus.DOES_NOT_EXIST, None, items


def local_analytic_pole_certificate(expr, x, point, domain, assumptions):
    p = sp.sympify(point)
    sides = _sides(x, p, domain, assumptions)
    if not sides or expr.has(sp.Float) or sp.count_ops(expr) > 70:
        return None
    if expr.is_rational_function(x):
        return None
    if any(q.exp.is_Integer and abs(q.exp) > 6 for q in expr.atoms(sp.Pow)):
        return None
    replacements = {}
    for atom in expr.atoms(sp.tan, sp.cot, sp.sec, sp.csc):
        u = atom.args[0]
        replacements[atom] = {
            sp.tan: sp.sin(u) / sp.cos(u),
            sp.cot: sp.cos(u) / sp.sin(u),
            sp.sec: 1 / sp.cos(u),
            sp.csc: 1 / sp.sin(u),
        }[atom.func]
    transformed = expr.xreplace(replacements)
    supported = {sp.sin, sp.cos, sp.exp, sp.log, sp.atan}
    for atom in transformed.atoms(sp.Function):
        if atom.func not in supported:
            return None
        center = atom.args[0].subs(x, p)
        if center.is_finite is not True:
            return None
        if atom.func is sp.log and center.is_positive is not True:
            return None
        if atom.func is sp.atan and center.is_real is not True:
            return None
    for power in transformed.atoms(sp.Pow):
        if power.exp.is_Integer is True:
            continue
        if not power.exp.is_Rational or power.exp.q > 4:
            return None
        center = power.base.subs(x, p)
        if center.is_positive is not True or center.is_finite is not True:
            return None
    num, den = sp.fraction(sp.together(transformed))
    if max(sp.count_ops(num), sp.count_ops(den)) > 90:
        return None

    def first_coefficient(part):
        derivative = part
        for k in range(5):
            c = sp.simplify(derivative.subs(x, p)) / sp.factorial(k)
            if c.is_zero is True:
                derivative = sp.diff(derivative, x)
                if sp.count_ops(derivative) > 220:
                    return None
                continue
            if c.is_finite is True and c.is_zero is False:
                return k, c
            return None
        return None

    a, b = first_coefficient(num), first_coefficient(den)
    if a is None or b is None:
        return None
    order = a[0] - b[0]
    c = sp.simplify(a[1] / b[1])
    if order >= 0 or (len(sides) == 2 and order % 2 == 0):
        return None
    if c.is_finite is not True or (
        c.is_positive is not True and c.is_negative is not True
    ):
        return None
    n = sp.Dummy("attained_analytic_pole_n", positive=True, integer=True)
    items = []
    for side in sides:
        value = sp.oo if (c * sp.Integer(side) ** order).is_positive is True else -sp.oo
        items.append(
            LimitEvidence(
                "attained_analytic_pole_sides",
                "Exact derivatives through order four identify the first nonzero analytic numerator and denominator coefficients. All earlier coefficients are proved zero. Trigonometric reciprocals are identities on their common domain, roots have positive finite centers and logarithms stay off their cut. The real Laurent leading coefficient fixes each signed pole limit at x=p+/-1/n. Nonzero analytic germs have isolated zeros, so original denominators are eventually avoided.",
                ((x, p + side / n),),
                value,
            )
        )
    if len(items) == 1:
        return LimitStatus.PROVED, items[0].value, tuple(items)
    return LimitStatus.DOES_NOT_EXIST, None, tuple(items)
