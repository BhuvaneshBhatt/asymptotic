"""Attained periodic witnesses and principal-root phase cancellation."""

import sympy as sp

from .elementary_limit_germs import rational_value
from .limit_models import LimitEvidence, LimitStatus
from .special_function_limit_germs import linear_in


def periodic_composition_certificate(expr, x, point, domain, assumptions):
    if (
        point is not sp.oo
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_real is False
        or x.is_integer is True
        or sp.count_ops(expr) > 80
    ):
        return None
    primitive = []
    for atom in expr.atoms(sp.sin, sp.cos):
        if not atom.args[0].has(x) or sp.count_ops(atom.args[0]) > 15:
            continue
        try:
            p = sp.Poly(atom.args[0], x)
        except sp.PolynomialError:
            continue
        if p.degree() == 1:
            primitive.append(atom)
    if not primitive:
        return None
    phase = primitive[0].args[0]
    p = sp.Poly(phase, x)
    a, b = p.nth(1), p.nth(0)
    if (
        a.is_finite is not True
        or b.is_finite is not True
        or b.is_real is not True
        or (a.is_positive is not True and a.is_negative is not True)
    ):
        return None
    if any(atom.args[0] != phase for atom in primitive):
        return None

    def finite_value(e):
        if not e.has(x):
            return e if e.is_finite is True and e.is_real is True else None
        if e.is_rational_function(x):
            if any(p.exp.is_Integer and abs(p.exp) > 8 for p in e.atoms(sp.Pow)):
                return None
            v = rational_value(e, x, sp.oo)
            return (
                v
                if v is not None and v.is_finite is True and v.is_real is True
                else None
            )
        if e.func not in (
            sp.Add,
            sp.Mul,
            sp.Pow,
            sp.Min,
            sp.Max,
            sp.sin,
            sp.cos,
            sp.sinh,
            sp.cosh,
            sp.exp,
            sp.atan,
            sp.Abs,
            sp.log,
        ):
            return None
        values = [finite_value(arg) for arg in e.args]
        if any(v is None for v in values):
            return None
        if e.func is sp.log and values[0].is_positive is not True:
            return None
        if (
            e.func is sp.Pow
            and values[1].is_Integer is not True
            and values[0].is_positive is not True
        ):
            return None
        v = e.func(*values)
        return v if v.is_finite is True and v.is_real is True else None

    n = sp.Dummy("attained_periodic_n", positive=True, integer=True)
    orientation = 1 if a.is_positive is True else -1
    items = []
    for theta in (sp.S.Zero, sp.pi / 2, sp.pi / 4, sp.pi):
        collapsed = expr.xreplace({atom: atom.func(theta) for atom in primitive})
        value = finite_value(collapsed)
        if value is None:
            continue
        sequence = (orientation * 2 * sp.pi * n + theta - b) / a
        item = LimitEvidence(
            "attained_affine_periodic_composition",
            "This real sequence tends to +infinity and attains affine phase +/-2*pi*n+theta exactly. Periodicity fixes the inner sine/cosine values. Finite rational limits and continuous outer functions give the stated limit. Each specialized rational denominator is a nonzero polynomial and is eventually avoided; logarithms and noninteger powers have positive limiting arguments.",
            ((x, sequence),),
            value,
        )
        for old in items:
            if sp.simplify(old.value - value).is_zero is False:
                return LimitStatus.DOES_NOT_EXIST, None, (old, item)
        items.append(item)
    return None


def imaginary_root_phase_cancellation_certificate(expr, x, point, domain, assumptions):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or domain is not sp.S.true
        or assumptions is sp.S.false
        or assumptions.has(x)
        or sp.count_ops(expr) > 55
    ):
        return None
    atoms = expr.atoms(sp.sinh)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    root = atom.args[0]
    if not root.is_Pow or root.exp != sp.Rational(1, 2):
        return None
    c = sp.cancel(root.base + x * x)
    if c.has(x) or c.is_real is not True or c.is_finite is not True:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    coefficient, rest = data
    if rest != sp.sin(x) or coefficient == 0:
        return None
    d = sp.cancel(root / coefficient - x)
    if d.has(x) or d.is_real is not True or d.is_finite is not True:
        return None
    return (
        LimitStatus.PROVED,
        sp.S.Zero,
        LimitEvidence(
            "principal_imaginary_root_common_phase",
            "For real x sufficiently large, the principal sqrt(c-x^2) is i*h with h=sqrt(x^2-c)>0. sinh(i*h)=i*sin(h), so the expression is sin(x)-h*sin(h)/(x+d). The phase difference is x-h=c/(x+h)->0, and h/(x+d)->1. The real sine Lipschitz bound and |sin(h)|<=1 prove the limit zero. x+d is eventually nonzero.",
            value=sp.S.Zero,
        ),
    )


def independent_periodic_tail_witness(expr, variables, target, domain, assumptions):
    """Attain two values of an affine sum of independent periodic phases."""
    if (
        not 2 <= len(variables) <= 4
        or any(point not in (sp.oo, -sp.oo) for point in target)
        or sp.sympify(domain) is not sp.S.true
        or sp.sympify(assumptions) is not sp.S.true
        or not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 60
    ):
        return None
    atoms = sorted(expr.atoms(sp.sin, sp.cos), key=sp.default_sort_key)
    if not atoms or len(atoms) > len(variables):
        return None
    phases = {}
    markers = {atom: sp.Dummy("periodic_value") for atom in atoms}
    try:
        polynomial = sp.Poly(expr.xreplace(markers), *markers.values())
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() != 1:
        return None
    constant = polynomial.coeff_monomial(1)
    if constant.free_symbols or constant.is_finite is not True:
        return None
    for atom in atoms:
        active = [x for x in variables if atom.has(x)]
        if len(active) != 1 or active[0] in phases:
            return None
        x = active[0]
        try:
            phase = sp.Poly(atom.args[0], x)
        except sp.PolynomialError:
            return None
        if phase.degree() != 1:
            return None
        a, b = phase.nth(1), phase.nth(0)
        coefficient = polynomial.coeff_monomial(markers[atom])
        if any(z.free_symbols or z.is_finite is not True for z in (a, b, coefficient)):
            return None
        if b.is_real is not True or coefficient.is_zero is not False:
            return None
        if a.is_positive is not True and a.is_negative is not True:
            return None
        phases[x] = (atom, a, b)
    j = sp.Dummy("attained_periodic_tail_index", positive=True, integer=True)
    selected = atoms[0]
    witnesses = []
    for alternate in (False, True):
        substitutions = []
        values = {}
        for x, point in zip(variables, target, strict=True):
            direction = 1 if point is sp.oo else -1
            if x not in phases:
                substitutions.append((x, direction * j))
                continue
            atom, a, b = phases[x]
            theta = (
                (sp.pi if atom.func is sp.cos else sp.pi / 2)
                if alternate and atom == selected
                else 0
            )
            substitutions.append(
                (x, direction * 2 * sp.pi * j / sp.Abs(a) + (theta - b) / a)
            )
            values[markers[atom]] = atom.func(theta)
        value = polynomial.as_expr().subs(values)
        witnesses.append(
            LimitEvidence(
                "independent_periodic_tail_subsequence",
                "Each affine phase is an integer multiple of 2*pi plus its chosen "
                "fixed angle. All coordinates tend to their prescribed infinities, "
                "and sine/cosine have no domain holes or poles. Changing one phase "
                "with a nonzero coefficient attains a different exact value.",
                tuple(substitutions),
                value,
            )
        )
    return LimitStatus.DOES_NOT_EXIST, None, tuple(witnesses)
