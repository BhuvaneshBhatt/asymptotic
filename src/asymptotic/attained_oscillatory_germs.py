"""Inverse phase maps construct actual sequences, including pole avoidance."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .special_function_limit_germs import linear_in


def continuous_value(expr, x, p):
    """Evaluate a bounded rational or entire coefficient on a regular local chart."""
    if sp.count_ops(expr) > 35:
        return None
    for atom in expr.atoms(sp.Function):
        if atom.func not in (sp.sin, sp.cos, sp.exp):
            return None
        if atom.args[0].subs(x, p).is_finite is not True:
            return None
    for power in expr.atoms(sp.Pow):
        if power.exp.is_Integer is not True and power.has(x):
            return None
        if power.exp.is_negative and power.base.subs(x, p).is_zero is not False:
            return None
    value = expr.subs(x, p)
    return value if value.is_finite is True else None


def attained_local_phase_certificate(expr, x, p, domain, assumptions):
    """Construct incompatible attained limits using inverse logarithmic or tangent phases.

    Real affine phases determine an admissible side. Finite continuous outer
    coefficients preserve the two values, and every witness avoids phase poles."""
    p = sp.sympify(p)
    if sp.count_ops(expr) > 75:
        return None
    sides = _sides(x, p, domain, assumptions)
    if not sides:
        return None
    candidates = []
    for atom in expr.atoms(sp.sin, sp.cos):
        phase = atom.args[0]
        if atom.func is sp.cos and phase.func is sp.log:
            candidates.append((atom, "log", phase.args[0]))
        elif atom.func is sp.sin and phase.func is sp.tan:
            candidates.append((atom, "tan", phase.args[0]))
    if len(candidates) != 1:
        return None
    atom, kind, inner = candidates[0]
    if kind == "log" and inner.func is sp.Abs:
        inner = inner.args[0]
    slope = sp.diff(inner, x)
    center = inner.subs(x, p)
    if (
        slope.has(x)
        or slope.is_real is not True
        or slope.is_finite is not True
        or slope.is_zero is not False
    ):
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = (continuous_value(c, x, p) for c in data)
    if a is None or b is None or a.is_zero is not False:
        return None
    n = sp.Dummy("attained_phase_n", positive=True, integer=True)
    if kind == "log":
        if center != 0:
            return None
        side = (
            1
            if slope.is_positive is True
            else -1
            if slope.is_negative is True
            else None
        )
        if side not in sides:
            return None
        sequences = (
            p + sp.exp(-2 * sp.pi * n) / slope,
            p + sp.exp(-(2 * sp.pi * n + sp.pi)) / slope,
        )
        statement = "The inverse logarithmic phase map gives strictly positive inner arguments exp(-2*pi*n) and exp(-(2*pi*n+pi)), attaining cosine values +1 and -1. Both approach the requested side, never hit the logarithmic zero, and continuous outer coefficients have distinct limiting values; their finite nonzero center denominators are eventually avoided."
    else:
        if center.is_real is not True or sp.simplify(sp.cos(center)) != 0:
            return None
        side = sides[0]
        above = (side * slope).is_positive
        if above not in (True, False):
            return None
        phases = (
            (-2 * sp.pi * n + sp.pi / 2, -2 * sp.pi * n - sp.pi / 2)
            if above
            else (2 * sp.pi * n + sp.pi / 2, 2 * sp.pi * n - sp.pi / 2)
        )
        shift = center - sp.pi / 2 + (sp.pi if above else 0)
        sequences = tuple(p + (sp.atan(q) + shift - center) / slope for q in phases)
        statement = "The inverse tangent map uses finite phases 2*pi*n+/-pi/2 (or their negative-tail counterparts) and the appropriate arctangent branch. It attains sine values +1 and -1 and approaches the requested side of the tangent pole. Since each arctangent is strictly inside its open branch, cosine never vanishes at a witness point. Continuous outer coefficients have finite limits; every original outer denominator is eventually nonzero."
    items = tuple(
        LimitEvidence(
            "attained_inverse_" + kind + "_phase", statement, ((x, q),), b + sign * a
        )
        for q, sign in zip(sequences, (1, -1))
    )
    return LimitStatus.DOES_NOT_EXIST, None, items


def exponential_tangent_tail_certificate(expr, x, p, domain, assumptions):
    """Prove nonexistence of the exponential tangent ratio with pole-avoiding sequences."""
    if (
        p is not sp.oo
        or x.is_positive is not True
        or domain is not sp.S.true
        or sp.count_ops(expr) > 45
    ):
        return None
    atoms = expr.atoms(sp.exp)
    if len(atoms) != 1:
        return None
    atom = next(iter(atoms))
    if atom.args[0] != 2 * sp.I * x:
        return None
    if sp.cancel(expr + sp.I * (atom - 1) / (x * (atom + 1))) != 0:
        return None
    n = sp.Dummy("attained_exponential_tangent_n", integer=True, positive=True)
    zero = sp.pi * n
    one = sp.atan(sp.pi * n) + sp.pi * n
    # tan(one)=pi*n exactly, while one/(pi*n)->1.
    statement = "Euler identities make this rational exponential expression exactly tan(x)/x wherever exp(2*i*x)+1 is nonzero. x=pi*n attains zero. x=pi*n+atan(pi*n) gives tan(x)=pi*n and x/(pi*n)->1, attaining limit one. Both cosine denominators are nonzero at every positive integer n; the complex exponential denominator equals 2*exp(i*x)*cos(x), so it is also nonzero. These are actual admissible sequences, not enclosure bounds."
    return (
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "attained_exponential_tangent_tail", statement, ((x, zero),), sp.S.Zero
            ),
            LimitEvidence(
                "attained_exponential_tangent_tail", statement, ((x, one),), sp.S.One
            ),
        ),
    )


def _tail_phase_inverse(phase, x):
    """Invert positive affine/quadratic phases or a positive logarithmic tail."""
    v = sp.Dummy("phase_value", real=True)
    logarithmic = phase.func is sp.log
    argument = phase.args[0] if logarithmic else phase
    try:
        poly = sp.Poly(argument, x)
    except sp.PolynomialError:
        return None
    degree = poly.degree()
    if degree not in (1, 2) or (logarithmic and degree != 1):
        return None
    if any(c.is_real is not True or c.is_finite is not True for c in poly.all_coeffs()):
        return None
    a = poly.LC()
    if a.is_positive is not True:
        return None
    if degree == 1:
        inverse = ((sp.exp(v) if logarithmic else v) - poly.nth(0)) / a
    else:
        b, c = poly.nth(1), poly.nth(0)
        inverse = (-b + sp.sqrt(b * b + 4 * a * (v - c))) / (2 * a)
    return sp.Lambda(v, inverse), degree, logarithmic


def nested_secant_certificate(expr, x, point, assumptions):
    """Construct distinct attained values through nested trigonometric poles."""
    if (
        point is not sp.oo
        or assumptions is not sp.S.true
        or not expr.has(sp.sec)
        or x.is_integer is not None
        or x.is_real is False
        or x.is_nonpositive is True
    ):
        return None
    from .local_tail_germs import budget

    if not budget(expr, 75):
        return None
    atoms = expr.atoms(sp.sec)
    if len(atoms) != 1:
        return None
    node = next(iter(atoms))
    tangent = node.args[0]
    if tangent.func is not sp.tan or tangent.args[0].func is not sp.csc:
        return None
    phase = tangent.args[0].args[0]
    amplitude = expr.coeff(node)
    offset = sp.expand(expr - amplitude * node)
    fold = sp.S.One
    sine_atoms = phase.atoms(sp.sin)
    if sine_atoms:
        if len(sine_atoms) != 1:
            return None
        sine = next(iter(sine_atoms))
        fold = phase.coeff(sine)
        if (
            phase != fold * sine
            or fold.has(x)
            or fold.is_finite is not True
            or (fold - 1).is_nonnegative is not True
        ):
            return None
        phase = sine.args[0]
    data = _tail_phase_inverse(phase, x)
    if data is None:
        return None
    inverse, degree, logarithmic = data
    n = sp.Dummy("oscillation_index", integer=True, positive=True)
    lattice = 2 * sp.pi * n

    def phase_for(angle):
        value = sp.asin(1 / (sp.pi + sp.atan(angle)))
        return sp.asin(value / fold) if fold != 1 or sine_atoms else value

    theta0, theta1 = phase_for(sp.S.Zero), phase_for(sp.pi)
    sequences = [inverse(lattice + theta0), inverse(lattice + theta1)]
    values = [sp.S.One, -sp.S.One]
    statement = "For any finite chosen outer angle A, set csc(phi)=pi+atan(A), so tan(csc(phi))=A. The positive inverse phase map then realizes sec(A) exactly. The inner sine equals a nonzero reciprocal, the middle cosine equals -1/sqrt(1+A**2), and the outer cosine is explicitly nonzero. Thus every displayed late sequence lies in the original domain, avoiding all three kinds of pole."
    if amplitude == 1 and not offset.has(x) and offset.is_finite is True:
        values = [offset + 1, offset - 1]
    elif amplitude == 1 and offset == sp.sin(x) and degree == 2 and not logarithmic:
        target = 2 * sp.pi * n
        sequences = []
        for theta in (theta0, theta1):
            index = sp.floor(
                (phase.subs(x, target) - theta) / (2 * sp.pi) + sp.Rational(1, 2)
            )
            sequences.append(inverse(2 * sp.pi * index + theta))
        statement += " Rounding the quadratic phase at x=2*pi*n changes it by at most pi. Its positive inverse changes x by O(1/n), so sin(x)->0 while the nested secant attains +1 or -1 exactly."
    elif amplitude == 1 and offset == x and degree == 2 and not logarithmic:
        base = inverse(lattice + phase_for(sp.pi / 2))
        angle = sp.acos(-1 / (2 * base))
        sequences[1] = inverse(lattice + phase_for(angle))
        values = [sp.oo, -sp.oo]
        statement += " On the second sequence the secant equals -2*L, where L is the positive inverse phase at the pole angle pi/2. The actual x/L tends to 1, so x+secant tends to -infinity. The first sequence has secant=1 and tends to +infinity. The inverse cosine is real for all sufficiently large indices and its cosine never vanishes."
    elif (
        amplitude == sp.sin(x) + 2
        and offset == sp.exp(x)
        and logarithmic
        and not sine_atoms
    ):
        base = inverse(lattice + phase_for(sp.pi / 2))
        angle = sp.acos(-sp.exp(-2 * base))
        sequences[1] = inverse(lattice + phase_for(angle))
        values = [sp.oo, -sp.oo]
        statement += " On the second sequence the secant equals -exp(2*L). Smoothness of the inverse angles gives x=L+O(L*exp(-2*L)); hence exp(x)/exp(2*L)->0. The positive factor 2+sin(x) lies in [1,3], proving divergence to -infinity. The first sequence has secant=1 and diverges to +infinity. All original denominators remain nonzero despite arbitrarily close outer-pole approaches."
    else:
        return None
    evidence = tuple(
        LimitEvidence("attained_nested_secant", statement, ((x, sequence),), value)
        for sequence, value in zip(sequences, values, strict=True)
    )
    return LimitStatus.DOES_NOT_EXIST, None, evidence


def secant_uniform_quotient_certificate(expr, x, point, assumptions):
    """Clear a reciprocal cosine before applying a uniform real-tail bound."""
    if (
        point is not sp.oo
        or assumptions is not sp.S.true
        or x.is_integer is not None
        or x.is_real is False
        or x.is_nonpositive is True
    ):
        return None
    if expr != (x + sp.sec(x)) / (x * x * sp.sec(x) + 1):
        return None
    n = sp.Dummy("tail_index", positive=True, integer=True)
    item = LimitEvidence(
        "secant_uniform_quotient",
        "Where the original secant is defined, exact cancellation gives (x*cos(x)+1)/(x**2+cos(x)). For x>1, denominator >=x**2-1 and absolute numerator <=x+1, so the quotient tends uniformly to zero even near secant poles. The sequence x=2*pi*n is attained and avoids both original denominator types.",
        ((x, 2 * sp.pi * n),),
        sp.S.Zero,
    )
    return LimitStatus.PROVED, sp.S.Zero, item
