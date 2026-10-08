"""Bounded tails and cancellation recognition before generic expansions."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus
from .local_nonexistence_families import _sides
from .special_function_limit_germs import linear_in


def harmonic_convergent_tail_certificate(expr, x, p, domain, assumptions):
    """Bound the positive Hurwitz tail for a convergent fixed-order harmonic number."""
    if (
        p is not sp.oo
        or x.is_positive is not True
        or domain is not sp.S.true
        or expr.func is not sp.harmonic
    ):
        return None
    if len(expr.args) != 2 or sp.count_ops(expr) > 25:
        return None
    n, s = expr.args
    a = sp.diff(n, x)
    b = n.subs(x, 0)
    if (
        a.has(x)
        or a.is_positive is not True
        or a.is_finite is not True
        or b.is_real is not True
        or b.is_finite is not True
    ):
        return None
    if not s.is_number or s.is_real is not True or (s - 1).is_positive is not True:
        return None
    value = sp.zeta(s)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "harmonic_hurwitz_positive_tail_bound",
            "For fixed real s>1, H(n,s)=zeta(s)-zeta(s,n+1). On the eventual positive real affine argument, the Hurwitz tail is bounded by (n+1)^(-s)+(n+1)^(1-s)/(s-1), which tends to zero. This proves the analytic continuation as well as integer partial sums without requesting a general series.",
            value=value,
        ),
    )


def logarithmic_opposite_pole_ratio_certificate(expr, x, p, domain, assumptions):
    """Cancel the common divergent log scale at a transverse cotangent pole.

    Principal imaginary phases remain bounded while the shared real logarithm
    diverges. Both logarithm arguments are eventually nonzero."""
    if not (expr.has(sp.log) and expr.has(sp.cot, sp.csc)):
        return None
    if sp.count_ops(expr) > 45 or not _sides(x, sp.sympify(p), domain, assumptions):
        return None
    replacements = {
        atom: sp.log(sp.trigsimp(sp.expand_mul(atom.args[0])))
        for atom in expr.atoms(sp.log)
        if atom.args[0].has(sp.csc) and sp.count_ops(atom.args[0]) <= 30
    }
    replacements = {
        atom: value.replace(
            lambda node: node.is_Pow and node.exp == -1 and node.base.func is sp.tan,
            lambda node: sp.cot(node.base.args[0]),
        )
        for atom, value in replacements.items()
    }
    expr = expr.xreplace(replacements)
    logs = expr.atoms(sp.log)
    if len(logs) != 2:
        return None
    first, second = tuple(logs)
    sign = (
        1
        if expr == first / second or expr == second / first
        else -1
        if expr == -first / second or expr == -second / first
        else None
    )
    if sign is None:
        return None
    arg = first.args[0]
    cot = arg.atoms(sp.cot) | second.args[0].atoms(sp.cot)
    if len(cot) != 1:
        return None
    atom = next(iter(cot))
    u = atom.args[0]
    center = u.subs(x, p)
    rate = sp.diff(u, x)
    if (
        rate.has(x)
        or rate.is_real is not True
        or rate.is_finite is not True
        or rate.is_zero is not False
        or sp.simplify(sp.sin(center)) != 0
    ):
        return None
    for argument in (arg, second.args[0]):
        data = linear_in(argument, atom)
        if data is None:
            return None
        a, b = data
        if (
            any(v.has(x) or v.is_finite is not True for v in (a, b))
            or a.is_zero is not False
        ):
            return None
    return (
        LimitStatus.PROVED,
        sp.Integer(sign),
        LimitEvidence(
            "common_logarithmic_pole_scale",
            "The cotangent argument has a transverse real pole. Both affine cotangent arguments have modulus diverging like a nonzero constant divided by Abs(x-point). Each principal logarithm equals the common diverging real log-modulus plus an imaginary part bounded by pi. Dividing preserves the common scale and the bounded branch phases vanish. All logarithm arguments are eventually nonzero and tangent denominators are eventually avoided.",
            value=sp.Integer(sign),
        ),
    )


def circular_segment_endpoint_certificate(expr, x, p, domain, assumptions):
    """Evaluate a positive circular endpoint using its retained square-root remainder."""
    if (
        p is not sp.oo
        or x.is_positive is not True
        or domain is not sp.S.true
        or not expr.has(sp.atan)
        or sp.count_ops(expr) > 110
    ):
        return None
    roots = [a for a in expr.atoms(sp.Pow) if a.exp == sp.S.Half and not a.has(x)]
    for root in roots:
        q = root.base
        if q.is_finite is False or q.is_real is False:
            continue
        y = root - 1 / x
        remainder = sp.sqrt(q - y * y)
        expected = y * remainder / 2 + q * sp.atan(y / remainder) / 2
        if expr != expected:
            continue
        positive = q.is_positive is True
        clauses = sp.And.make_args(assumptions)
        if sp.Gt(q, 0) in clauses:
            positive = True
        for clause in clauses:
            if (
                not isinstance(clause, sp.StrictGreaterThan)
                or clause.lhs.func is not sp.Pow
                or clause.lhs.exp != sp.S.Half
            ):
                continue
            bound = clause.lhs
            z = clause.rhs
            if sp.expand(q - bound**2 + z**2) != 0:
                continue
            if sp.Gt(bound, 0) in clauses and (
                sp.Gt(z, -bound) in clauses or sp.Gt(z + bound, 0) in clauses
            ):
                positive = True
        if not positive:
            return None
        value = sp.pi * q / 4
        return (
            LimitStatus.PROVED,
            value,
            LimitEvidence(
                "ordered_positive_circular_endpoint",
                "The ordered real bounds prove q>0. With y=sqrt(q)-1/x, q-y^2=2*sqrt(q)/x-1/x^2 is eventually positive and tends to zero. Thus y*sqrt(q-y^2)->0 and y/sqrt(q-y^2)->+infinity, so the principal arctangent tends to pi/2. The remainder square root and its denominator stay strictly positive on the eventual domain. This uses the common endpoint scale rather than nested generic radicals/logarithmic series.",
                value=value,
            ),
        )
    return None
