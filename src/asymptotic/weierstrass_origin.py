"""Origin germs of invariant-normalized Weierstrass zeta and sigma.

DLMF 23.9: zeta(u)=1/u+O(u**3), sigma(u)=u+O(u**5).
Only rational compositions whose bounded remainders disappear are certified.
"""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus


class WeierstrassZeta(sp.Function):
    nargs = 2


class WeierstrassSigma(sp.Function):
    nargs = 2


def weierstrass_origin_certificate(expr, x, point, domain, assumptions=sp.S.true):
    expr = sp.sympify(expr)
    point = sp.sympify(point)
    domain = sp.sympify(domain)
    assumptions = sp.sympify(assumptions)
    if not isinstance(expr, sp.Expr):
        return None
    atoms = expr.atoms(WeierstrassZeta, WeierstrassSigma)
    if (
        not atoms
        or len(atoms) > 4
        or sp.count_ops(expr) > 100
        or point.is_finite is not True
        or point.has(x)
    ):
        return None
    if x.is_real is True and point.is_real is False:
        return None
    if (
        assumptions is sp.S.false
        or assumptions.has(x)
        or x.is_integer is True
        or x.is_zero is True
    ):
        return None
    if x.is_imaginary is True and point != 0 and point.is_imaginary is not True:
        return None
    if (x.is_nonnegative is True and point.is_negative is True) or (
        x.is_nonpositive is True and point.is_positive is True
    ):
        return None
    if domain is not sp.S.true:
        return None
    remainders = []
    replacements = {}
    for atom in atoms:
        u, invariants = atom.args
        if (
            not isinstance(invariants, sp.Tuple)
            or len(invariants) != 2
            or invariants.has(x, sp.oo, -sp.oo, sp.zoo, sp.nan)
        ):
            return None
        if any(
            a.is_finite is False
            or (not isinstance(a, sp.Symbol) and a.is_finite is not True)
            for a in invariants
        ):
            return None
        if not u.is_rational_function(x):
            return None
        numerator, denominator = sp.fraction(sp.cancel(u))
        try:
            if (
                max(sp.Poly(numerator, x).degree(), sp.Poly(denominator, x).degree())
                > 8
            ):
                return None
        except sp.PolynomialError:
            return None
        if (
            sp.simplify(denominator.subs(x, point)).is_zero is not False
            or sp.simplify(numerator.subs(x, point)) != 0
            or numerator == 0
        ):
            return None
        r = sp.Dummy("bounded_weierstrass_remainder")
        remainders.append(r)
        replacements[atom] = (
            1 / u + u**3 * r if atom.func is WeierstrassZeta else u + u**5 * r
        )
    reduced = expr.xreplace(replacements)
    if not reduced.is_rational_function(x, *remainders):
        return None
    try:
        numerator, denominator = sp.fraction(sp.cancel(reduced))
        polynomials = [sp.Poly(p, x, *remainders) for p in (numerator, denominator)]
        if any(p.total_degree() > 32 for p in polynomials):
            return None
        for p in polynomials:
            for coefficient in p.coeffs():
                if coefficient.is_finite is True:
                    continue
                parameters = tuple(coefficient.free_symbols)
                if (
                    not parameters
                    or any(a.is_finite is False for a in parameters)
                    or not coefficient.is_polynomial(*parameters)
                ):
                    return None
        at_numerator = sp.simplify(numerator.subs(x, point))
        at_denominator = sp.simplify(denominator.subs(x, point))
        if (
            at_numerator.has(*remainders)
            or at_denominator.has(*remainders)
            or at_denominator.is_zero is not False
        ):
            return None
        value = sp.simplify(at_numerator / at_denominator)
        if value.is_finite is not True:
            return None
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "weierstrass_bounded_origin_germ",
            "DLMF 23.9 gives zeta(u)=1/u+u^3 R(u) and sigma(u)=u+u^5 S(u), with bounded local remainders for fixed finite invariants. Exact rational cancellation leaves a nonzero denominator and a finite target value independent of every remainder, uniformly on bounded remainder sets.",
            value=value,
        ),
    )
