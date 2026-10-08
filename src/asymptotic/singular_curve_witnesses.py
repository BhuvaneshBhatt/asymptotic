"""Attained algebraic curves approaching signed-root pole sets."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus


def signed_root_pole_witness(expr, variables, target, domain, assumptions):
    """Separate an axis from a curve with a prescribed pole-cancellation order.

    For P=a*x**(q*n)+b*y**n, choose c=(-a/b)**(1/n)>0 and
    y=c*x**q*(1+x**k). The root denominator has order q+k/n.
    Taking k=n*(u+q*v-q)>0 matches a monomial numerator x**u*y**v.
    The resulting nonzero limit disagrees with the zero on y=0.
    """
    if (
        len(variables) != 2
        or tuple(target) != (0, 0)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 100
    ):
        return None
    signs = expr.atoms(sp.sign)
    if len(signs) != 1:
        return None
    sign = next(iter(signs))
    polynomial = sign.args[0]
    powers = expr.as_powers_dict()
    exponent = powers.get(sp.Abs(polynomial))
    if powers.get(sign) != -1 or exponent is None or not exponent.is_Rational:
        return None
    n = -1 / exponent
    if not (n.is_Integer and 3 <= n <= 7 and n.is_odd):
        return None
    numerator = sp.Mul(
        *(
            base**power
            for base, power in powers.items()
            if base not in (sign, sp.Abs(polynomial))
        )
    )
    for x, y in (variables, variables[::-1]):
        try:
            terms = sp.Poly(polynomial, x, y).terms()
            monomials = sp.Poly(numerator, x, y).terms()
        except sp.PolynomialError:
            continue
        if len(terms) != 2 or len(monomials) != 1:
            continue
        (first, a), (second, b) = terms
        m = first[0]
        if first != (m, 0) or second != (0, int(n)) or m <= 0 or m > 21 or m % n:
            continue
        (u, v), scale = monomials[0]
        if not (1 <= u <= 6 and 1 <= v <= 6):
            continue
        if any(
            z.free_symbols
            or z.is_real is not True
            or z.is_finite is not True
            or z.is_zero is not False
            for z in (a, b, scale)
        ):
            continue
        ratio = -a / b
        if ratio.is_positive is not True:
            continue
        q = m // int(n)
        k = int(n) * (u + q * v - q)
        if k <= 0:
            continue
        c = ratio ** (1 / n)
        root = sp.sign(-a * n) * sp.Abs(-a * n) ** (1 / n)
        value = sp.simplify(scale * c**v / root)
        j = sp.Dummy("attained_pole_curve_index", positive=True, integer=True)
        evidence = (
            LimitEvidence(
                "signed_root_axis_subsequence",
                "At x=1/j,y=0 the numerator vanishes and P=a*x**m is nonzero, "
                "so the expression is defined and exactly zero.",
                ((x, 1 / j), (y, 0)),
                sp.S.Zero,
            ),
            LimitEvidence(
                "signed_root_pole_curve_subsequence",
                "At x=1/j,y=c*x**q*(1+x**k), c**n=-a/b, so "
                "P=a*x**m*(1-(1+x**k)**n) never vanishes for x>0. "
                "Its leading term is -a*n*x**(m+k). The signed odd root "
                "and numerator have the same order, giving the stated "
                "nonzero limit. These defined points attain the origin.",
                ((x, 1 / j), (y, c / j**q * (1 + j ** (-k)))),
                value,
            ),
        )
        return LimitStatus.DOES_NOT_EXIST, None, evidence
    return None
