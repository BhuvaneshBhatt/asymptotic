"""Bounded rational ray poles and signed finite special-function cut germs."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus
from .special_function_limit_germs import linear_in


class DirectionalInfinity(sp.Function):
    """Infinite modulus with a certified nonzero normalized complex direction."""

    nargs = 1

    @classmethod
    def eval(cls, d):
        if d.is_zero is True or d.is_finite is False or d.has(sp.nan, sp.zoo):
            raise ValueError("an infinite direction must be finite and nonzero")
        if d.is_positive is True:
            return sp.oo
        if d.is_negative is True:
            return -sp.oo


def rational_ray_pole_certificate(expr, t):
    """Certify a bounded rational Laurent pole on a fixed complex ray."""
    if sp.count_ops(expr) > 65 or not expr.is_rational_function(t):
        return None
    if any(p.exp.is_Integer and abs(p.exp) > 6 for p in expr.atoms(sp.Pow)):
        return None
    try:
        num, den = (sp.Poly(v, t) for v in sp.fraction(sp.cancel(expr)))
    except sp.PolynomialError:
        return None
    if num.is_zero or den.is_zero or max(num.degree(), den.degree()) > 6:
        return None
    ni, nc = min(num.terms())
    di, dc = min(den.terms())
    c = nc / dc
    if (
        ni[0] >= di[0]
        or c.free_symbols
        or c.is_finite is not True
        or c.is_zero is not False
    ):
        return None
    phase = sp.simplify(c / sp.Abs(c))
    value = DirectionalInfinity(phase)
    n = sp.Dummy("attained_ray_pole_n", positive=True, integer=True)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "certified_fixed_ray_laurent_pole",
            "Exact rational Laurent valuation on the positive ray parameter gives c*t^(-m)*(1+O(t)), with m>0 and fixed finite nonzero c. Therefore modulus diverges and normalized direction tends to c/Abs(c). t=1/n attains that direction; polynomial denominators have only finitely many nonzero roots and are eventually avoided. This certifies the requested ray only.",
            ((t, 1 / n),),
            value,
        ),
    )


def ray_sign_certificate(expr, t):
    """Take the normalized phase of a rational leading coefficient on a fixed ray."""
    atoms = expr.atoms(sp.sign)
    if len(atoms) != 1 or sp.count_ops(expr) > 45:
        return None
    atom = next(iter(atoms))
    arg = atom.args[0]
    if not arg.is_rational_function(t):
        return None
    try:
        num, den = (sp.Poly(v, t) for v in sp.fraction(sp.cancel(arg)))
    except sp.PolynomialError:
        return None
    if num.is_zero or den.is_zero or max(num.degree(), den.degree()) > 6:
        return None
    c = min(num.terms())[1] / min(den.terms())[1]
    data = linear_in(expr, atom)
    if data is None or any(v.has(t) or v.is_finite is not True for v in data):
        return None
    if c.free_symbols or c.is_finite is not True or c.is_zero is not False:
        return None
    a, b = data
    value = sp.simplify(a * c / sp.Abs(c) + b)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "certified_fixed_ray_complex_sign",
            "A rational argument on t>0 has a nonzero fixed Laurent coefficient c. Its complex sign is arg/Abs(arg), tending to c/Abs(c), including arguments tending to zero or infinity. Original rational denominators and zeros are isolated and eventually avoided.",
            value=value,
        ),
    )


def _constant_outer(expr, atom, t):
    data = linear_in(expr, atom)
    if data is None:
        return None
    values = []
    for v in data:
        if v.has(t) and not v.is_rational_function(t):
            return None
        w = v.subs(t, 0)
        if w.is_finite is not True:
            return None
        values.append(w)
    return values


def special_cut_ray_certificate(expr, t):
    """Select signed fixed-order Bessel and elliptic cut values on regular affine charts."""
    if sp.count_ops(expr) > 70:
        return None
    atoms = [
        f
        for f in expr.atoms(sp.Function)
        if f.has(t)
        and f.func
        in (
            sp.elliptic_k,
            sp.elliptic_e,
            sp.besselj,
            sp.bessely,
            sp.besseli,
            sp.besselk,
        )
    ]
    if len(atoms) != 1:
        return None
    atom = atoms[0]
    arg = atom.args[-1]
    center = arg.subs(t, 0)
    rate = sp.diff(arg, t)
    if (
        rate.has(t)
        or rate.is_finite is not True
        or rate.is_zero is not False
        or center.is_finite is not True
    ):
        return None
    imaginary = sp.simplify(sp.im(rate))
    if imaginary == 0:
        upper = False
        lower = False
    elif imaginary.is_positive is True:
        upper = True
        lower = False
    elif imaginary.is_negative is True:
        upper = False
        lower = True
    else:
        return None
    h = atom.func
    if h in (sp.elliptic_k, sp.elliptic_e):
        if (center - 1).is_positive is not True:
            return None
        baseline = h(center)
        boundary = sp.conjugate(baseline) if upper else baseline
        if center == 2:
            if h is sp.elliptic_k:
                constant = (
                    sp.sqrt(sp.pi)
                    * sp.gamma(sp.Rational(5, 4))
                    / sp.gamma(sp.Rational(3, 4))
                )
                boundary = constant * (1 + sp.I if upper else 1 - sp.I)
            else:
                constant = (
                    sp.sqrt(sp.pi)
                    * sp.gamma(sp.Rational(3, 4))
                    / sp.gamma(sp.Rational(1, 4))
                )
                boundary = constant * (1 - sp.I if upper else 1 + sp.I)
            data = linear_in(expr, atom)
            if data is not None:
                data = tuple(
                    v.xreplace(
                        {
                            baseline: constant
                            * (1 - sp.I if h is sp.elliptic_k else 1 + sp.I)
                        }
                    )
                    for v in data
                )
            if data is None or any(v.has(t) or v.is_finite is not True for v in data):
                return None
        else:
            data = _constant_outer(expr, atom, t)
    else:
        nu = atom.args[0]
        if (
            center.is_negative is not True
            or not nu.is_number
            or nu.is_real is not True
            or nu.is_finite is not True
        ):
            return None
        baseline = h(nu, center)
        boundary = sp.conjugate(baseline) if lower else baseline
        # Positive-real J has a decreasing alternating series when its first
        # term ratio is <1; I has strictly positive terms for nu>-1.
        if expr == atom / baseline and h in (sp.besselj, sp.besseli):
            r = -center
            nonzero = (nu + 1).is_positive is True and (
                h is sp.besseli or (1 - r * r / (4 * (nu + 1))).is_positive is True
            )
            if not nonzero:
                return None
            value = sp.exp(-2 * sp.pi * sp.I * nu) if lower else sp.S.One
            return (
                LimitStatus.PROVED,
                value,
                LimitEvidence(
                    "certified_bessel_cut_ratio",
                    "DLMF 10.11 gives the principal J/I continuation phase exp(-2*pi*i*nu) on the lower side of the negative axis; the upper side and real cut use the stored principal value. Nonzero denominator is proved by the decreasing alternating J power series with first term ratio <1, or the positive I series for nu>-1. No numerical zero test is used.",
                    value=value,
                ),
            )
        data = _constant_outer(expr, atom, t)
    if data is None:
        return None
    a, b = data
    value = sp.simplify(a * boundary + b)
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "certified_finite_special_function_cut",
            "Fixed real order and nonzero negative Bessel argument, or real elliptic parameter >1, admit finite analytic boundary germs on each half-plane. Real coefficient continuation gives conjugate opposite boundaries. The affine ray has fixed imaginary sign and avoids the branch points. Finite continuous outer coefficients preserve the O(t) error. DLMF 10.11 and 19.2 define the principal branches; elliptic m=2 constants follow the beta-integral evaluations.",
            value=value,
        ),
    )


def original_bessel_ray_certificate(expr, x, p, d):
    """Recognize Bessel branches before substitution can rewrite a negative argument."""
    atoms = [a for a in expr.atoms(sp.besselj, sp.besseli) if a.has(x)]
    if len(atoms) != 1 or sp.count_ops(expr) > 60:
        return None
    atom = atoms[0]
    nu, arg = atom.args
    h = atom.func
    if (
        arg != x
        or p.is_negative is not True
        or not nu.is_number
        or nu.is_real is not True
    ):
        return None
    imaginary = sp.simplify(sp.im(d))
    if imaginary == 0:
        lower = False
    elif imaginary.is_negative is True:
        lower = True
    elif imaginary.is_positive is True:
        lower = False
    else:
        return None
    r = -p
    nonzero = (nu + 1).is_positive is True and (
        h is sp.besseli or (1 - r * r / (4 * (nu + 1))).is_positive is True
    )
    if expr == atom / h(nu, p):
        if not nonzero:
            return None
        value = sp.exp(-2 * sp.pi * sp.I * nu) if lower else sp.S.One
    elif expr == atom:
        value = sp.exp((-1 if lower else 1) * sp.pi * sp.I * nu) * h(nu, r)
    else:
        return None
    return (
        LimitStatus.PROVED,
        value,
        LimitEvidence(
            "certified_original_bessel_cut_ray",
            "Principal J/I(z) continuation across the negative real axis contributes exp(+/-i*pi*nu) times the positive-real value (DLMF 10.11). This is checked before symbolic substitution can rewrite negative arguments. A ratio denominator is nonzero by a decreasing alternating J series with first term ratio <1, or positive I series for nu>-1. The nonzero affine ray stays in its half-plane and away from the origin.",
            value=value,
        ),
    )
