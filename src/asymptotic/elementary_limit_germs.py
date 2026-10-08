"""Bounded phase and cancellation certificates with attained witnesses."""

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus
from .tail_cancellation_germs import clean, finite_clean, perturbation_budget


def answer(value, method, statement):
    return LimitStatus.PROVED, value, LimitEvidence(method, statement, value=value)


def finite_coefficients(poly):
    for c in poly.all_coeffs():
        if c.is_finite is True:
            continue
        parameters = tuple(c.free_symbols)
        if (
            not parameters
            or any(p.is_finite is False for p in parameters)
            or not c.is_polynomial(*parameters)
        ):
            return False
    return True


def rational_value(expr, x, point):
    if not expr.is_rational_function(x):
        return None
    try:
        n, d = map(lambda p: sp.Poly(p, x), sp.fraction(sp.cancel(expr)))
        if (
            max(n.degree(), d.degree()) > 12
            or d.is_zero
            or not all(finite_coefficients(p) for p in (n, d))
        ):
            return None
        if n.is_zero:
            return sp.S.Zero
        if point is sp.oo:
            degree = n.degree() - d.degree()
            coefficient = n.LC() / d.LC()
            if d.LC().is_zero is not False:
                return None
        elif point == 0:
            ni, nc = min(n.terms())
            di, dc = min(d.terms())
            degree = di[0] - ni[0]
            coefficient = nc / dc
            if dc.is_zero is not False:
                return None
        else:
            return None
        if degree < 0:
            return sp.S.Zero
        if degree == 0:
            return sp.simplify(coefficient)
        if coefficient.is_positive is True:
            return sp.oo
        if coefficient.is_negative is True:
            return -sp.oo
    except (sp.PolynomialError, ValueError, TypeError):
        return None


def real_parameter_chart(expr, x, assumptions):
    replacements = {}
    clauses = sp.And.make_args(assumptions)
    for p in expr.free_symbols - {x}:
        if p.is_real is True:
            continue
        if sp.Eq(sp.im(p), 0) in clauses or sp.Q.real(p) in clauses:
            replacements[p] = sp.Dummy(str(p), real=True)
    return expr.xreplace(replacements), {v: k for k, v in replacements.items()}


def unit_phase_envelope_certificate(expr, x, point, domain, assumptions):
    """Bound a unit-modulus complex phase by a vanishing real amplitude."""
    if (
        point is not sp.oo
        or x.is_positive is not True
        or domain is not sp.S.true
        or not expr.has(sp.exp)
        or not perturbation_budget(expr, 70)
    ):
        return None
    chart, reverse = real_parameter_chart(expr, x, assumptions)
    atoms = [a for a in chart.atoms(sp.exp) if (a.args[0] / sp.I).is_real is True]
    if not atoms or len(atoms) > 3:
        return None
    w = [sp.Dummy("unit_phase") for a in atoms]
    transformed = chart.xreplace(dict(zip(atoms, w)))
    try:
        p = sp.Poly(transformed, *w)
    except sp.PolynomialError:
        return None
    if p.total_degree() > 4:
        return None
    if any(rational_value(c, x, point) != 0 for c in p.coeffs()):
        return None
    return answer(
        sp.S.Zero,
        "unit_phase_vanishing_envelope",
        "Each exponential has a proved real phase and unit modulus. The expression is a bounded-degree polynomial in those phases, and every rational coefficient tends to zero; the triangle inequality proves the limit without inferring nonexistence from bounds.",
    )


def exponential_phase_subsequences(expr, x, point, domain):
    """Invert a real exponential phase to attain incompatible oscillatory values."""
    if (
        domain is not sp.S.true
        or x.is_integer is True
        or x.is_real is False
        or not perturbation_budget(expr, 70)
    ):
        return None
    atoms = []
    inverse = None
    if point is sp.oo and x.is_positive is True:
        atoms = [a for a in expr.atoms(sp.exp) if (a.args[0] / sp.I).is_real is True]
        if len(atoms) != 1:
            return None
        a, r = sp.expand(atoms[0].args[0] / sp.I).as_coeff_exponent(x)
        if (
            a.has(x)
            or a.is_finite is not True
            or r.is_Rational is not True
            or r <= 0
            or (a.is_positive is not True and a.is_negative is not True)
        ):
            return None

        def inverse(y):
            return (y / sp.Abs(a)) ** (1 / r)
    elif point == 0 and x.is_nonpositive is not True and x.is_integer is not True:
        atoms = [
            p
            for p in expr.atoms(sp.Pow)
            if p.base == x and (p.exp / sp.I).is_real is True and p.exp != 0
        ]
        if len(atoms) != 1:
            return None
        a = atoms[0].exp / sp.I
        if (
            a.has(x)
            or a.is_finite is not True
            or (a.is_positive is not True and a.is_negative is not True)
        ):
            return None

        def inverse(y):
            return sp.exp(-y / sp.Abs(a))
    else:
        return None
    atom = atoms[0]
    w = sp.Dummy("attained_unit_phase")
    transformed = expr.xreplace({atom: w})
    if not transformed.is_rational_function(x, w):
        return None
    n = sp.Dummy("subsequence_n", positive=True, integer=True)
    witnesses = []
    for offset in (0, sp.pi):
        sequence = inverse(2 * sp.pi * n + offset)
        unit = sp.S.One if offset == 0 else -sp.S.One
        if sp.simplify(atom.subs(x, sequence) - unit) != 0:
            return None
        collapsed = sp.cancel(transformed.subs(w, unit))
        value = rational_value(collapsed, x, point)
        if value is None or value.has(sp.zoo, sp.nan):
            continue
        if point == 0 and x.is_zero is True:
            return None
        # rational_value checked a nonzero leading denominator coefficient.
        # A nonzero polynomial has finitely many zeros; the attained sequence
        # eventually avoids them as it tends to zero or positive infinity.
        evidence = LimitEvidence(
            "attained_exponential_phase_subsequence",
            "The positive real sequence attains the stated unit phase exactly and approaches the target. Its specialized denominator is a nonzero polynomial with finitely many zeros, all avoided on a sufficiently late tail of the sequence.",
            ((x, sequence),),
            value,
        )
        witnesses.append((value, evidence))
    if (
        len(witnesses) == 2
        and sp.simplify(witnesses[0][0] - witnesses[1][0]).is_zero is False
    ):
        return LimitStatus.DOES_NOT_EXIST, None, tuple(e for v, e in witnesses)
    return None


def nonpositive_radical_subsequences(expr, x, point, domain):
    """Construct signed radical approaches under the checked real-domain conditions."""
    if (
        point != 0
        or domain is not sp.S.true
        or x.is_integer is True
        or x.is_real is False
        or x.is_nonnegative is True
        or x.is_nonpositive is True
        or not perturbation_budget(expr, 35)
    ):
        return None
    for root in expr.atoms(sp.Pow):
        if root.exp != sp.Rational(1, 2):
            continue
        a = sp.cancel(x * expr + root)
        if (
            a.has(x)
            or (a != 0 and a.is_negative is not True)
            or sp.expand(root.base - a * a - x * x) != 0
        ):
            continue
        n = sp.Dummy("subsequence_n", integer=True, positive=True)
        values = (-sp.S.One, sp.S.One) if a == 0 else (-sp.oo, sp.oo)
        evidence = tuple(
            LimitEvidence(
                "attained_nonpositive_radical_subsequence",
                "For real a<=0, x=+/-1/n stays in the real radical domain and has nonzero denominator. At a=0 the attained values are -1 and +1; at a<0 the numerator tends to 2a<0 and the two denominator signs give opposing infinite limits.",
                ((x, sign / n),),
                value,
            )
            for sign, value in zip((1, -1), values)
        )
        return LimitStatus.DOES_NOT_EXIST, None, evidence
    return None


def near_one_cancellation_certificate(expr, x, point):
    """Resolve a near-one exponential power through its logarithmic perturbation scale."""
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not perturbation_budget(expr, 90)
    ):
        return None
    atoms = [
        p
        for p in expr.atoms(sp.Pow)
        if p.exp.has(x)
        and p.base != x
        and p.base.is_real is True
        and p.exp.is_real is True
    ]
    if len(atoms) != 1:
        return None
    atom = atoms[0]
    delta = sp.cancel(atom.base - 1)
    power = atom.exp
    if delta == 0 or not delta.is_rational_function(x) or clean(delta, x, point) != 0:
        return None
    w = sp.Dummy("near_one_power")
    try:
        p = sp.Poly(sp.expand_mul(expr.xreplace({atom: w})), w)
    except sp.PolynomialError:
        return None
    if p.degree() != 1:
        return None
    # A pure power/product has no subtraction to protect. Preserve its cheap
    # log-remainder/Stirling routes rather than computing cancellation terms.
    if p.coeff_monomial(1) == 0:
        return None
    coefficient = p.coeff_monomial(w)
    constant = clean(power * delta, x, point)
    if (
        constant is None
        or constant.is_real is not True
        or constant.is_finite is not True
    ):
        return None
    logarithm = power * (delta - delta**2 / 2 + delta**3 / 3)
    small = sp.expand(logarithm - constant)
    if (
        clean(small, x, point) != 0
        or clean(sp.Abs(coefficient * power) * delta**4, x, point) != 0
        or clean(sp.Abs(coefficient) * small**4, x, point) != 0
    ):
        return None
    approximation = sp.exp(constant) * (1 + small + small**2 / 2 + small**3 / 6)
    value = clean(sp.expand(p.as_expr().subs(w, approximation)), x, point)
    if value is None:
        return None
    return answer(
        value,
        "near_one_power_cancellation_remainder",
        "For real delta->0 the base is eventually positive. Three log and exponential terms are retained around the finite limiting exponent; both amplified fourth-order remainders tend to zero before the cancellation is evaluated.",
    )


def gamma_ratio_cancellation_certificate(expr, x, point):
    """Use fixed-shift gamma asymptotics with sufficient terms for the observed cancellation."""
    atoms = expr.atoms(sp.gamma)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 2
        or not perturbation_budget(expr, 90)
    ):
        return None
    for denominator in atoms:
        numerator = next(a for a in atoms if a != denominator)
        a = sp.expand(numerator.args[0] - x)
        b = sp.expand(denominator.args[0] - x)
        if not all(c.is_Rational and abs(c) <= 16 for c in (a, b)):
            continue
        w = sp.Dummy("normalized_gamma_ratio")
        transformed = sp.cancel(
            expr.xreplace({numerator: w * denominator * x ** (a - b)})
        )
        if transformed.has(sp.gamma):
            continue
        try:
            p = sp.Poly(transformed, w)
        except sp.PolynomialError:
            continue
        if p.degree() != 1:
            continue
        coefficient = p.coeff_monomial(w)
        if clean(sp.Abs(coefficient) / x**4, x, point) != 0:
            continue
        c = [
            (-1) ** (k + 1)
            * (sp.bernoulli(k + 1, a) - sp.bernoulli(k + 1, b))
            / (k * (k + 1))
            for k in range(1, 4)
        ]
        approximation = (
            1
            + c[0] / x
            + (c[1] + c[0] ** 2 / 2) / x**2
            + (c[2] + c[0] * c[1] + c[0] ** 3 / 6) / x**3
        )
        value = clean(sp.expand(p.as_expr().subs(w, approximation)), x, point)
        if value is None:
            continue
        return answer(
            value,
            "gamma_ratio_bernoulli_cancellation",
            "DLMF 5.11 fixed-shift log-gamma ratio coefficients are retained through x^-3 and exponentiated with remainder O(x^-4). The exact coefficient multiplying this remainder divided by x^4 tends to zero, so cancellation is certified.",
        )
    return None


def zeta_laurent_cancellation_certificate(expr, x, point, domain):
    """Resolve a zeta pole cancellation using the retained Laurent coefficients."""
    atoms = expr.atoms(sp.zeta)
    if point in (sp.oo, -sp.oo) or len(atoms) != 1 or not perturbation_budget(expr, 80):
        return None
    atom = next(iter(atoms))
    delta = sp.cancel(atom.args[0] - 1)
    q = atom.args[1] if len(atom.args) > 1 else sp.S.One
    if (
        q.has(x)
        or q.is_positive is not True
        or q.is_finite is not True
        or delta == 0
        or not delta.is_rational_function(x)
        or delta.subs(x, point) != 0
    ):
        return None
    w = sp.Dummy("zeta_laurent")
    try:
        p = sp.Poly(sp.expand_mul(expr.xreplace({atom: w})), w)
    except sp.PolynomialError:
        return None
    if p.degree() != 1:
        return None
    coefficient = p.coeff_monomial(w)
    if finite_clean(coefficient * delta**4, x, point, domain) != 0:
        return None
    approximation = (
        1 / delta
        - sp.polygamma(0, q)
        + sum(
            (-1) ** k
            * (sp.stieltjes(k) if q == 1 else sp.stieltjes(k, q))
            * delta**k
            / sp.factorial(k)
            for k in range(1, 4)
        )
    )
    value = finite_clean(
        sp.expand(p.as_expr().subs(w, approximation)), x, point, domain
    )
    if value is None:
        return None
    return answer(
        value,
        "zeta_laurent_cancellation_remainder",
        "For fixed positive q, the Laurent expansion at s=1 is retained through delta^3, with analytic remainder O(delta^4). Its exact multiplying coefficient times delta^4 tends to zero on every requested side.",
    )
