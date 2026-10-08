"""Bounded extensions: rational pullbacks, adaptive order and local real charts.

Only fixed orders up to six are used. Every retained remainder must vanish
in the original expression; no generic series is requested to discover scales.
"""

import sympy as sp

from .elementary_limit_germs import answer, rational_value
from .limit_models import LimitEvidence, LimitStatus
from .special_function_limit_germs import linear_in
from .tail_cancellation_germs import clean, finite_clean, perturbation_budget


def exp_coefficients(log_coefficients):
    """Formal exp coefficients, by n*c_n=sum(k*l_k*c_(n-k))."""
    result = [sp.S.One]
    for n in range(1, len(log_coefficients) + 1):
        result.append(
            sp.expand(
                sum(
                    k * log_coefficients[k - 1] * result[n - k] for k in range(1, n + 1)
                )
                / n
            )
        )
    return result


def vanishing_remainder(value, x, point):
    """Avoid generic limit analysis for explicitly rational remainder bounds."""
    if value.is_rational_function(x):
        result = rational_value(value, x, point)
        if result is not None:
            return result == 0
    return clean(value, x, point) == 0


def gamma_rational_pullback_certificate(expr, x, point):
    atoms = expr.atoms(sp.gamma)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 2
        or not perturbation_budget(expr, 100)
    ):
        return None
    for denominator in atoms:
        u = denominator.args[0]
        if (
            not u.is_rational_function(x)
            or u.is_real is not True
            or rational_value(u, x, point) is not sp.oo
        ):
            continue
        numerator = next(a for a in atoms if a != denominator)
        shift = sp.cancel(numerator.args[0] - u)
        if not shift.is_Rational or abs(shift) > 16:
            continue
        w = sp.Dummy("normalized_gamma_pullback")
        transformed = sp.cancel(expr.xreplace({numerator: w * denominator * u**shift}))
        if transformed.has(sp.gamma):
            continue
        data = linear_in(transformed, w)
        if data is None:
            continue
        a, b = data
        # Select required order from the original multiplier before expanding.
        order = next(
            (
                n
                for n in range(1, 7)
                if vanishing_remainder(sp.Abs(a) / u ** (n + 1), x, point)
            ),
            None,
        )
        if order is None:
            continue
        logs = [
            (-1) ** (k + 1)
            * (sp.bernoulli(k + 1, shift) - sp.bernoulli(k + 1))
            / (k * (k + 1))
            for k in range(1, order + 1)
        ]
        coeffs = exp_coefficients(logs)
        approximation = sum(c / u**k for k, c in enumerate(coeffs))
        value = clean(sp.expand(a * approximation + b), x, point)
        if value is None:
            continue
        return answer(
            value,
            "gamma_rational_pullback_adaptive_remainder",
            f"Fixed-shift DLMF 5.11 expansion along real rational u->+infinity, retained through u^-{order}; the original multiplier times O(u^-{order + 1}) vanishes. Rational u has only finitely many poles, eventually avoided, and both gamma arguments are eventually positive.",
        )
    return None


def near_one_adaptive_certificate(expr, x, point):
    if (
        point is not sp.oo
        or x.is_positive is not True
        or not perturbation_budget(expr, 100)
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
    if (
        delta == 0
        or not delta.is_rational_function(x)
        or rational_value(delta, x, point) != 0
    ):
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    if b == 0:
        return None  # Leave pure products to existing cheap paths.
    if not power.is_rational_function(x):
        return None
    constant = rational_value(sp.cancel(power * delta), x, point)
    if (
        constant is None
        or constant.is_real is not True
        or constant.is_finite is not True
    ):
        return None
    order = next(
        (
            n
            for n in range(1, 7)
            if vanishing_remainder(sp.Abs(a * power) * delta ** (n + 1), x, point)
        ),
        None,
    )
    if order is None:
        return None
    small = sp.cancel(
        power * sum((-1) ** (k + 1) * delta**k / k for k in range(1, order + 1))
        - constant
    )
    if rational_value(small, x, point) != 0:
        return None
    exp_order = next(
        (
            n
            for n in range(1, 7)
            if vanishing_remainder(sp.Abs(a) * small ** (n + 1), x, point)
        ),
        None,
    )
    if exp_order is None:
        return None
    approximation = sp.exp(constant) * sum(
        small**k / sp.factorial(k) for k in range(exp_order + 1)
    )
    value = clean(sp.expand(a * approximation + b), x, point)
    if value is None:
        return None
    return answer(
        value,
        "near_one_adaptive_cancellation_remainder",
        f"Eventually positive real base: log retained through delta^{order}, exp through perturbation^{exp_order}. Both original amplified remainders vanish. Required orders are selected before expansion and capped at six.",
    )


def real_error_tail_certificate(expr, x, point):
    atoms = expr.atoms(sp.erfc, sp.erf)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 100)
    ):
        return None
    atom = next(iter(atoms))
    u = atom.args[0]
    if not u.is_rational_function(x) or u.is_real is not True:
        return None
    direction = rational_value(u, x, point)
    if direction not in (sp.oo, -sp.oo):
        return None
    sign = 1 if direction is sp.oo else -1
    u = sign * u
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    # Mills integration-by-parts bound holds for all eventually positive u.
    order = next(
        (
            n
            for n in range(1, 5)
            if clean(sp.Abs(a) * sp.exp(-u * u) / u ** (2 * n + 1), x, point) == 0
        ),
        None,
    )
    if order is None:
        return None
    tail = (
        sp.exp(-u * u)
        / sp.sqrt(sp.pi)
        * sum(
            (-1) ** k * sp.factorial2(2 * k - 1) / (2**k * u ** (2 * k + 1))
            for k in range(order)
        )
    )
    approximation = (
        tail
        if atom.func is sp.erfc and sign == 1
        else 2 - tail
        if atom.func is sp.erfc
        else sign * (1 - tail)
    )
    value = clean(sp.expand(a * approximation + b), x, point)
    if value is None:
        return None
    return answer(
        value,
        "real_error_rational_tail_remainder",
        f"Real rational argument tends to signed infinity; erfc positive-tail integration by parts retains {order} terms, with absolute remainder bounded by a fixed multiple of exp(-u^2)/u^{2 * order + 1}. The original multiplier times this bound vanishes. Reflection gives negative-tail and erf values.",
    )


def local_real_sign_certificate(expr, x, point, domain, assumptions):
    point = sp.sympify(point)
    if (
        point.is_real is not True
        or point.is_finite is not True
        or domain is not sp.S.true
        or x.is_real is False
        or x.is_integer is True
        or assumptions.has(x)
        or not perturbation_budget(expr, 80)
    ):
        return None
    if (
        x.is_finite is False
        or (x.is_nonnegative is True and point.is_negative is True)
        or (x.is_nonpositive is True and point.is_positive is True)
    ):
        return None
    atoms = expr.atoms(sp.Abs, sp.floor, sp.ceiling, sp.sign, sp.Heaviside)
    if not atoms or len(atoms) > 3:
        return None
    t = sp.Dummy("positive_local_chart", positive=True)
    n = sp.Dummy("attained_local_n", positive=True, integer=True)
    sides = (
        (1,)
        if point == 0 and x.is_nonnegative is True
        else (-1,)
        if point == 0 and x.is_nonpositive is True
        else (1, -1)
    )
    evidence = []
    for side in sides:
        chart = expr.subs(x, point + side * t)
        replacements = {}
        for atom in chart.atoms(sp.Abs, sp.floor, sp.ceiling, sp.sign, sp.Heaviside):
            u = atom.args[0]
            if not u.is_rational_function(t):
                return None
            if any(p.exp.is_Integer and abs(p.exp) > 8 for p in u.atoms(sp.Pow)):
                return None
            if u.is_real is not True:
                try:
                    polynomials = [sp.Poly(v, t) for v in sp.fraction(sp.cancel(u))]
                except sp.PolynomialError:
                    return None
                if any(
                    c.is_real is not True for q in polynomials for c in q.all_coeffs()
                ):
                    return None
            if atom.func in (sp.Abs, sp.sign, sp.Heaviside):
                # Rational leading sign gives an eventually exact identity.
                num, den = map(lambda v: sp.Poly(v, t), sp.fraction(sp.cancel(u)))
                if num.is_zero:
                    replacements[atom] = atom.func(0, *atom.args[1:])
                    continue
                nc = min(num.terms())[1]
                dc = min(den.terms())[1]
                c = nc / dc
                if c.is_positive is True:
                    replacements[atom] = u if atom.func is sp.Abs else sp.S.One
                elif c.is_negative is True:
                    replacements[atom] = (
                        -u
                        if atom.func is sp.Abs
                        else -sp.S.One
                        if atom.func is sp.sign
                        else sp.S.Zero
                    )
                else:
                    return None
            else:
                value = rational_value(u, t, 0)
                if (
                    value is None
                    or value.is_finite is not True
                    or value.is_real is not True
                ):
                    return None
                if value.is_integer is not True:
                    if value.is_number is not True or value.is_integer is not False:
                        return None
                    replacements[atom] = atom.func(value)
                    continue
                delta = sp.cancel(u - value)
                if delta == 0:
                    replacements[atom] = value
                    continue
                num, den = map(lambda v: sp.Poly(v, t), sp.fraction(delta))
                c = min(num.terms())[1] / min(den.terms())[1]
                if c.is_positive is True:
                    replacements[atom] = value if atom.func is sp.floor else value + 1
                elif c.is_negative is True:
                    replacements[atom] = value - 1 if atom.func is sp.floor else value
                else:
                    return None
        leading = chart.xreplace(replacements)
        value = rational_value(leading, t, 0)
        if value is None:
            return None
        evidence.append(
            LimitEvidence(
                "attained_local_real_sign_chart",
                "On this real side rational leading coefficients prove each Abs sign and floor/ceiling interval eventually exactly. x=point+/-1/n attains the resulting limit; nonzero rational denominators have only finitely many zeros and are eventually avoided.",
                ((x, point + side / n),),
                value,
            )
        )
    if len(evidence) == 1:
        return LimitStatus.PROVED, evidence[0].value, tuple(evidence)
    if evidence[0].value == evidence[1].value:
        return LimitStatus.PROVED, evidence[0].value, tuple(evidence)
    if sp.simplify(evidence[0].value - evidence[1].value).is_zero is False:
        return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence)
    return None


def rounded_tail_certificate(expr, x, point):
    from .function_normalization import NearestInteger

    atoms = list(expr.atoms(NearestInteger))
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 80)
    ):
        return None
    atom = atoms[0]
    u = atom.args[0]
    real_u = u
    gamma_atoms = u.atoms(sp.gamma)
    if gamma_atoms and all(g.args[0].is_positive is True for g in gamma_atoms):
        real_u = u.xreplace(
            {g: sp.Dummy("positive_gamma_value", positive=True) for g in gamma_atoms}
        )
    if real_u.is_real is not True:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    if clean(sp.Abs(a), x, point) != 0:
        return None
    value = clean(a * u + b, x, point)
    if value is None:
        return None
    return answer(
        value,
        "nearest_integer_vanishing_error",
        "For real u, source Round selects a nearest integer, so |Round(u)-u|<=1/2 including ties. Its original multiplying coefficient tends to zero; no tie convention is needed for this limit.",
    )


def finite_error_germ_certificate(expr, x, point, domain):
    point = sp.sympify(point)
    if (
        x.is_finite is False
        or x.is_integer is True
        or (x.is_nonnegative is True and point.is_negative is True)
        or (x.is_nonpositive is True and point.is_positive is True)
    ):
        return None
    atoms = expr.atoms(sp.erf, sp.erfc, sp.erfi)
    if point in (sp.oo, -sp.oo) or len(atoms) != 1 or not perturbation_budget(expr, 90):
        return None
    atom = next(iter(atoms))
    u = atom.args[0]
    if not u.is_rational_function(x):
        return None
    center = sp.cancel(u).subs(x, point)
    if center.has(x) or center.is_finite is not True:
        return None
    delta = sp.cancel(u - center)
    if delta == 0:
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data
    order = next(
        (
            n
            for n in range(1, 7)
            if finite_clean(a * delta ** (n + 1), x, point, domain) == 0
        ),
        None,
    )
    if order is None:
        return None
    approximation = atom.func(center)
    for k in range(1, order + 1):
        if atom.func is sp.erfi:
            derivative = (
                2
                / sp.sqrt(sp.pi)
                * (-sp.I) ** (k - 1)
                * sp.hermite(k - 1, sp.I * center)
                * sp.exp(center**2)
            )
        else:
            derivative = (
                (1 if atom.func is sp.erf else -1)
                * 2
                / sp.sqrt(sp.pi)
                * (-1) ** (k - 1)
                * sp.hermite(k - 1, center)
                * sp.exp(-(center**2))
            )
        approximation += derivative * delta**k / sp.factorial(k)
    value = finite_clean(sp.expand(a * approximation + b), x, point, domain)
    if value is None:
        return None
    return answer(
        value,
        "entire_error_rational_local_remainder",
        f"Entire erf/erfc/erfi Taylor coefficients at a fixed finite center are exact Hermite-polynomial derivatives through order {order}. Rational delta tends to zero; the original multiplier times the analytic O(delta^{order + 1}) remainder vanishes on all requested sides.",
    )


def integral_rational_adaptive_tail_certificate(expr, x, point):
    from .special_function_limit_germs import transfer_tail

    atoms = expr.atoms(sp.Si, sp.fresnels, sp.fresnelc)
    if (
        point is not sp.oo
        or x.is_positive is not True
        or len(atoms) != 1
        or not perturbation_budget(expr, 100)
    ):
        return None
    atom = next(iter(atoms))
    u = atom.args[0]
    if (
        not u.is_rational_function(x)
        or u.is_real is not True
        or rational_value(u, x, point) is not sp.oo
    ):
        return None
    data = linear_in(expr, atom)
    if data is None:
        return None
    a, b = data

    def exponent(m):
        return 2 * m + 1 if atom.func is sp.Si else 4 * m + 1

    order = next(
        (m for m in range(1, 5) if clean(sp.Abs(a) / u ** exponent(m), x, point) == 0),
        None,
    )
    if order is None:
        return None
    if atom.func is sp.Si:
        f = sum(
            (-1) ** k * sp.factorial(2 * k) / u ** (2 * k + 1) for k in range(order)
        )
        g = sum(
            (-1) ** k * sp.factorial(2 * k + 1) / u ** (2 * k + 2) for k in range(order)
        )
        approximation = sp.pi / 2 - sp.cos(u) * f - sp.sin(u) * g
    else:
        phase = sp.pi * u * u / 2
        f = sum(
            (-1) ** k
            * sp.factorial2(4 * k - 1)
            / (sp.pi ** (2 * k + 1) * u ** (4 * k + 1))
            for k in range(order)
        )
        g = sum(
            (-1) ** k
            * sp.factorial2(4 * k + 1)
            / (sp.pi ** (2 * k + 2) * u ** (4 * k + 3))
            for k in range(order)
        )
        approximation = (
            sp.Rational(1, 2) - sp.cos(phase) * f - sp.sin(phase) * g
            if atom.func is sp.fresnels
            else sp.Rational(1, 2) + sp.sin(phase) * f - sp.cos(phase) * g
        )
    leading = sp.cancel(a * approximation + b)
    return transfer_tail(
        leading,
        x,
        "integral_rational_adaptive_tail_remainder",
        f"Repeated real integration by parts retains {order} paired Si/Fresnel terms along eventually positive rational u. The remainder is O(u^-{exponent(order)}); its original multiplier vanishes. Nonexistence transfers only from attained distinct subsequences with denominator avoidance.",
    )


def triangle_reciprocal_phase_certificate(expr, x, point, domain, assumptions):
    from .function_normalization import TriangleWave

    atoms = expr.atoms(TriangleWave)
    if (
        point != 0
        or len(atoms) != 1
        or x.is_real is False
        or x.is_integer is True
        or x.is_finite is False
        or assumptions.has(x)
        or assumptions is sp.S.false
        or domain not in (sp.S.true, sp.Gt(x, 0))
        or not perturbation_budget(expr, 70)
    ):
        return None
    if domain == sp.Gt(x, 0) and x.is_nonpositive is True:
        return None
    atom = next(iter(atoms))
    c = sp.cancel(atom.args[0] * x)
    if (
        c.has(x)
        or c.is_finite is not True
        or (c.is_positive is not True and c.is_negative is not True)
    ):
        return None
    w = sp.Dummy("attained_triangle_value")
    transformed = expr.xreplace({atom: w})
    if not transformed.is_rational_function(x, w):
        return None
    side = -1 if x.is_nonpositive is True else 1
    phase_sign = side * (1 if c.is_positive is True else -1)
    n = sp.Dummy("triangle_sequence_n", positive=True, integer=True)
    t = sp.Dummy("triangle_positive_chart", positive=True)
    evidence = []
    for theta in (sp.S.Zero, sp.Rational(1, 8)):
        sequence = side * sp.Abs(c) / (n + theta)
        wave = TriangleWave(phase_sign * theta)
        collapsed = sp.cancel(transformed.subs(w, wave).subs(x, side * t))
        value = rational_value(collapsed, t, 0)
        if value is None:
            return None
        evidence.append(
            LimitEvidence(
                "attained_triangle_reciprocal_subsequence",
                "The real sequence x=side*|c|/(n+theta) approaches zero and attains phase +/-n+/-theta. Unit periodicity gives exact triangle values at theta=0 and 1/8. The specialized rational denominators are nonzero polynomials; finitely many zeros are eventually avoided, including every triangle-induced pole.",
                ((x, sequence),),
                value,
            )
        )
    if (
        evidence[0].value != evidence[1].value
        and sp.simplify(evidence[0].value - evidence[1].value).is_zero is False
    ):
        return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence)
    return None
