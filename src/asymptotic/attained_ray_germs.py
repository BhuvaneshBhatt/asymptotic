"""Bounded Puiseux leading terms on attained elementary rays."""

from functools import lru_cache

import sympy as sp

from ._polynomial_bounds import bounded_degree, bounded_expansion_width
from .limit_models import LimitEvidence, LimitStatus


def _real_coefficient(value):
    return value.is_real is True and value.is_finite is True and value.is_zero is False


def ray_leading_term(expression, parameter):
    """Return an exact c*t**q*(1+o(1)) germ, or an identically zero germ.

    Polynomial coefficients, products and fixed rational powers propagate
    their leading terms. Fractional powers require a positive leading base.
    Analytic scalar heads use their first nonzero constant or linear term.
    Cancellation without a supported next term remains unresolved. No series
    call or agreement of sampled paths is used.
    """

    @lru_cache(maxsize=256)
    def leading(expr):
        if expr == 0:
            return sp.S.Zero, sp.S.Zero
        if expr.is_number:
            return (expr, sp.S.Zero) if _real_coefficient(expr) else None
        if expr == parameter:
            return sp.S.One, sp.S.One
        if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return None
        degree = bounded_degree(expr, (parameter,), 64)
        if degree is not None:
            try:
                polynomial = sp.Poly(expr, parameter, domain=sp.EX)
            except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
                return None
            if polynomial.is_zero:
                return sp.S.Zero, sp.S.Zero
            powers, coefficient = min(polynomial.terms(), key=lambda item: item[0][0])
            return (
                (coefficient, sp.Integer(powers[0]))
                if _real_coefficient(coefficient)
                else None
            )
        if expr.is_Mul:
            parts = [leading(a) for a in expr.args]
            if any(p is None for p in parts):
                return None
            if any(p[0] == 0 for p in parts):
                return sp.S.Zero, sp.S.Zero
            coefficient = sp.Mul(*(p[0] for p in parts))
            return (
                (coefficient, sp.Add(*(p[1] for p in parts)))
                if _real_coefficient(coefficient)
                else None
            )
        if expr.is_Add:
            constant, rest = expr.as_coeff_Add()
            factor, head = rest.as_coeff_Mul()
            if (constant, factor) in ((-1, 1), (1, -1)) and head.func in (
                sp.exp,
                sp.cos,
            ):
                inner = leading(head.args[0])
                if inner is not None and inner[0] != 0 and inner[1] > 0:
                    if head.func is sp.exp:
                        return factor * inner[0], inner[1]
                    return -factor * inner[0] ** 2 / 2, 2 * inner[1]
            parts = [leading(a) for a in expr.args]
            if any(p is None for p in parts):
                return None
            nonzero = [p for p in parts if p[0] != 0]
            if not nonzero:
                return sp.S.Zero, sp.S.Zero
            order = min(p[1] for p in nonzero)
            coefficient = sp.Add(*(p[0] for p in nonzero if p[1] == order))
            return (coefficient, order) if _real_coefficient(coefficient) else None
        if expr.is_Pow:
            if (
                expr.exp.is_Rational is not True
                or abs(expr.exp) > 64
                or expr.exp.q > 64
            ):
                return None
            base = leading(expr.base)
            if base is None:
                return None
            coefficient, order = base
            if coefficient == 0:
                return (sp.S.Zero, sp.S.Zero) if expr.exp > 0 else None
            if expr.exp.is_Integer is not True and coefficient.is_positive is not True:
                return None
            coefficient = coefficient**expr.exp
            return (
                (coefficient, order * expr.exp)
                if _real_coefficient(coefficient)
                else None
            )
        if expr.func is sp.Heaviside:
            from .multivariate_pole_bounds import _locally_real

            if not _locally_real(expr.args[0], (parameter,), (sp.S.Zero,)):
                return None
            inner = leading(expr.args[0])
            if inner is None:
                return None
            if inner[0] == 0:
                value = expr.args[1]
                return (
                    (value, sp.S.Zero)
                    if value == 0 or _real_coefficient(value)
                    else None
                )
            if inner[0].is_positive is True:
                return sp.S.One, sp.S.Zero
            if inner[0].is_negative is True:
                return sp.S.Zero, sp.S.Zero
            return None
        if expr.func in (sp.Abs, sp.sign):
            inner = leading(expr.args[0])
            if inner is None or inner[0] == 0:
                return inner
            coefficient, order = inner
            if expr.func is sp.Abs:
                return sp.Abs(coefficient), order
            if coefficient.is_positive is True:
                return sp.S.One, sp.S.Zero
            if coefficient.is_negative is True:
                return -sp.S.One, sp.S.Zero
            return None
        if expr.func in (
            sp.sin,
            sp.cos,
            sp.sinh,
            sp.cosh,
            sp.exp,
            sp.log,
            sp.atan,
            sp.asin,
            sp.acos,
            sp.tanh,
            sp.erf,
        ):
            argument = expr.args[0]
            inner = leading(argument)
            if inner is None:
                return None
            if inner[1] < 0:
                if expr.func in (sp.atan, sp.tanh, sp.erf):
                    sign = sp.sign(inner[0])
                    return sign * (sp.pi / 2 if expr.func is sp.atan else 1), sp.S.Zero
                return None
            center = sp.S.Zero if inner[0] == 0 or inner[1] > 0 else inner[0]
            if expr.func is sp.log and center.is_positive is not True:
                return None
            value = expr.func(center)
            if _real_coefficient(value):
                return value, sp.S.Zero
            if value == 0:
                displacement = leading(argument - center)
                if displacement is None or displacement[0] == 0:
                    return displacement
                if displacement[1] <= 0:
                    return None
                derivative = {
                    sp.sin: sp.cos(center),
                    sp.cos: -sp.sin(center),
                    sp.sinh: sp.cosh(center),
                    sp.cosh: sp.sinh(center),
                    sp.exp: sp.exp(center),
                    sp.log: sp.S.One / center if center != 0 else None,
                    sp.atan: 1 / (1 + center**2),
                    sp.asin: 1 / sp.sqrt(1 - center**2)
                    if center not in (-1, 1)
                    else None,
                    sp.acos: -1 / sp.sqrt(1 - center**2)
                    if center not in (-1, 1)
                    else None,
                    sp.tanh: 1 / sp.cosh(center) ** 2,
                    sp.erf: 2 * sp.exp(-(center**2)) / sp.sqrt(sp.pi),
                }[expr.func]
                if derivative is None:
                    return None
                coefficient = derivative * displacement[0]
                return (
                    (coefficient, displacement[1])
                    if _real_coefficient(coefficient)
                    else None
                )
        return None

    return leading(expression)


def logarithmic_ray_value(expression, parameter):
    """Evaluate an exact rational germ in t and log(t) on t=1/j.

    Every logarithm must have an exactly positive monomial argument. Leading
    powers of t dominate all log powers; at equal t order, the largest log
    power dominates. This certifies a single attained path, never a joint limit.
    """
    logarithms = expression.atoms(sp.log)
    if (
        not logarithms
        or len(logarithms) > 4
        or bounded_expansion_width(expression, 64) > 64
    ):
        return None
    logarithm = sp.Dummy("ray_logarithm", real=True)
    replacements = {}
    for atom in logarithms:
        argument = atom.args[0]
        if bounded_degree(argument, (parameter,), 64) is None:
            return None
        polynomial = sp.Poly(argument, parameter, domain=sp.EX)
        if len(polynomial.terms()) != 1:
            return None
        powers, coefficient = polynomial.terms()[0]
        if coefficient.is_positive is not True or coefficient.is_finite is not True:
            return None
        replacements[atom] = sp.log(coefficient) + powers[0] * logarithm
    translated = expression.xreplace(replacements)
    numerator, denominator = translated.as_numer_denom()
    leading = []
    for part in (numerator, denominator):
        if bounded_degree(part, (parameter, logarithm), 64) is None:
            return None
        polynomial = sp.Poly(part, parameter, logarithm, domain=sp.EX)
        if polynomial.is_zero:
            return sp.S.Zero if part == numerator else None
        powers, coefficient = min(
            polynomial.terms(), key=lambda term: (term[0][0], -term[0][1])
        )
        if not _real_coefficient(coefficient):
            return None
        leading.append((coefficient, powers))
    coefficient = leading[0][0] / leading[1][0]
    power = leading[0][1][0] - leading[1][1][0]
    log_power = leading[0][1][1] - leading[1][1][1]
    if power > 0 or (power == 0 and log_power < 0):
        return sp.S.Zero
    if power == 0 and log_power == 0:
        return coefficient
    signed = coefficient * (-1) ** log_power
    if signed.is_positive is True:
        return sp.oo
    if signed.is_negative is True:
        return -sp.oo
    return None


def bounded_ray_value(expression, parameter):
    """Evaluate a ray after bounding oscillatory terms that vanish uniformly."""
    from .uniform_radial_bounds import _upper_order

    value = sp.S.Zero
    for term in sp.Add.make_args(expression):
        leading = ray_leading_term(term, parameter)
        if leading is not None and leading[1] >= 0:
            if leading[1] == 0:
                value += leading[0]
            continue
        estimate = _upper_order(term, (parameter,))
        if estimate is None or not (
            estimate[2]
            or estimate[0].is_positive is True
            or (estimate[0] == 0 and estimate[1] < 0)
        ):
            return None
    return value


def _different(a, b):
    if a == b:
        return False
    if a in (sp.oo, -sp.oo) or b in (sp.oo, -sp.oo):
        return True
    difference = a - b
    if difference.is_algebraic is True and difference.is_zero is False:
        return True
    # A nonzero rational phase cannot be a multiple of 2*pi.
    for constant, oscillatory in ((a, b), (b, a)):
        if (
            constant == 1
            and oscillatory.func is sp.cos
            and oscillatory.args[0].is_Rational is True
            and oscillatory.args[0] != 0
        ):
            return True
    return False


def elementary_ray_conflict(expr, variables, target, domain, assumptions):
    """Prove nonexistence with two explicit, eventually defined real rays.

    Leading-term calculus certifies each attained value and the nonvanishing
    of every original denominator and logarithm argument. Piecewise branches
    must resolve exactly on the ray. Failure of this bounded search remains
    unresolved; equal ray limits never prove a joint limit.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 80
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(p.is_Rational is not True for p in target)
    ):
        return None
    allowed = {
        sp.sin,
        sp.cos,
        sp.sinh,
        sp.cosh,
        sp.exp,
        sp.log,
        sp.atan,
        sp.asin,
        sp.acos,
        sp.tanh,
        sp.erf,
        sp.Abs,
        sp.sign,
        sp.Piecewise,
        sp.Heaviside,
    }
    if any(atom.func not in allowed for atom in expr.atoms(sp.Function)):
        return None
    if any(
        p.exp.is_Rational is not True or abs(p.exp) > 64 or p.exp.q > 64
        for p in expr.atoms(sp.Pow)
    ):
        return None
    guards = [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]
    guards.extend(1 + a.args[0] ** 2 for a in expr.atoms(sp.atan))
    logarithms = [a.args[0] for a in expr.atoms(sp.log)]
    directions = [
        tuple(sp.S.One if i == k else sp.S.Zero for i in range(len(variables)))
        for k in range(len(variables))
    ]
    directions.append((sp.S.One,) * len(variables))
    t = sp.Dummy("ray_germ_parameter", positive=True)
    j = sp.Dummy("attained_ray_index", positive=True, integer=True)
    witnessed = []
    paths = [tuple(c * t for c in direction) for direction in directions]
    paths.extend(tuple(-c * t for c in direction) for direction in directions)
    if len(variables) == 2:
        paths.extend(((t, t**2), (t**2, t)))
    for path in paths:
        mapping = dict(
            zip(
                variables,
                (p + delta for p, delta in zip(target, path, strict=True)),
                strict=True,
            )
        )
        original_guards = [
            ray_leading_term(g.subs(mapping, simultaneous=True), t) for g in guards
        ]
        if any(g is None or g[0] == 0 for g in original_guards):
            continue
        log_guards = [
            ray_leading_term(g.subs(mapping, simultaneous=True), t) for g in logarithms
        ]
        if any(g is None or g[0].is_positive is not True for g in log_guards):
            continue
        along = expr.subs(mapping, simultaneous=True)
        if along.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            continue
        term = ray_leading_term(along, t)
        if term is None:
            value = logarithmic_ray_value(along, t)
            if value is None:
                value = bounded_ray_value(along, t)
            if value is None:
                continue
        else:
            coefficient, order = term
            if coefficient == 0 or order > 0:
                value = sp.S.Zero
            elif order == 0:
                value = coefficient
            elif coefficient.is_positive is True:
                value = sp.oo
            elif coefficient.is_negative is True:
                value = -sp.oo
            else:
                continue
        sequence = tuple((v, mapping[v].subs(t, 1 / j)) for v in variables)
        evidence = LimitEvidence(
            "attained_elementary_ray",
            "Every coordinate follows the displayed t=1/j ray. Exact polynomial/Puiseux or rational logarithmic leading terms and local analytic transfer give the stated limit, and nonzero leading original denominator terms exclude poles for all sufficiently large j. Every original logarithm argument is positive eventually. Piecewise conditions resolve exactly on this ray. Distinct attained limits prove nonexistence; no agreement of rays is used to prove existence.",
            sequence,
            value,
        )
        for previous in witnessed:
            if _different(previous.value, value):
                return LimitStatus.DOES_NOT_EXIST, None, (previous, evidence)
        witnessed.append(evidence)
    return None


def exponential_linear_pole_conflict(expr, variables, target, domain, assumptions):
    """Separate growth and decay rays of an exponential at an odd linear pole.

    For a nonzero rational alpha and an odd m, alpha/L**m changes sign
    across the real linear form L=0. Exact rational cofactor germs survive
    neither exponential growth nor decay. Original polynomial denominator
    guards prove that both displayed subsequences are eventually defined.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 45
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(p.is_Rational is not True for p in target)
    ):
        return None
    atoms = expr.atoms(sp.exp)
    if len(atoms) != 1:
        return None
    exponential = next(iter(atoms))
    factors = sp.Mul.make_args(expr)
    if exponential not in factors:
        return None
    coefficient, denominator = exponential.args[0].as_numer_denom()
    linear, power = denominator.as_base_exp()
    if (
        coefficient.is_Rational is not True
        or coefficient == 0
        or power.is_Integer is not True
        or not 1 <= power <= 9
        or power.is_even
    ):
        return None
    if bounded_degree(linear, variables, 1) is None:
        return None
    cofactor = sp.Mul(*(factor for factor in factors if factor != exponential))
    numerator, denominator = cofactor.as_numer_denom()
    if any(bounded_degree(p, variables, 32) is None for p in (numerator, denominator)):
        return None
    try:
        form = sp.Poly(linear, *variables, domain=sp.QQ)
        sp.Poly(numerator, *variables, domain=sp.QQ)
        sp.Poly(denominator, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    point = dict(zip(variables, target, strict=True))
    if form.total_degree() != 1 or form.as_expr().subs(point) != 0:
        return None
    directions = [
        tuple(sp.S.One if i == k else sp.S.Zero for i in range(len(variables)))
        for k in range(len(variables))
    ]
    directions.append((sp.S.One,) * len(variables))
    directions.extend(
        tuple(sp.S.One + (i == k) for i in range(len(variables)))
        for k in range(len(variables))
    )
    guards = [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]
    t = sp.Dummy("exponential_ray_parameter", positive=True)
    j = sp.Dummy("attained_exponential_index", positive=True, integer=True)
    for direction in directions:
        slope = sum(
            form.coeff_monomial(v) * c
            for v, c in zip(variables, direction, strict=True)
        )
        if slope == 0:
            continue
        evidence = []
        for orientation in (sp.S.One, -sp.S.One):
            mapping = dict(
                zip(
                    variables,
                    (
                        p + orientation * c * t
                        for p, c in zip(target, direction, strict=True)
                    ),
                    strict=True,
                )
            )
            terms = [
                ray_leading_term(g.subs(mapping, simultaneous=True), t) for g in guards
            ]
            if any(term is None or term[0] == 0 for term in terms):
                break
            term = ray_leading_term(cofactor.subs(mapping, simultaneous=True), t)
            if term is None or term[0] == 0:
                break
            rate = coefficient / (orientation * slope) ** power
            if rate < 0:
                value = sp.S.Zero
            elif term[0].is_positive is True:
                value = sp.oo
            elif term[0].is_negative is True:
                value = -sp.oo
            else:
                break
            sequence = tuple((v, mapping[v].subs(t, 1 / j)) for v in variables)
            evidence.append(
                LimitEvidence(
                    "attained_exponential_pole_ray",
                    "The linear pole equals orientation*slope/j, nonzero for every positive integer j. Its odd power gives opposite exponential rates on the two rays. A certified nonzero rational cofactor germ and all original denominator guards remain defined eventually. Exponential decay dominates every cofactor power, while growth dominates it with the displayed cofactor sign.",
                    sequence,
                    value,
                )
            )
        if len(evidence) == 2:
            return LimitStatus.DOES_NOT_EXIST, None, tuple(evidence)
    return None
