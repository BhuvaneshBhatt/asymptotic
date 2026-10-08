"""Uniform real poles and bounded logarithmic divided differences."""

import sympy as sp

from ._limit_composition import _regular_composition
from ._polynomial_bounds import bounded_degree, bounded_expansion_width
from .attained_ray_germs import ray_leading_term
from .limit_models import LimitEvidence, LimitStatus
from .multivariate_pole_bounds import _locally_real


def _eligible(expr, variables, target, domain, assumptions):
    return (
        isinstance(expr, sp.Expr)
        and len(variables) in (2, 3)
        and domain is sp.S.true
        and assumptions is sp.S.true
        and not expr.free_symbols - set(variables)
        and not expr.has(sp.Float)
        and sp.count_ops(expr) <= 60
        and all(
            p.exp.is_Rational is True and abs(p.exp) <= 16 and p.exp.q <= 16
            for p in expr.atoms(sp.Pow)
        )
        and bounded_expansion_width(expr, 64) <= 64
        and all(
            v.assumptions0 == sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        and all(p.is_Rational is True for p in target)
    )


def _attained_guards(expr, variables, target, replacements=None):
    t = sp.Dummy("real_pole_parameter", positive=True)
    j = sp.Dummy("attained_real_pole_index", positive=True, integer=True)
    guards = [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]
    guards.extend(a.args[0] for a in expr.atoms(sp.log))
    for weights in [
        (1,) * len(variables),
        tuple(range(1, len(variables) + 1)),
        *(
            tuple(1 if i == k else 0 for i in range(len(variables)))
            for k in range(len(variables))
        ),
    ]:
        mapping = dict(
            zip(
                variables,
                (p + c * t for p, c in zip(target, weights, strict=True)),
                strict=True,
            )
        )
        terms = [
            ray_leading_term(
                g.xreplace(replacements or {}).subs(mapping, simultaneous=True), t
            )
            for g in guards
        ]
        if all(term is not None and term[0] != 0 for term in terms):
            return tuple((v, mapping[v].subs(t, 1 / j)) for v in variables)
    return None


def signed_real_pole_certificate(expr, variables, target, domain, assumptions):
    """Prove full-domain poles from exact two-sided monomials or sign cancellation.

    A cylindrical scalar germ is checked on both real half-neighborhoods,
    which exhaust its defined local domain. For a multivariate polynomial h,
    sign(h)**k/h**m equals |h|**(-m) when k-m is even and h!=0.
    A continuous nonzero real cofactor fixes the sign of the resulting pole.
    """
    if not _eligible(expr, variables, target, domain, assumptions):
        return None
    active = tuple(v for v in variables if expr.has(v))
    if len(active) == 1 and all(
        atom.func in (sp.Abs, sp.sign) or not atom.has(active[0])
        for atom in expr.atoms(sp.Function)
    ):
        variable = active[0]
        center = target[variables.index(variable)]
        u = sp.Dummy("scalar_pole_magnitude", positive=True)
        values = []
        for orientation in (1, -1):
            branch = sp.powsimp(expr.subs(variable, center + orientation * u))
            coefficient, power = branch.as_coeff_exponent(u)
            if (
                coefficient.free_symbols
                or coefficient.is_real is not True
                or coefficient.is_finite is not True
                or coefficient.is_zero is not False
                or power.is_Rational is not True
                or not -64 <= power < 0
            ):
                break
            if coefficient.is_positive is True:
                values.append(sp.oo)
            elif coefficient.is_negative is True:
                values.append(-sp.oo)
            else:
                break
        if len(values) == 2 and values[0] == values[1]:
            sequence = _attained_guards(expr, variables, target)
            if sequence is not None:
                evidence = LimitEvidence(
                    "two_sided_real_monomial_pole",
                    "The expression is independent of the other coordinates. On each real half-neighborhood it is exactly a nonzero real constant times |v-p|**q with q<0, and both constants have the same sign. These half-neighborhoods exhaust the defined local domain, so all simultaneous approaches diverge with that sign. The displayed sequence avoids every original denominator eventually.",
                    sequence,
                    values[0],
                )
                return LimitStatus.PROVED, values[0], evidence
    for sign in sorted(expr.atoms(sp.sign), key=sp.default_sort_key):
        h = sign.args[0]
        k = expr.as_powers_dict().get(sign)
        if k is None or k.is_Integer is not True or abs(k) > 8:
            continue
        if (
            bounded_degree(h, variables, 8) is None
            or _regular_composition(h, variables, target) != 0
        ):
            continue
        try:
            polynomial = sp.Poly(h, *variables, domain=sp.QQ)
        except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
            continue
        if polynomial.is_zero:
            continue
        for m in range(1, 9):
            if (k - m) % 2:
                continue
            cofactor = sp.cancel(expr * h**m / sign**k)
            if not _locally_real(cofactor, variables, target):
                continue
            value = _regular_composition(cofactor, variables, target)
            if (
                value is None
                or value.is_finite is not True
                or (value.is_positive is not True and value.is_negative is not True)
            ):
                continue
            sequence = _attained_guards(expr, variables, target)
            if sequence is None:
                continue
            infinity = sp.oo if value.is_positive is True else -sp.oo
            evidence = LimitEvidence(
                "uniform_signed_polynomial_pole",
                "On the original defined domain, h!=0 and sign(h)**k/h**m equals |h|**(-m) because k-m is even. The continuous real cofactor has a nonzero signed center. Since h tends uniformly to zero, every defined approach has the displayed signed divergence. Nonzero leading original denominator terms certify the attained sequence.",
                sequence,
                infinity,
            )
            return LimitStatus.PROVED, infinity, evidence
    return None


def positive_logarithmic_pole_certificate(expr, variables, target, domain, assumptions):
    """Transfer uniform growth through the real logarithm without coercivity.

    The numerator stays positive and the nonnegative polynomial denominator
    tends uniformly to zero. On the original defined domain the denominator
    is strictly positive, so their ratio and its logarithm tend to infinity.
    """
    if (
        not _eligible(expr, variables, target, domain, assumptions)
        or expr.func is not sp.log
    ):
        return None
    numerator, denominator = expr.args[0].as_numer_denom()
    if any(
        bounded_degree(part, variables, 32) is None for part in (numerator, denominator)
    ):
        return None
    try:
        sp.Poly(numerator, *variables, domain=sp.QQ)
        polynomial = sp.Poly(denominator, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    center = _regular_composition(numerator, variables, target)
    if (
        polynomial.is_zero
        or denominator.is_nonnegative is not True
        or center is None
        or center.is_positive is not True
        or _regular_composition(denominator, variables, target) != 0
    ):
        return None
    sequence = _attained_guards(expr, variables, target)
    if sequence is None:
        return None
    evidence = LimitEvidence(
        "uniform_positive_logarithmic_pole",
        "The real numerator is bounded below by a positive constant near the center. The nonnegative polynomial denominator tends uniformly to zero and is positive at every original defined point. The ratio therefore tends uniformly to positive infinity, and the increasing real logarithm preserves that divergence. Original denominator and logarithm guards are nonzero on the displayed attained sequence eventually.",
        sequence,
        sp.oo,
    )
    return LimitStatus.PROVED, sp.oo, evidence


def logarithmic_unit_vanishing_certificate(
    expr, variables, target, domain, assumptions
):
    """Bound |h/log(1+h)| locally before transferring a vanishing cofactor.

    For real h near zero, |log(1+h)|>=|h|/2 on h!=0. A positive
    integer power of this divided difference is bounded. The original
    logarithm's zero set is retained when constructing the attained approach.
    """
    if not _eligible(expr, variables, target, domain, assumptions):
        return None
    from .uniform_radial_bounds import radial_vanishing_certificate

    for atom in sorted(expr.atoms(sp.log), key=sp.default_sort_key):
        h = atom.args[0] - 1
        exponent = expr.as_powers_dict().get(atom)
        if (
            exponent is None
            or exponent.is_Integer is not True
            or not -8 <= exponent < 0
        ):
            continue
        if (
            not _locally_real(h, variables, target)
            or _regular_composition(h, variables, target) != 0
        ):
            continue
        for factor in (sp.Abs(h), h):
            cofactor = sp.cancel(expr * atom ** (-exponent) / factor ** (-exponent))
            zero = radial_vanishing_certificate(
                cofactor, variables, target, domain, assumptions
            )
            if zero is None:
                continue
            sequence = _attained_guards(expr, variables, target, {atom: h})
            if sequence is None:
                continue
            evidence = (
                zero[2],
                LimitEvidence(
                    "bounded_logarithmic_divided_difference",
                    "The real inner argument h tends uniformly to zero. On the original domain h!=0 and |log(1+h)|>=|h|/2 eventually, so the extracted positive integer power of h/log(1+h), or its absolute-numerator variant, stays bounded. Its remaining cofactor tends uniformly to zero. The displayed attained sequence avoids h=0 and every other original denominator.",
                    sequence,
                    sp.S.Zero,
                ),
            )
            return LimitStatus.PROVED, sp.S.Zero, evidence
    return None
