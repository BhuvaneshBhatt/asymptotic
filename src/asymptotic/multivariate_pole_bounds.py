"""Elementary multivariate bounds and exact principal-root cancellations."""

import sympy as sp

from ._limit_composition import _regular_composition
from ._polynomial_bounds import bounded_degree
from .domain_witnesses import attained_ray
from .limit_models import LimitEvidence, LimitStatus


def _real_polynomial(expression, variables):
    try:
        polynomial = sp.Poly(expression, *variables)
    except sp.PolynomialError:
        return None
    if all(c.is_real is True and c.is_finite is True for c in polynomial.coeffs()):
        return polynomial
    return None


def _denominator_constraints(expression, variables):
    """Prove nonnegativity and return sufficient polynomial ray constraints.

    Positive even monomials have a coefficient-independent zero set. Absolute
    polynomials and even powers supply additional nonnegative factors. The
    returned constraints are used only to construct a defined approach, never
    as a necessary characterization of the whole denominator domain.
    """
    if expression.is_number and expression.is_positive is True:
        return []
    if expression.is_Add or expression.is_Mul:
        constraints = []
        for part in expression.args:
            result = _denominator_constraints(part, variables)
            if result is None:
                return None
            constraints.extend(result)
        return constraints
    if expression.func is sp.Abs:
        polynomial = _real_polynomial(expression.args[0], variables)
        return None if polynomial is None else [sp.Ne(polynomial.as_expr(), 0)]
    if expression.is_Pow:
        base, exponent = expression.args
        if exponent.is_Rational is not True or exponent <= 0:
            return None
        constraints = _denominator_constraints(base, variables)
        if constraints is not None:
            return constraints
        if exponent.is_even is True:
            polynomial = _real_polynomial(base, variables)
            return None if polynomial is None else [sp.Ne(polynomial.as_expr(), 0)]
        return None
    polynomial = _real_polynomial(expression, variables)
    if polynomial is None or polynomial.is_zero:
        return None
    if any(
        c.is_positive is not True or any(k % 2 for k in m)
        for m, c in polynomial.terms()
    ):
        return None
    support = sp.Add(
        *(
            sp.Mul(*(v**k for v, k in zip(variables, m, strict=True)))
            for m, c in polynomial.terms()
        )
    )
    return [sp.Ne(support, 0)]


def _locally_real(expression, variables, target):
    """Check the real codomain separately from continuity at the center."""
    if expression.is_real is True:
        return True
    if expression.is_Atom:
        return False
    if expression.is_Add or expression.is_Mul:
        return all(_locally_real(a, variables, target) for a in expression.args)
    if expression.is_Pow:
        base, exponent = expression.args
        if not _locally_real(base, variables, target):
            return False
        if exponent.is_integer is True:
            return True
        center = base.subs(dict(zip(variables, target, strict=True)))
        return exponent.is_Rational is True and (
            base.is_nonnegative is True or center.is_positive is True
        )
    real_functions = {
        sp.sin,
        sp.cos,
        sp.sinh,
        sp.cosh,
        sp.exp,
        sp.tan,
        sp.atan,
        sp.Abs,
        sp.erf,
        sp.erfc,
    }
    if expression.func in real_functions:
        return all(_locally_real(a, variables, target) for a in expression.args)
    if expression.func is sp.log:
        argument = expression.args[0]
        center = argument.subs(dict(zip(variables, target, strict=True)))
        return center.is_positive is True and _locally_real(argument, variables, target)
    return False


def regular_numerator_pole_certificate(
    expr, variables, target, domain, assumptions=sp.S.true
):
    """Certify a signed real infinity with an attained, denominator-safe ray."""
    if (
        domain is not sp.S.true
        or assumptions is not sp.S.true
        or not 2 <= len(variables) <= 3
        or not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 80
        or any(v.is_real is not True or v.is_integer is True for v in variables)
        or any(p.is_real is not True or p.is_finite is not True for p in target)
    ):
        return None
    # Rational(Float) preserves the represented binary value; QQ's approximate
    # float conversion would change the polynomial domain certificate.
    exact = {f: sp.Rational(f) for f in expr.atoms(sp.Float)}
    expr = expr.xreplace(exact)
    target = tuple(
        p.xreplace({f: sp.Rational(f) for f in p.atoms(sp.Float)}) for p in target
    )
    u = tuple(sp.Dummy("pole_coordinate", real=True) for v in variables)
    shifted = expr.subs(
        dict(
            zip(variables, (p + w for p, w in zip(target, u, strict=True)), strict=True)
        ),
        simultaneous=True,
    )
    numerator, denominator = sp.fraction(sp.together(shifted))
    zero = dict.fromkeys(u, sp.S.Zero)
    if denominator.subs(zero) != 0:
        return None
    constraints = _denominator_constraints(denominator, u)
    if constraints is None:
        return None
    if not _locally_real(numerator, u, tuple(zero.values())):
        return None
    center = _regular_composition(numerator, u, tuple(zero.values()))
    if center is None or center.is_finite is not True:
        return None
    value = (
        sp.oo
        if center.is_positive is True
        else (-sp.oo if center.is_negative is True else None)
    )
    if value is None:
        return None
    for v, p, w in zip(variables, target, u, strict=True):
        if v.is_nonnegative is True and p.is_nonnegative is not True:
            return None
        if v.is_nonpositive is True and p.is_nonpositive is not True:
            return None
        if p == 0:
            if v.is_positive is True:
                constraints.append(w > 0)
            elif v.is_negative is True:
                constraints.append(w < 0)
            elif v.is_nonnegative is True:
                constraints.append(w >= 0)
            elif v.is_nonpositive is True:
                constraints.append(w <= 0)
        if v.is_zero is True:
            constraints.append(sp.Eq(w, 0))
    constraints.append(sp.Ne(sp.Add(*(w * w for w in u)), 0))
    direction = attained_ray(sp.And(*constraints), u, tuple(zero.values()))
    if direction is None:
        return None
    j = sp.Dummy("attained_pole_index", positive=True, integer=True)
    substitutions = tuple(
        (v, p + d / j) for v, p, d in zip(variables, target, direction, strict=True)
    )
    evidence = LimitEvidence(
        "regular_numerator_positive_denominator_pole",
        "The numerator is real and continuous near the target with a nonzero signed value. "
        "The denominator is continuous and nonnegative and tends to zero, so it is strictly "
        "positive on every defined quotient approach. The attained polynomial ray has "
        "eventual nonzero denominators by its first nonzero polynomial coefficients; "
        "the regular numerator is also eventually defined there. Every admissible "
        "approach therefore diverges with the same real sign.",
        substitutions,
        value,
    )
    return LimitStatus.PROVED, value, evidence


def _positive_monomial(term, variables):
    coefficient, factors = term.as_coeff_Mul()
    if coefficient.is_Rational is not True or coefficient <= 0:
        return None
    powers = [sp.S.Zero] * len(variables)
    for factor in sp.Mul.make_args(factors):
        base, exponent = factor.as_base_exp()
        if exponent.is_Rational is not True or exponent < 0:
            return None
        if base.func is sp.Abs and base.args[0] in variables:
            powers[variables.index(base.args[0])] += exponent
        elif base in variables and exponent.is_even is True:
            powers[variables.index(base)] += exponent
        else:
            return None
    return coefficient, tuple(powers)


def _positive_product_bounds(numerator, denominator, variables):
    """Bound polynomial terms using one monomial from each positive factor.

    Each sum factor dominates every positive summand. Products preserve these
    inequalities, including positive rational powers on the defined domain.
    The bounded Cartesian search never expands the denominator polynomial.
    """
    from itertools import product

    factors = sp.Mul.make_args(denominator)
    if not 2 <= len(factors) <= 4:
        return None
    choices = []
    search_size = 1
    for factor in factors:
        base, power = factor.as_base_exp()
        if power.is_Rational is not True or power <= 0:
            return None
        terms = sp.Add.make_args(base)
        if len(terms) > 4:
            return None
        monomials = [_positive_monomial(term, variables) for term in terms]
        if any(m is None for m in monomials):
            return None
        search_size *= len(monomials)
        if search_size > 256:
            return None
        choices.append(
            [
                (c**power, tuple(power * e for e in exponents))
                for c, exponents in monomials
            ]
        )
    if bounded_degree(numerator, variables, 12) is None:
        return None
    try:
        polynomial = sp.Poly(numerator, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    candidates = []
    for selection in product(*choices):
        coefficient = sp.Mul(*(c for c, _ in selection))
        exponents = tuple(
            sum(p[i] for _, p in selection) for i in range(len(variables))
        )
        candidates.append((coefficient, exponents))
    bounds = []
    for exponents, coefficient in polynomial.terms():
        for scale, powers in candidates:
            excess = tuple(n - d for n, d in zip(exponents, powers, strict=True))
            if all(e >= 0 for e in excess) and any(e > 0 for e in excess):
                bounds.append(
                    sp.Abs(coefficient)
                    / scale
                    * sp.Mul(
                        *(
                            sp.Abs(v) ** e
                            for v, e in zip(variables, excess, strict=True)
                        )
                    )
                )
                break
        else:
            return None
    return sp.Add(*bounds) if bounds else None


def positive_sum_vanishing_certificate(expr, variables, target, domain, assumptions):
    """Bound each polynomial term by one positive denominator monomial.

    For D=sum(d_j*m_j)>=d_j*m_j and p>0, a numerator monomial whose
    exponents dominate those of m_j**p is bounded by the remaining absolute
    powers. Positive excess order makes that bound vanish uniformly on D>0.
    Signed absolute norms reduce to the same denominator on their defined
    domain. Exponential denominators use exp(u)-1>=u for u>=0. A positive
    diagonal sequence certifies accumulation of the original defined domain.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 60
        or expr.free_symbols - set(variables)
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    numerator, denominator = sp.fraction(expr)
    product_bound = _positive_product_bounds(numerator, denominator, variables)
    if product_bound is not None:
        index = sp.Dummy("positive_product_index", positive=True, integer=True)
        evidence = LimitEvidence(
            "positive_product_bound",
            "Every denominator factor is a positive sum raised to a positive rational "
            "power, and dominates a selected monomial on the defined domain. "
            f"The absolute quotient is bounded by {product_bound}, which tends to zero. "
            "The attained positive diagonal keeps every original factor nonzero.",
            tuple((v, 1 / index) for v in variables),
            sp.S.Zero,
        )
        return LimitStatus.PROVED, sp.S.Zero, evidence
    base, power = denominator.as_base_exp()
    factor = sp.S.One
    # On the original defined domain, a nonnegative norm has sign one.
    if denominator.is_Mul and len(denominator.args) == 2:
        sign = next((a for a in denominator.args if a.func is sp.sign), None)
        if sign is not None:
            radial = denominator / sign
            radial_base, radial_power = radial.as_base_exp()
            if radial_base == sp.Abs(sign.args[0]):
                base, power = sign.args[0], radial_power
    if power.is_Rational is not True or power <= 0:
        return None
    constant, growth = base.as_coeff_Add()
    if constant == -1:
        if growth.func is sp.exp:
            base = growth.args[0]
        elif growth.is_Pow and growth.base.is_Rational and growth.base > 1:
            base, factor = growth.exp, sp.log(growth.base)
    terms = [_positive_monomial(term, variables) for term in sp.Add.make_args(base)]
    if (
        any(term is None for term in terms)
        or base.subs(dict.fromkeys(variables, 0)) != 0
    ):
        return None
    if bounded_degree(numerator, variables, 12) is None:
        return None
    try:
        polynomial = sp.Poly(numerator, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    bounds = []
    for exponents, coefficient in polynomial.terms():
        choice = next(
            (
                (
                    c,
                    tuple(
                        sp.Integer(n) - power * d
                        for n, d in zip(exponents, powers, strict=True)
                    ),
                )
                for c, powers in terms
                if all(
                    sp.Integer(n) - power * d >= 0
                    for n, d in zip(exponents, powers, strict=True)
                )
                and any(
                    sp.Integer(n) - power * d > 0
                    for n, d in zip(exponents, powers, strict=True)
                )
            ),
            None,
        )
        if choice is None:
            return None
        c, excess = choice
        bounds.append(
            sp.Abs(coefficient)
            / (factor * c) ** power
            * sp.Mul(*(sp.Abs(v) ** e for v, e in zip(variables, excess, strict=True)))
        )
    if not bounds:
        return None
    index = sp.Dummy("positive_sum_index", positive=True, integer=True)
    sequence = tuple((v, 1 / index) for v in variables)
    evidence = LimitEvidence(
        "positive_sum_bound",
        "The nonnegative norm has sign one on its defined germ; exponential "
        "denominators satisfy exp(u)-1>=u for u>=0. Each numerator term is bounded "
        "by a positive denominator monomial and the fixed positive scale. "
        f"Their sum is bounded by {sp.Add(*bounds)}, which tends uniformly to zero. "
        "The attained positive diagonal t=1/j keeps every denominator monomial positive.",
        sequence,
        sp.S.Zero,
    )
    return LimitStatus.PROVED, sp.S.Zero, evidence


def radical_quotient_certificate(expr, variables, target, domain, assumptions):
    """Cancel a rational polynomial in coordinate principal square roots.

    Replacing x=u**2 and sqrt(x)=u preserves an exact polynomial identity on
    every principal-root branch. A polynomial quotient tends to its constant
    coefficient because all roots tend to zero. A bounded positive ray search
    separately proves accumulation of the original denominator domain.
    """
    from itertools import product

    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or expr.free_symbols - set(variables)
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    roots = tuple(sp.sqrt(v) for v in variables)
    if not any(expr.has(root) for root in roots):
        return None
    auxiliary = tuple(sp.Dummy("root_coordinate", real=True) for _ in variables)
    replacements = dict(zip(roots, auxiliary, strict=True))
    replacements.update((v, u * u) for v, u in zip(variables, auxiliary, strict=True))
    numerator, denominator = sp.fraction(expr)
    transformed_num = numerator.xreplace(replacements)
    transformed_den = denominator.xreplace(replacements)
    if (
        bounded_degree(transformed_num, auxiliary, 12) is None
        or bounded_degree(transformed_den, auxiliary, 12) is None
    ):
        return None
    try:
        num = sp.Poly(transformed_num, *auxiliary, domain=sp.QQ)
        den = sp.Poly(transformed_den, *auxiliary, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    if den.is_zero or den.total_degree() > 12 or num.total_degree() > 12:
        return None
    quotient, remainder = num.div(den)
    if not remainder.is_zero:
        return None
    t = sp.Dummy("root_ray_parameter", positive=True)
    direction = None
    for ray in product((1, 2, 3), repeat=len(variables)):
        along = den.as_expr().subs(
            dict(zip(auxiliary, (c * t for c in ray), strict=True))
        )
        if along != 0:
            direction = ray
            break
    if direction is None:
        return None
    value = quotient.as_expr().subs(dict.fromkeys(auxiliary, 0))
    index = sp.Dummy("attained_root_index", positive=True, integer=True)
    sequence = tuple(
        (v, sp.Integer(c) ** 2 / index**2)
        for v, c in zip(variables, direction, strict=True)
    )
    evidence = LimitEvidence(
        "radical_polynomial_quotient",
        "Exact polynomial division in principal roots removes the denominator on its "
        "defined domain. Every coordinate root tends to zero, so the quotient tends "
        "to its constant coefficient. The attained positive square-ray denominator "
        "is a nonzero polynomial in 1/j and is nonzero for all sufficiently large j.",
        sequence,
        value,
    )
    return LimitStatus.PROVED, value, evidence


def removable_germ_certificate(expr, variables, target, domain, assumptions):
    """Certify elementary removable quotients with a continuous cofactor.

    The exponential, sine, hyperbolic sine and tangent quotients extend
    holomorphically at zero. The exact identity
    sin(h)*tan(h)/(1-cos(h))=(1+cos(h))/cos(h) supplies the value two.
    Attained rays avoid every original denominator; the trigonometric identity
    uses the isolated zero of 1-cos(h) at h=0 in a small complex neighborhood.
    """
    from .local_path_witnesses import _nonzero_germ

    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(
            power.exp.is_Integer and abs(power.exp) > 12 for power in expr.atoms(sp.Pow)
        )
    ):
        return None
    candidates = []
    factors = expr.as_powers_dict()
    for atom in expr.atoms(sp.sin):
        inner = atom.args[0]
        tangent, denominator = sp.tan(inner), 1 - sp.cos(inner)
        if (
            factors.get(atom) == 1
            and factors.get(tangent) == 1
            and factors.get(denominator) == -1
        ):
            cofactor = sp.Mul(
                *(
                    base**power
                    for base, power in factors.items()
                    if base not in (atom, tangent, denominator)
                )
            )
            candidates.append((inner, cofactor, sp.Integer(2), denominator))
    functions = expr.atoms(sp.Function)
    if len(functions) == 1:
        atom = next(iter(functions))
        if atom.func in (sp.exp, sp.sin, sp.sinh, sp.tan):
            inner = atom.args[0]
            if (
                inner.has(*variables)
                and _regular_composition(inner, variables, target) == 0
            ):
                numerator = atom - 1 if atom.func is sp.exp else atom
                try:
                    cofactor = sp.cancel(expr * inner / numerator)
                except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError):
                    cofactor = None
                if cofactor is not None:
                    candidates.append((inner, cofactor, sp.S.One, None))
    denominators = tuple(
        power.base for power in expr.atoms(sp.Pow) if power.exp.is_negative is True
    )
    t = sp.Dummy("removable_ray_parameter", positive=True)
    paths = [
        tuple(t if i == k else sp.S.Zero for i in range(len(variables)))
        for k in range(len(variables))
    ]
    paths.append(tuple(t for _ in variables))
    for inner, cofactor, extension, trig_denominator in candidates:
        if (
            not inner.has(*variables)
            or _regular_composition(inner, variables, target) != 0
        ):
            continue
        coefficient = _regular_composition(cofactor, variables, target)
        if coefficient is None or coefficient.is_finite is not True:
            continue
        guards = (
            inner,
            *(base for base in denominators if base != trig_denominator),
        )
        chosen = next(
            (
                path
                for path in paths
                if all(
                    _nonzero_germ(
                        base.subs(
                            dict(zip(variables, path, strict=True)), simultaneous=True
                        ),
                        t,
                    )
                    for base in guards
                )
            ),
            None,
        )
        if chosen is None:
            continue
        index = sp.Dummy("attained_removable_index", positive=True, integer=True)
        sequence = tuple(
            (v, p.subs(t, 1 / index)) for v, p in zip(variables, chosen, strict=True)
        )
        value = extension * coefficient
        evidence = LimitEvidence(
            "holomorphic_trigonometric_quotient"
            if trig_denominator is not None
            else "holomorphic_removable_germ",
            "The inner germ tends to zero by continuity. The exact elementary quotient "
            f"has a holomorphic removable extension with value {extension}; the cofactor "
            f"tends to {coefficient}. An attained t=1/j ray avoids the original denominator "
            "zeros. For the sine-tangent identity, a nonzero small inner germ avoids "
            "the isolated zero of 1-cos(h), and cos(h) tends to one, excluding tangent poles.",
            sequence,
            value,
        )
        return LimitStatus.PROVED, value, evidence
    return None
