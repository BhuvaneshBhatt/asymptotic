"""Uniform real norm estimates for polynomial, logarithmic and flat germs."""

import sympy as sp

from ._limit_composition import _regular_composition
from ._polynomial_bounds import bounded_degree, bounded_expansion_width
from .limit_models import LimitEvidence, LimitStatus
from .local_path_witnesses import _nonzero_germ


def _polynomial(expression, variables):
    if bounded_degree(expression, variables, 64) is None:
        return None
    try:
        polynomial = sp.Poly(expression, *variables, domain=sp.QQ)
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None
    return polynomial if len(polynomial.terms()) <= 32 else None


def _nonnegative(expression):
    if expression.is_nonnegative is True:
        return True
    if expression.is_Mul:
        return all(_nonnegative(a) for a in expression.args)
    if expression.is_Pow:
        if expression.exp.is_even is True and expression.base.is_real is True:
            return True
        return expression.exp.is_real is True and _nonnegative(expression.base)
    return False


def _positive_unit_monomial(expression, variables):
    """Keep even monomials whose remaining factors are positive local units."""
    from .multivariate_pole_bounds import _locally_real, _positive_monomial

    direct = _positive_monomial(expression, variables)
    if direct is not None:
        return direct
    coefficient = sp.S.One
    powers = [sp.S.Zero] * len(variables)
    for factor in sp.Mul.make_args(expression):
        monomial = _positive_monomial(factor, variables)
        if monomial is not None:
            coefficient *= monomial[0]
            powers = [a + b for a, b in zip(powers, monomial[1], strict=True)]
        elif not (
            _locally_real(factor, variables, (0,) * len(variables))
            and _positive_order(factor, variables) == 0
        ):
            return None
    return coefficient, tuple(powers)


def _positive_order(expression, variables):
    """Find d with a positive lower bound c*r**d near the origin.

    A positive polynomial containing a pure even power of every coordinate
    dominates a power of the Euclidean norm. Mixed degrees use the largest
    pure-power degree on the unit ball. Products multiply these bounds.
    """
    if expression.is_number and expression.is_positive is True:
        return sp.S.Zero
    constant, remainder = expression.as_coeff_Add()
    if constant.is_positive is True and remainder != 0:
        from .multivariate_pole_bounds import _locally_real

        oscillatory_size = sp.S.Zero
        for term in sp.Add.make_args(remainder):
            coefficient, head = term.as_coeff_Mul()
            if head.func not in (sp.sin, sp.cos) or not _locally_real(
                head.args[0], variables, (0,) * len(variables)
            ):
                break
            oscillatory_size += sp.Abs(coefficient)
        else:
            if (constant - oscillatory_size).is_positive is True:
                return sp.S.Zero
    if expression.func is sp.Abs:
        return _positive_order(expression.args[0], variables)
    if expression.func is sp.sign:
        inner = _positive_order(expression.args[0], variables)
        return sp.S.Zero if inner is not None else None
    if expression.func is sp.Max and set(expression.args) == {
        sp.Abs(v) for v in variables
    }:
        return sp.S.One
    if expression.is_Mul:
        orders = [_positive_order(a, variables) for a in expression.args]
        return None if None in orders else sp.Add(*orders)
    if (
        expression.is_Pow
        and expression.exp.is_positive is True
        and expression.exp.is_real is True
    ):
        order = _positive_order(expression.base, variables)
        return None if order is None else order * expression.exp
    constant, remainder = expression.as_coeff_Add()
    if constant.is_positive is True and _nonnegative(remainder):
        return sp.S.Zero
    if constant.is_positive is True:
        estimate = _upper_order(remainder, variables)
        if estimate is not None and (estimate[2] or estimate[0].is_positive is True):
            return sp.S.Zero
    if expression.is_Add:
        parts = [(a, _positive_unit_monomial(a, variables)) for a in expression.args]
        terms = [term for _, term in parts if term is not None]
        degrees = []
        for i in range(len(variables)):
            pure = [
                powers[i]
                for _, powers in terms
                if powers[i] > 0 and all(k == 0 for j, k in enumerate(powers) if j != i)
            ]
            if not pure:
                break
            degrees.append(min(pure))
        if len(degrees) == len(variables):
            degree = max(degrees)
            remainders = [
                _upper_order(a, variables) for a, term in parts if term is None
            ]
            if all(
                estimate is not None
                and (estimate[2] or (estimate[0] - degree).is_positive is True)
                for estimate in remainders
            ):
                return degree
    polynomial = _polynomial(expression, variables)
    if polynomial is None or polynomial.is_zero:
        return None
    if any(c <= 0 or any(k % 2 for k in powers) for powers, c in polynomial.terms()):
        return None
    if polynomial.TC() > 0:
        return sp.S.Zero
    degrees = []
    for i in range(len(variables)):
        pure = [
            powers[i]
            for powers, _ in polynomial.terms()
            if powers[i] > 0 and all(k == 0 for j, k in enumerate(powers) if j != i)
        ]
        if not pure:
            return None
        degrees.append(min(pure))
    return sp.Integer(max(degrees))


def _upper_order(expression, variables):
    """Return a bound O(r**d*(1+abs(log(r)))**k), or a flat bound.

    The Boolean component records decay faster than every norm power. It
    survives multiplication by polynomial poles and fixed logarithmic powers,
    but not addition to a non-flat term.
    """
    from .discontinuous_functions import SignedFractionalPart

    if expression == 0:
        return sp.S.Zero, 0, True
    if expression.func is SignedFractionalPart and expression.args[0].is_real is True:
        return sp.S.Zero, 0, False
    if expression.is_number and expression.is_finite is True:
        return sp.S.Zero, 0, False
    if expression.is_Mul:
        factors = expression.as_powers_dict()
        for factor, exponent in factors.items():
            if exponent == 1 and factor.func in (
                sp.sin,
                sp.tan,
                sp.sinh,
                sp.atan,
                sp.erf,
            ):
                argument = factor.args[0]
                if (
                    factors.get(argument) == -1
                    and _regular_composition(argument, variables, (0,) * len(variables))
                    == 0
                ):
                    # These quotients extend continuously at zero. The original
                    # denominator remains a guard when constructing a domain ray.
                    cofactor = expression / factor * argument
                    estimate = _upper_order(cofactor, variables)
                    if estimate is not None:
                        return estimate
        estimates = [_upper_order(a, variables) for a in expression.args]
        if any(a is None for a in estimates):
            return None
        return (
            sp.Add(*(a[0] for a in estimates)),
            sum(a[1] for a in estimates),
            any(a[2] for a in estimates),
        )
    if expression.is_Add:
        constant, rest = expression.as_coeff_Add()
        if constant == -1 and rest.func in (sp.exp, sp.cos):
            inner = rest.args[0]
            if _regular_composition(inner, variables, (0,) * len(variables)) == 0:
                estimate = _upper_order(inner, variables)
                if estimate is not None:
                    multiplier = 2 if rest.func is sp.cos else 1
                    return (
                        estimate[0] * multiplier,
                        estimate[1] * multiplier,
                        estimate[2],
                    )
        estimates = [_upper_order(a, variables) for a in expression.args]
        if any(a is None for a in estimates):
            return None
        nonflat = [a for a in estimates if not a[2]]
        if not nonflat:
            return sp.S.Zero, 0, True
        return min(a[0] for a in nonflat), max(a[1] for a in nonflat), False
    if expression.func in (sp.Abs, sp.sign):
        if expression.func is sp.sign:
            return sp.S.Zero, 0, False
        return _upper_order(expression.args[0], variables)
    if (
        expression.is_Pow
        and expression.exp.is_real is True
        and expression.exp.is_finite is True
    ):
        if expression.exp.is_negative is True:
            if expression.base.func is sp.log and expression.exp.is_Integer is True:
                argument = expression.base.args[0]
                estimate = _upper_order(argument, variables)
                if (
                    argument.is_nonnegative is True
                    and estimate is not None
                    and estimate[0].is_positive is True
                    and not estimate[1]
                    and _regular_composition(argument, variables, (0,) * len(variables))
                    == 0
                ):
                    return sp.S.Zero, expression.exp, False
            if expression.base.is_Add and expression.exp.is_Integer is True:
                parts = list(expression.base.args)
                for term in parts:
                    coefficient, head = term.as_coeff_Mul()
                    if (
                        head.func is not sp.log
                        or coefficient.is_real is not True
                        or coefficient.is_zero is not False
                    ):
                        continue
                    argument = head.args[0]
                    argument_order = _upper_order(argument, variables)
                    other_orders = [
                        _upper_order(other, variables)
                        for other in parts
                        if other != term
                    ]
                    if (
                        _positive_order(argument, variables) is not None
                        and argument_order is not None
                        and argument_order[0].is_positive is True
                        and not argument_order[1]
                        and all(
                            order is not None
                            and (order[2] or (order[0] >= 0 and order[1] == 0))
                            for order in other_orders
                        )
                    ):
                        # A logarithm tending to -infinity dominates every
                        # bounded real perturbation, uniformly on the defined germ.
                        return sp.S.Zero, expression.exp, False
            lower = _positive_order(expression.base, variables)
            return None if lower is None else (lower * expression.exp, 0, False)
        if expression.exp.is_positive is True:
            estimate = _upper_order(expression.base, variables)
            if estimate is None or (
                estimate[1] and expression.exp.is_Integer is not True
            ):
                return None
            return (
                estimate[0] * expression.exp,
                estimate[1] * expression.exp,
                estimate[2],
            )
    if expression.func in (sp.sin, sp.sinh, sp.tan, sp.atan, sp.erf):
        inner = expression.args[0]
        if _regular_composition(inner, variables, (0,) * len(variables)) == 0:
            return _upper_order(inner, variables)
    if expression.func in (sp.sin, sp.cos, sp.atan, sp.tanh, sp.erf):
        from .multivariate_pole_bounds import _locally_real

        if _locally_real(expression.args[0], variables, (0,) * len(variables)):
            return sp.S.Zero, 0, False
    if (
        expression.func in (sp.asin, sp.acos)
        and bounded_expansion_width(expression.args[0], 64) <= 64
    ):
        numerator, denominator = expression.args[0].as_numer_denom()
        if (
            _positive_order(denominator, variables) is not None
            and _nonnegative(sp.expand(denominator + numerator))
            and _nonnegative(sp.expand(denominator - numerator))
        ):
            return sp.S.Zero, 0, False
    if expression.func is sp.log:
        argument = expression.args[0]
        if _regular_composition(argument, variables, (0,) * len(variables)) == 1:
            return _upper_order(argument - 1, variables)
        lower = _positive_order(argument, variables)
        polynomial = _polynomial(argument, variables)
        if lower is not None and polynomial is not None and polynomial.TC() == 0:
            return sp.S.Zero, 1, False
    if expression.func is sp.exp:
        argument = expression.args[0]
        numerator, denominator = argument.as_numer_denom()
        coefficient, magnitude = numerator.as_coeff_Mul()
        if (
            coefficient.is_negative is True
            and magnitude.func is sp.Abs
            and bounded_degree(denominator, variables, 8) is not None
            and bounded_expansion_width(denominator, 32) <= 32
            and sp.expand(denominator - magnitude.args[0] ** 2) == 0
        ):
            magnitude_order = _upper_order(magnitude, variables)
            if (
                magnitude_order is not None
                and magnitude_order[0].is_positive is True
                and not magnitude_order[1]
            ):
                return sp.S.Zero, 0, True
        from .multivariate_pole_bounds import _locally_real

        if (
            numerator.is_number
            and numerator.is_negative is True
            and numerator.is_finite is True
            and _nonnegative(denominator)
            and (denominator_order := _upper_order(denominator, variables)) is not None
            and denominator_order[0].is_positive is True
            and not denominator_order[1]
            and _regular_composition(denominator, variables, (0,) * len(variables)) == 0
        ):
            return sp.S.Zero, 0, True
        coefficient, factors = argument.as_coeff_Mul()
        if (
            coefficient < 0
            and factors.is_Pow
            and factors.exp.is_negative is True
            and factors.exp.is_finite is True
        ):
            base = factors.base
            lower = _positive_order(base, variables)
            upper = _upper_order(base, variables)
            if (
                (lower is not None or _nonnegative(base))
                and upper is not None
                and upper[0].is_positive is True
                and not upper[1]
                and _regular_composition(base, variables, (0,) * len(variables)) == 0
            ):
                return sp.S.Zero, 0, True
        if (
            denominator.is_nonnegative is True
            and _locally_real(numerator, variables, (0,) * len(variables))
            and (
                numerator.is_nonpositive is True
                or _positive_order(-numerator, variables) == 0
            )
        ):
            return sp.S.Zero, 0, False
        center = _regular_composition(argument, variables, (0,) * len(variables))
        if center is not None and center.is_finite is True:
            return sp.S.Zero, 0, False
    polynomial = _polynomial(expression, variables)
    if polynomial is not None:
        if polynomial.is_zero:
            return sp.S.One, 0, False
        return (
            sp.Integer(min(sum(powers) for powers, _ in polynomial.terms())),
            0,
            False,
        )
    return None


def radial_limit_certificate(expr, variables, target, domain, assumptions):
    """Prove a finite limit from a uniform Euclidean norm estimate.

    Positive power excess dominates every fixed logarithmic factor. Flat
    exponentials dominate all admitted polynomial poles. An attained positive
    ray separately establishes accumulation and avoids original denominator
    holes; the estimate itself applies to the entire defined real germ.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or any(
            p.is_real is not True or p.is_finite is not True or p.free_symbols
            for p in target
        )
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or sp.count_ops(expr) > 80
        or expr.free_symbols - set(variables)
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    exact = expr.xreplace({f: sp.Rational(f) for f in expr.atoms(sp.Float)})
    center = tuple(
        p.xreplace({f: sp.Rational(f) for f in p.atoms(sp.Float)}) for p in target
    )
    exact = exact.subs(
        dict(
            zip(
                variables,
                (v + p for v, p in zip(variables, center, strict=True)),
                strict=True,
            )
        ),
        simultaneous=True,
    )
    value = sp.S.Zero
    remainder = exact
    if exact.is_Add:
        continuous = []
        singular = []
        for term in exact.args:
            center_value = _regular_composition(term, variables, (0,) * len(variables))
            if center_value is not None and center_value.is_finite is True:
                value += center_value
                continuous.append(term - center_value)
            else:
                singular.append(term)
        remainder = sp.Add(*continuous, *singular)
    if exact.is_Add:
        estimates = [_upper_order(term, variables) for term in continuous + singular]
        if any(estimate is None for estimate in estimates):
            return None
        nonflat = [estimate for estimate in estimates if not estimate[2]]
        estimate = (
            (min(e[0] for e in nonflat), max(e[1] for e in nonflat), False)
            if nonflat
            else (sp.S.Zero, 0, True)
        )
    else:
        estimate = _upper_order(remainder, variables)
    if estimate is None or (
        not estimate[2]
        and estimate[0].is_positive is not True
        and not (estimate[0] == 0 and estimate[1] < 0)
    ):
        return None
    guards = [p.base for p in exact.atoms(sp.Pow) if p.exp.is_negative is True]
    guards.extend(a.args[0] for a in exact.atoms(sp.log))
    t = sp.Dummy("radial_parameter", positive=True)
    chosen = None
    for multipliers in ((1,) * len(variables), tuple(range(1, len(variables) + 1))):
        substitution = dict(zip(variables, (c * t for c in multipliers), strict=True))
        along = [g.subs(substitution, simultaneous=True) for g in guards]
        if all(
            g.is_positive is True
            or g.is_negative is True
            or _positive_order(original, variables) is not None
            or (
                original.func is sp.log
                and original.args[0].is_nonnegative is True
                and _regular_composition(
                    original.args[0], variables, (0,) * len(variables)
                )
                == 0
            )
            or (
                original.has(sp.log)
                and (inverse_order := _upper_order(1 / original, variables)) is not None
                and inverse_order[0] == 0
                and inverse_order[1] < 0
            )
            or _nonzero_germ(g, t)
            for original, g in zip(guards, along, strict=True)
        ):
            chosen = multipliers
            break
    if chosen is None:
        return None
    index = sp.Dummy("radial_index", positive=True, integer=True)
    sequence = tuple(
        (v, p + sp.Integer(c) / index)
        for v, p, c in zip(variables, center, chosen, strict=True)
    )
    statement = (
        "A positive radial rate gives decay faster than every norm power."
        if estimate[2]
        else f"The absolute difference from the returned limit is O(r**({estimate[0]})*(1+abs(log(r)))**({estimate[1]})); positive power excess or a negative pure logarithmic exponent makes this bound vanish."
    )
    evidence = LimitEvidence(
        "uniform_radial_bound",
        statement
        + " Positive polynomial pure powers provide coercive norm bounds on the unit ball. The attained positive ray avoids every original denominator and logarithm zero.",
        sequence,
        value,
    )
    return LimitStatus.PROVED, value, evidence


def signed_power_vanishing_certificate(expr, variables, target, domain, assumptions):
    """Remove unit-modulus sign factors on the original punctured domain.

    For real u approaching zero and 0<=p<1, u/abs(u)**p has
    magnitude abs(u)**(1-p). Integer powers of sign(u) have magnitude one
    wherever defined. A continuous cofactor preserves the zero limit.
    """
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 40
        or any(p.exp.is_Integer and abs(p.exp) > 16 for p in expr.atoms(sp.Pow))
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None
    exponents = {sp.S.Zero}
    for power in expr.atoms(sp.Pow):
        if (
            power.base.has(sp.Abs)
            and power.exp.is_Rational is True
            and -1 < power.exp < 0
        ):
            exponents.add(-power.exp)
    factors = expr.as_powers_dict()
    for sign in sorted(expr.atoms(sp.sign), key=sp.default_sort_key):
        inner = sign.args[0]
        k = factors.get(sign)
        if (
            k is None
            or k.is_Integer is not True
            or inner.is_real is not True
            or _regular_composition(inner, variables, target) != 0
        ):
            continue
        for power in sorted(exponents):
            cofactor = sp.cancel(expr * sign ** (-k) * sp.Abs(inner) ** power / inner)
            coefficient = _regular_composition(cofactor, variables, target)
            if coefficient is None or coefficient.is_finite is not True:
                continue
            guards = [inner] + [
                p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True
            ]
            t = sp.Dummy("signed_power_parameter", positive=True)
            for ray in ((1,) * len(variables), tuple(range(1, len(variables) + 1))):
                paths = tuple(p + c * t for p, c in zip(target, ray, strict=True))
                substitution = dict(zip(variables, paths, strict=True))
                if not all(
                    _nonzero_germ(g.subs(substitution, simultaneous=True), t)
                    for g in guards
                ):
                    continue
                j = sp.Dummy("signed_power_index", positive=True, integer=True)
                sequence = tuple(
                    (v, p + sp.Integer(c) / j)
                    for v, p, c in zip(variables, target, ray, strict=True)
                )
                evidence = LimitEvidence(
                    "signed_power_bound",
                    f"On the original defined real germ, integer sign powers have unit magnitude. The absolute quotient is abs(u)**({1 - power}) times a continuous cofactor, hence tends uniformly to zero. An attained ray avoids u=0 and every original denominator.",
                    sequence,
                    sp.S.Zero,
                )
                return LimitStatus.PROVED, sp.S.Zero, evidence
    return None


def radial_pole_certificate(expr, variables, target, domain, assumptions):
    """Prove a signed polynomial pole dominates lower-order radial terms."""
    if (
        not isinstance(expr, sp.Expr)
        or len(variables) not in (2, 3)
        or tuple(target) != (0,) * len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.free_symbols - set(variables)
        or expr.has(sp.Float)
        or sp.count_ops(expr) > 60
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
    ):
        return None

    def expansion_width(expression):
        if expression.is_Add:
            return min(17, sum(expansion_width(a) for a in expression.args))
        if expression.is_Mul:
            width = 1
            for part in expression.args:
                width *= expansion_width(part)
                if width > 16:
                    return 17
            return width
        return 1

    if expansion_width(expr) > 16:
        return None
    expanded = sp.expand_mul(expr, deep=False)
    terms = sp.Add.make_args(expanded)
    if len(terms) > 12:
        return None
    for term in terms:
        numerator, denominator = sp.fraction(term)
        lower = _positive_order(denominator, variables)
        upper = _upper_order(denominator, variables)
        center = _regular_composition(numerator, variables, target)
        if (
            lower is None
            or upper is None
            or upper[1]
            or upper[2]
            or upper[0].is_positive is not True
            or center is None
            or center.is_real is not True
            or center.is_finite is not True
            or (center.is_positive is not True and center.is_negative is not True)
        ):
            continue
        remainders = [
            _upper_order(other, variables) for other in terms if other != term
        ]
        if not all(
            a is not None and (a[2] or (a[0] + upper[0]).is_positive is True)
            for a in remainders
        ):
            continue
        guards = [p.base for p in expr.atoms(sp.Pow) if p.exp.is_negative is True]
        guards.extend(a.args[0] for a in expr.atoms(sp.log))
        t = sp.Dummy("radial_pole_parameter", positive=True)
        for ray in ((1,) * len(variables), tuple(range(1, len(variables) + 1))):
            along = dict(zip(variables, (c * t for c in ray), strict=True))
            if not all(
                _nonzero_germ(g.subs(along, simultaneous=True), t) for g in guards
            ):
                continue
            j = sp.Dummy("radial_pole_index", positive=True, integer=True)
            sequence = tuple(
                (v, sp.Integer(c) / j) for v, c in zip(variables, ray, strict=True)
            )
            value = sp.oo if center.is_positive is True else -sp.oo
            evidence = LimitEvidence(
                "dominant_radial_pole",
                f"The locally positive denominator is O(r**({upper[0]})) and the numerator has a nonzero signed real center. Every other term has a strictly smaller radial growth order, including its logarithmic powers. The attained ray avoids original denominator and logarithm zeros.",
                sequence,
                value,
            )
            return LimitStatus.PROVED, value, evidence
    return None


def radial_vanishing_certificate(expr, variables, target, domain, assumptions):
    """Return the radial certificate only when the proved limit is zero."""
    result = radial_limit_certificate(expr, variables, target, domain, assumptions)
    return result if result is not None and result[1] == 0 else None
