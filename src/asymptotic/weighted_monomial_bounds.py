"""Uniform quotient bounds from positive sums and weighted monomials."""

from itertools import combinations

import sympy as sp

from ._limit_composition import _regular_composition
from ._polynomial_bounds import bounded_degree, bounded_expansion_width
from .limit_models import LimitEvidence, LimitStatus
from .multivariate_pole_bounds import _locally_real, _positive_monomial

_MAX_TERMS = 24
_ELEMENTARY_HEADS = {
    sp.Abs,
    sp.exp,
    sp.log,
    sp.sin,
    sp.cos,
    sp.tan,
    sp.atan,
    sp.sinh,
    sp.cosh,
    sp.erf,
    sp.erfc,
    sp.sign,
}


def _multiply(left, right):
    if len(left) * len(right) > _MAX_TERMS:
        return None
    return [
        (a * b, tuple(x + y for x, y in zip(p, q, strict=True)))
        for a, p in left
        for b, q in right
    ]


def _absolute_monomial(expression, variables):
    coefficient, factors = expression.as_coeff_Mul()
    powers = [sp.S.Zero] * len(variables)
    for factor in sp.Mul.make_args(factors):
        base, power = factor.as_base_exp()
        if power.is_Rational is not True or power < 0:
            return None
        if base.func is sp.Abs:
            base = base.args[0]
        if base not in variables:
            return None
        powers[variables.index(base)] += power
    return sp.Abs(coefficient), tuple(powers)


def _log_cancellation_terms(expression, variables, depth):
    """Preserve quadratic logarithmic cancellation with a cubic error bound.

    Real polynomial bases tending to one are positive locally, so integer
    powers inside their principal logarithm can be pulled out exactly. For
    |h| <= 1/2, log(1+h) = h-h**2/2+R with |R| <= 2*|h|**3.
    Only expressions linear in at most three logarithms are considered.
    """
    atoms = tuple(sorted(expression.atoms(sp.log), key=sp.default_sort_key))
    if not 1 <= len(atoms) <= 3:
        return None
    labels = tuple(sp.Dummy("logarithm_atom") for _ in atoms)
    try:
        linear = sp.Poly(
            expression.xreplace(dict(zip(atoms, labels, strict=True))),
            *labels,
            domain=sp.EX,
        )
    except sp.PolynomialError:
        return None
    if linear.total_degree() != 1:
        return None
    prefix = linear.coeff_monomial((0,) * len(atoms))
    errors = []
    for i, atom in enumerate(atoms):
        argument = atom.args[0]
        factor = sp.S.One
        if argument.is_Pow and argument.exp.is_Integer and 1 <= argument.exp <= 4:
            factor, argument = argument.exp, argument.base
        if (
            bounded_degree(argument, variables, 6) is None
            or argument.subs(dict.fromkeys(variables, 0)) != 1
            or not _locally_real(argument, variables, (0,) * len(variables))
        ):
            return None
        inner = argument - 1
        coefficient = linear.coeff_monomial(
            tuple(int(j == i) for j in range(len(atoms)))
        )
        local_prefix = factor * (inner - inner**2 / 2)
        prefix_coefficient = coefficient
        if bounded_degree(coefficient, variables, 6) is None:
            # Preserve cancellation against a unit exponential without asking
            # for a general series: exp(g)-1 is O(g), while exp(g) stays bounded.
            if (
                coefficient.func is not sp.exp
                or bounded_degree(coefficient.args[0], variables, 6) is None
                or coefficient.args[0].subs(dict.fromkeys(variables, 0)) != 0
                or not _locally_real(
                    coefficient.args[0], variables, (0,) * len(variables)
                )
            ):
                return None
            unit_error = _terms(
                (coefficient - 1) * local_prefix,
                variables,
                lower=False,
                depth=depth + 1,
            )
            if unit_error is None:
                return None
            errors.extend(unit_error)
            prefix_coefficient = sp.S.One
        prefix += prefix_coefficient * local_prefix
        error = _terms(
            2 * sp.Abs(factor) * sp.Abs(coefficient) * inner**3,
            variables,
            lower=False,
            depth=depth + 1,
        )
        if error is None:
            return None
        errors.extend(error)
    if (
        bounded_degree(prefix, variables, 12) is None
        or bounded_expansion_width(prefix, 24) > 24
    ):
        return None
    bound = _terms(sp.expand(prefix), variables, lower=False, depth=depth + 1)
    if bound is None or len(bound) + len(errors) > _MAX_TERMS:
        return None
    return bound + errors


def _terms(expression, variables, *, lower, depth=0):
    """Bound an expression by a sum of absolute coordinate monomials.

    Positive rational powers use Jensen's inequality and subadditivity. Lower
    bounds require nonnegative factors; upper bounds use absolute values and
    the triangle inequality. Continuous units have fixed local bounds.
    """
    if depth > 6:
        return None
    zero_powers = (sp.S.Zero,) * len(variables)
    if expression == 0:
        return []
    direct = (_positive_monomial if lower else _absolute_monomial)(
        expression, variables
    )
    if direct is not None:
        return [direct]
    if lower and expression in variables and expression.is_nonnegative is True:
        # Attained univariate charts carry explicit positive parameters; their
        # odd powers are nonnegative, while plain real coordinates still need Abs.
        powers = tuple(sp.S.One if v == expression else sp.S.Zero for v in variables)
        return [(sp.S.One, powers)]
    if expression.is_number and expression.is_finite is True:
        if lower and expression.is_positive is not True:
            return None
        return [(expression if lower else sp.Abs(expression), zero_powers)]

    if expression.func is sp.sign and not lower:
        argument = expression.args[0]
        if bounded_degree(argument, variables, 6) is not None and _locally_real(
            argument, variables, (0,) * len(variables)
        ):
            return [(sp.S.One, zero_powers)]

    def descend(part):
        return _terms(part, variables, lower=lower, depth=depth + 1)

    # These local quotients extend to one at zero. A factor of two bounds
    # their modulus; on a nonnegative real phase one half is a lower bound.
    inner = None
    if expression.func in (sp.sin, sp.sinh, sp.tan, sp.atan):
        inner = expression.args[0]
    elif expression.func is sp.log:
        inner = expression.args[0] - 1
    elif expression.is_Add:
        constant, remainder = expression.as_coeff_Add()
        if constant == -1 and remainder.func is sp.exp:
            inner = remainder.args[0]
    if (
        inner is not None
        and _regular_composition(inner, variables, (0,) * len(variables)) == 0
    ):
        bound = descend(inner)
        factor = sp.S.Half if lower else sp.Integer(2)
        return None if bound is None else [(factor * c, p) for c, p in bound]
    if expression.is_Mul:
        result = [(sp.S.One, zero_powers)]
        for factor in expression.args:
            bound = descend(factor)
            if bound is None:
                return None
            result = _multiply(result, bound)
            if result is None:
                return None
        return result
    if expression.is_Pow:
        power = expression.exp
        if power.is_Rational is not True or not 0 < power <= 12:
            return None
        bound = descend(expression.base)
        if bound is None or not bound:
            return bound
        count = len(bound)
        factor = sp.Integer(count) ** (power - 1)
        factor = min(factor, sp.S.One) if lower else max(factor, sp.S.One)
        return [
            (factor * c**power, tuple(power * p for p in powers)) for c, powers in bound
        ]
    if expression.func is sp.Abs and expression.args[0].is_Mul:
        return descend(sp.Mul(*(sp.Abs(a) for a in expression.args[0].args)))
    if expression.is_Add and not lower:
        cancellation = _log_cancellation_terms(expression, variables, depth)
        if cancellation is not None:
            return cancellation
        result = []
        for part in expression.args:
            bound = descend(part)
            if bound is None:
                return None
            result.extend(bound)
            if len(result) > _MAX_TERMS:
                return None
        return result
    center = _regular_composition(expression, variables, (0,) * len(variables))
    if center is not None and center.is_finite is True:
        if lower:
            if center.is_positive is True and _locally_real(
                expression, variables, (0,) * len(variables)
            ):
                return [(center / 2, zero_powers)]
        else:
            return [(sp.Abs(center) + 1, zero_powers)]
    if expression.is_Add:
        result = []
        for part in expression.args:
            bound = descend(part)
            if bound is None:
                return None
            result.extend(bound)
            if len(result) > _MAX_TERMS:
                return None
        return result
    return None


def _excess_bound(numerator, denominator, variables):
    """Allocate numerator powers to one monomial or a weighted pair.

    Weighted AM-GM gives A+B >= A**w*B**(1-w)/(w**w*(1-w)**(1-w)).
    Coordinate inequalities restrict w to an exact rational interval. Only a
    nonnegative excess vector with at least one positive entry can certify zero.
    """
    bounds = []
    for coefficient, exponents in numerator:
        choices = list(denominator)
        for (a, p), (b, q) in combinations(denominator, 2):
            low, high = sp.S.Zero, sp.S.One
            for n, x, y in zip(exponents, p, q, strict=True):
                difference = x - y
                if difference > 0:
                    high = min(high, (n - y) / difference)
                elif difference < 0:
                    low = max(low, (n - y) / difference)
                elif n < y:
                    low, high = sp.S.One, sp.S.Zero
                    break
            if low > high:
                continue
            weight = (low + high) / 2
            if not 0 < weight < 1:
                continue
            scale = (
                a**weight
                * b ** (1 - weight)
                / (weight**weight * (1 - weight) ** (1 - weight))
            )
            powers = tuple(
                weight * x + (1 - weight) * y for x, y in zip(p, q, strict=True)
            )
            choices.append((scale, powers))
        for scale, powers in choices:
            excess = tuple(n - d for n, d in zip(exponents, powers, strict=True))
            if all(e >= 0 for e in excess) and any(e > 0 for e in excess):
                bounds.append(
                    coefficient
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
    return sp.Add(*bounds)


def _weighted_quotient_at_origin(expr, variables, target, domain, assumptions):
    """Certify uniform vanishing or positive growth from monomial estimates."""
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
    if any(atom.func not in _ELEMENTARY_HEADS for atom in expr.atoms(sp.Function)):
        return None
    numerator, denominator = sp.fraction(expr)
    if denominator == 1:
        return None
    if numerator.func is sp.exp:
        coefficient, phase_denominator = sp.fraction(numerator.args[0])
        if (
            coefficient.is_positive is True
            and coefficient.is_number is True
            and bounded_degree(phase_denominator, variables, 6) is not None
            and phase_denominator.subs(dict.fromkeys(variables, 0)) == 0
            and _terms(phase_denominator, variables, lower=True)
            and bounded_degree(denominator, variables, 6) is not None
            and denominator.subs(dict.fromkeys(variables, 0)) == 0
            and _terms(denominator, variables, lower=True)
        ):
            index = sp.Dummy("positive_exponential_index", positive=True, integer=True)
            evidence = LimitEvidence(
                "positive_exponential_pole_bound",
                "Both polynomial denominators are nonnegative locally and positive "
                "on the original defined germ. The real exponential is at least one, "
                "so the quotient is bounded below by the reciprocal of its vanishing "
                "positive denominator. The attained positive diagonal avoids both zeros.",
                tuple((v, 1 / index) for v in variables),
                sp.oo,
            )
            return LimitStatus.PROVED, sp.oo, evidence
    upper = _terms(numerator, variables, lower=False)
    if upper is None or not upper:
        return None
    lower = _terms(denominator, variables, lower=True)
    if lower is None and denominator.is_Add:
        positive, remainder = [], []
        for term in denominator.args:
            bound = _terms(term, variables, lower=True)
            if bound is None:
                remainder.append(term)
            else:
                positive.extend(bound)
        if not positive or len(positive) > _MAX_TERMS:
            return None
        error = _terms(sp.Add(*remainder), variables, lower=False)
        if error is None or _excess_bound(error, positive, variables) is None:
            return None
        # An o(positive part) perturbation can consume at most half its lower
        # bound near the origin, including at zeros of individual monomials.
        lower = [(c / 2, p) for c, p in positive]
    if lower is None or not lower:
        return None
    bound = _excess_bound(upper, lower, variables)
    if bound is None:
        return None
    index = sp.Dummy("weighted_bound_index", positive=True, integer=True)
    evidence = LimitEvidence(
        "weighted_monomial_bound",
        "Triangle, power and local unit bounds give positive monomial estimates. "
        "Weighted AM-GM allocates denominator powers to each numerator term; "
        "higher-order signed denominator perturbations are absorbed locally. "
        f"The absolute quotient is bounded by {bound}, which tends uniformly to zero. "
        "The estimates remain valid where selected monomials vanish. The attained "
        "positive diagonal keeps the original denominator positive eventually; local "
        "exponential, logarithmic and trigonometric germs remain defined there.",
        tuple((v, 1 / index) for v in variables),
        sp.S.Zero,
    )
    return LimitStatus.PROVED, sp.S.Zero, evidence


def _original_germ_attained(expr, variables, target):
    """Prove eventual pole avoidance on the original positive diagonal.

    A locally nonzero continuous base stays nonzero. A zero-centered base must
    have either a positive/negative monomial bound or a nonzero polynomial
    restriction, whose roots are isolated. Unsupported or identically zero
    restrictions decline; algebraic cancellation never removes this obligation.
    """
    poles = {a.args[0] for a in expr.atoms(sp.log)}
    poles.update(
        a.base
        for a in expr.atoms(sp.Pow)
        if not (a.exp.is_Rational is True and a.exp >= 0)
    )
    parameter = sp.Dummy("original_germ_parameter", positive=True)
    chart = dict(zip(variables, (p + parameter for p in target), strict=True))
    for base in poles:
        center = _regular_composition(base, variables, target)
        if center is None or center.is_finite is not True:
            return False
        if center.is_zero is False:
            continue
        along = base.subs(chart, simultaneous=True)
        if along == 0 or along.is_zero is True:
            return False
        if bounded_degree(along, (parameter,), 12) is not None:
            coefficients = sp.Poly(along, parameter).all_coeffs()
            if any(c.is_zero is False and c.is_finite is True for c in coefficients):
                continue
            return False
        if _terms(along, (parameter,), lower=True):
            continue
        if _terms(-along, (parameter,), lower=True):
            continue
        return False
    return True


def weighted_quotient_certificate(expr, variables, target, domain, assumptions):
    """Reuse uniform quotient estimates through bounded real local normalization.

    Finite numeric centers are translated exactly. A real polynomial with a
    nonzero center has fixed local sign, which removes its absolute value.
    Elementary compositions continuous at zero reuse a certified inner zero.
    Every retained witness is translated back into the original coordinates.
    """
    target = tuple(map(sp.sympify, target))
    if (
        not isinstance(expr, sp.Expr)
        or not 2 <= len(variables) <= 3
        or len(target) != len(variables)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or expr.has(sp.Float)
        or any(p.has(sp.Float) for p in target)
        or sp.count_ops(expr) > 60
        or expr.free_symbols - set(variables)
        or any(
            v.assumptions0 != sp.Symbol(v.name, real=True).assumptions0
            for v in variables
        )
        or any(
            p.is_number is not True or p.is_real is not True or p.is_finite is not True
            for p in target
        )
    ):
        return None
    for atom in expr.atoms(sp.Function):
        if atom.func not in _ELEMENTARY_HEADS:
            return None
        if not atom.has(*variables) and atom.func(*atom.args).is_finite is not True:
            return None
    coordinates = variables
    normalized = expr
    if any(p != 0 for p in target):
        coordinates = tuple(sp.Dummy("bound_coordinate", real=True) for _ in variables)
        normalized = expr.subs(
            dict(
                zip(
                    variables,
                    (p + u for p, u in zip(target, coordinates, strict=True)),
                    strict=True,
                )
            ),
            simultaneous=True,
        )
    zero = dict.fromkeys(coordinates, sp.S.Zero)
    replacements = {}
    for atom in normalized.atoms(sp.Abs):
        argument = atom.args[0]
        if bounded_degree(argument, coordinates, 6) is None or not _locally_real(
            argument, coordinates, tuple(zero.values())
        ):
            continue
        center = argument.subs(zero)
        if center.is_positive is True:
            replacements[atom] = argument
        elif center.is_negative is True:
            replacements[atom] = -argument
    normalized = normalized.xreplace(replacements)
    coefficient, head = normalized.as_coeff_Mul()
    composition = head.func in (sp.sin, sp.cos, sp.exp, sp.atan)
    candidate = head.args[0] if composition else normalized
    if candidate.is_Add:
        candidate = sp.together(candidate)
    numerator, denominator = sp.fraction(candidate)
    if bounded_expansion_width(numerator, _MAX_TERMS) > _MAX_TERMS:
        return None
    candidate = sp.expand_mul(numerator) / denominator
    certificate = _weighted_quotient_at_origin(
        candidate, coordinates, (0,) * len(coordinates), domain, assumptions
    )
    if certificate is None:
        return None
    status, value, evidence = certificate
    if composition:
        if value != 0 or coefficient.is_finite is not True:
            return None
        value = coefficient * head.func(sp.S.Zero)
    if not _original_germ_attained(expr, variables, target):
        return None
    substitutions = dict(evidence.substitutions)
    witness = tuple(
        (v, p + substitutions[u])
        for v, p, u in zip(variables, target, coordinates, strict=True)
    )
    statement = evidence.statement
    if normalized != expr or composition:
        statement += " Exact finite translation and fixed-sign polynomial absolute values preserve the local germ; the outer elementary head is continuous at the certified inner zero."
    return status, value, LimitEvidence(evidence.method, statement, witness, value)
