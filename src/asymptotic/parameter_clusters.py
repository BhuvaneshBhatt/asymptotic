"""Certified projective cluster geometry on symbolic parameter strata."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask
from .coverage import CoverageCertificate
from .multivariate_limit_models import AdvancedLimitStatus, ClusterSetResult


@dataclass(frozen=True)
class ParameterClusterStratum:
    """A certified cluster set valid throughout one parameter regime."""

    condition: sp.Expr
    cluster_result: ClusterSetResult


def _relation_to_threshold(condition, exponent, threshold):
    """Return ``-1, 0, +1`` when a condition fixes exponent vs threshold."""
    condition = sp.sympify(condition)
    pairs = (
        (sp.StrictLessThan, -1),
        (sp.StrictGreaterThan, 1),
        (sp.Equality, 0),
    )
    for cls, relation in pairs:
        if isinstance(condition, cls):
            if (
                sp.simplify(condition.lhs - exponent) == 0
                and sp.simplify(condition.rhs - threshold) == 0
            ):
                return relation
            if (
                sp.simplify(condition.rhs - exponent) == 0
                and sp.simplify(condition.lhs - threshold) == 0
            ):
                return -relation
    # Stratification canonicalization commonly rewrites Eq(p, a) as Eq(p-a, 0).
    if isinstance(condition, sp.Equality):
        delta = sp.simplify(condition.lhs - condition.rhs)
        if (
            sp.simplify(delta - (exponent - threshold)) == 0
            or sp.simplify(delta + (exponent - threshold)) == 0
        ):
            return 0
    return None


def _homogeneous_degree(poly, variables):
    try:
        p = sp.Poly(poly, *variables)
    except sp.PolynomialError:
        return None
    degrees = {sum(mon) for mon, coeff in p.terms() if coeff != 0}
    return next(iter(degrees)) if len(degrees) == 1 else None


def _radial_factor(denominator, r2, variables):
    """Split ``denominator`` into a radial symbolic power and angular factor."""
    factors = denominator.args if denominator.is_Mul else (denominator,)
    radial = None
    rest = []
    for factor in factors:
        if (
            factor.is_Pow
            and sp.expand(factor.base - r2) == 0
            and factor.exp.free_symbols - set(variables)
        ):
            if radial is not None:
                return None
            radial = factor.exp
            continue
        rest.append(factor)
    if radial is None and denominator.is_Pow and sp.expand(denominator.base - r2) == 0:
        radial = denominator.exp
        rest = []
    if radial is None:
        return None
    return radial, sp.Mul(*rest)


def _diagonal_quadratic_range(expr, variables):
    """Exact sphere image for a diagonal quadratic form."""
    expanded = sp.expand(expr)
    coeffs = []
    reconstructed = 0
    for variable in variables:
        coeff = sp.simplify(expanded.coeff(variable, 2))
        if coeff.free_symbols & set(variables):
            return None
        coeffs.append(coeff)
        reconstructed += coeff * variable**2
    if sp.simplify(expanded - reconstructed) != 0:
        return None
    return sp.Min(*coeffs), sp.Max(*coeffs), "diagonal_quadratic_simplex"


def _diagonal_quadratic_coefficients(expr, variables):
    expanded = sp.expand(expr)
    coeffs = []
    reconstructed = 0
    for variable in variables:
        coeff = sp.simplify(expanded.coeff(variable, 2))
        if coeff.free_symbols & set(variables):
            return None
        coeffs.append(coeff)
        reconstructed += coeff * variable**2
    if sp.simplify(expanded - reconstructed) != 0:
        return None
    return tuple(coeffs)


def _combine_condition(base, extra):
    return sp.simplify_logic(sp.And(sp.sympify(base), sp.sympify(extra)))


def _quadratic_parameter_strata(
    expr, variables, target, condition, numerator, relation
):
    """Refine a two-axis diagonal quadratic by its symbolic coefficient order/sign."""
    coeffs = _diagonal_quadratic_coefficients(numerator, variables)
    if coeffs is None or len(coeffs) != 2:
        return ()
    a, b = coeffs
    delta = sp.simplify(a - b)
    free = delta.free_symbols - set(variables)
    if relation == 0 and free:
        out = []
        for extra, lo, hi in (
            (sp.Lt(a, b), a, b),
            (sp.Eq(a, b), b, b),
            (sp.Gt(a, b), b, a),
        ):
            combined = _combine_condition(condition, extra)
            cluster_set = (
                sp.FiniteSet(lo) if sp.simplify(lo - hi) == 0 else sp.Interval(lo, hi)
            )
            out.append(
                ParameterClusterStratum(
                    combined,
                    _cluster(
                        expr,
                        variables,
                        target,
                        cluster_set,
                        lo,
                        hi,
                        "parameter_diagonal_quadratic_cluster",
                        "complete diagonal-quadratic angular image on a refined parameter stratum",
                    ),
                )
            )
        return tuple(out)

    if relation > 0 and len(free) == 1 and b.is_positive is True:
        # Relative to a fixed positive axis coefficient, the sign of the other
        # coefficient completely determines the supercritical projective image.
        out = []
        for extra, cluster_set, lo, hi in (
            (sp.Lt(a, 0), sp.S.Reals, -sp.oo, sp.oo),
            (sp.Eq(a, 0), sp.Interval(0, sp.oo), 0, sp.oo),
            (sp.Gt(a, 0), sp.FiniteSet(sp.oo), sp.oo, sp.oo),
        ):
            out.append(
                ParameterClusterStratum(
                    _combine_condition(condition, extra),
                    _cluster(
                        expr,
                        variables,
                        target,
                        cluster_set,
                        lo,
                        hi,
                        "parameter_signed_quadratic_cluster",
                        "complete supercritical projective image after coefficient-sign stratification",
                    ),
                )
            )
        return tuple(out)
    return ()


def _even_monomial_sign_range(poly, variables):
    """Certify sign endpoints from same-sign even monomials or pure-axis terms."""
    try:
        p = sp.Poly(poly, *variables)
    except sp.PolynomialError:
        return None
    terms = [(mon, sp.sympify(coeff)) for mon, coeff in p.terms() if coeff != 0]
    if not terms or any(any(power % 2 for power in mon) for mon, _ in terms):
        return None
    signs = [bounded_ask(sp.Q.positive(c)) for _, c in terms]
    negs = [bounded_ask(sp.Q.negative(c)) for _, c in terms]
    zero_direction = any(
        all(mon[i] > 0 for mon, _ in terms) for i in range(len(variables))
    )
    if all(s is True for s in signs):
        return (0 if zero_direction else 1), 1, "positive_even_monomials"
    if all(s is True for s in negs):
        return -1, (0 if zero_direction else -1), "negative_even_monomials"

    # Opposite-signed pure-axis coefficients give explicit positive and negative
    # projective directions. Continuity on the sphere supplies a zero direction.
    axis_signs = []
    for i in range(len(variables)):
        for mon, coeff in terms:
            if mon[i] > 0 and sum(mon) == mon[i]:
                axis_signs.append(sp.sign(coeff))
    if 1 in axis_signs and -1 in axis_signs:
        return -1, 1, "signed_axis_directions"
    return None


def _bounded_angular_range(numerator, angular_denominator, variables):
    """Return a certified continuous angular interval when cheaply available."""
    if angular_denominator != 1:
        return None
    from .angular_optimization import parameterized_angular_range

    optimized = parameterized_angular_range(numerator, variables)
    if optimized.certified:
        return optimized.minimum, optimized.maximum, optimized.provider
    sign_range = _even_monomial_sign_range(numerator, variables)
    if sign_range is not None:
        lo_sign, hi_sign, provider = sign_range
        # Only signs, not finite extrema, are certified for higher degree forms.
        return None, None, provider, lo_sign, hi_sign
    return None


def _projective_reciprocal_difference(numerator, denominator, variables):
    """Recognize c/(u_i^2-u_j^2), whose image is disconnected."""
    if len(variables) != 2 or numerator.free_symbols & set(variables):
        return None
    x, y = variables
    scale = sp.simplify(denominator.coeff(x, 2))
    if scale == 0 or sp.simplify(denominator - scale * (x**2 - y**2)) != 0:
        return None
    c = sp.simplify(numerator / scale)
    if c.is_real is not True or c == 0:
        return None
    a = sp.Abs(c)
    return sp.Union(sp.Interval(-sp.oo, -a), sp.Interval(a, sp.oo)), -sp.oo, sp.oo


def _cluster(expr, variables, target, cluster_set, lo, hi, provider, statement):
    return ClusterSetResult(
        expr,
        variables,
        target,
        cluster_set,
        lo,
        hi,
        AdvancedLimitStatus.CERTIFIED,
        provider,
        statement,
        coverage=CoverageCertificate.complete(
            f"{provider}_cover",
            "the certified parameter stratum and its angular/projective geometry exhaust the stated local family",
            (provider,),
        ),
    )


def parameter_cluster_strata(expr, variables, target, condition, *, domain=True):
    """Certify projective cluster sets uniformly on a parameter stratum.

    Supported exact families include continuous homogeneous angular images,
    signed supercritical radial powers, diagonal quadratic forms with symbolic
    angular extrema, and the disconnected projective reciprocal-difference
    image.  Unsupported geometry remains unclassified rather than sampled.
    """
    if sp.sympify(domain) is not sp.S.true:
        return ()
    variables = tuple(variables)
    target = tuple(map(sp.sympify, target))
    if any(a != 0 for a in target):
        return ()
    expr = sp.sympify(expr)
    numerator, denominator = sp.fraction(sp.together(expr))
    denominator = sp.factor_terms(denominator)
    r2 = sp.Add(*(v**2 for v in variables))
    split = _radial_factor(denominator, r2, variables)
    if split is None:
        return ()
    exponent, angular_denominator = split
    ndegree = _homogeneous_degree(numerator, variables)
    ddegree = _homogeneous_degree(angular_denominator, variables)
    if ndegree is None or ddegree is None:
        return ()
    threshold = sp.Rational(ndegree - ddegree, 2)
    relation = _relation_to_threshold(condition, exponent, threshold)
    if relation is None:
        return ()

    if angular_denominator == 1:
        refined = _quadratic_parameter_strata(
            expr, variables, target, condition, numerator, relation
        )
        if refined:
            return refined
    if relation == 0:
        if angular_denominator == 1:
            from .angular_optimization import parameterized_angular_strata

            angular_strata = parameterized_angular_strata(
                numerator, variables, condition
            )
            if len(angular_strata) > 1:
                return tuple(
                    ParameterClusterStratum(
                        angular_stratum.condition,
                        _cluster(
                            expr,
                            variables,
                            target,
                            angular_stratum.cluster_set,
                            angular_stratum.minimum,
                            angular_stratum.maximum,
                            f"parameter_{angular_stratum.provider}_cluster",
                            "complete angular image on an exact algebraic parameter-discriminant stratum",
                        ),
                    )
                    for angular_stratum in angular_strata
                    if angular_stratum.condition is not sp.S.false
                )
        if angular_denominator != 1:
            from .projective_clusters import parameterized_projective_cluster_strata

            projective_strata = parameterized_projective_cluster_strata(
                numerator, angular_denominator, variables, condition
            )
            if projective_strata:
                return tuple(
                    ParameterClusterStratum(
                        stratum.condition,
                        _cluster(
                            expr,
                            variables,
                            target,
                            stratum.decomposition.cluster_set,
                            stratum.decomposition.cluster_set.inf,
                            stratum.decomposition.cluster_set.sup,
                            stratum.decomposition.provider,
                            stratum.decomposition.statement,
                        ),
                    )
                    for stratum in projective_strata
                )

        disconnected = _projective_reciprocal_difference(
            numerator, angular_denominator, variables
        )
        if disconnected is not None:
            cluster_set, lo, hi = disconnected
            result = _cluster(
                expr,
                variables,
                target,
                cluster_set,
                lo,
                hi,
                "parameter_projective_disconnected_cluster",
                "complete disconnected projective angular image on the critical radial stratum",
            )
            return (ParameterClusterStratum(sp.sympify(condition), result),)

        angular_range = _bounded_angular_range(
            numerator, angular_denominator, variables
        )
        if angular_range is None:
            return ()
        if len(angular_range) == 3:
            lo, hi, provider = angular_range
            cluster_set = sp.Interval(lo, hi)
            result = _cluster(
                expr,
                variables,
                target,
                cluster_set,
                lo,
                hi,
                f"parameter_{provider}_cluster",
                "complete parameter-dependent angular image on the critical radial stratum",
            )
            return (ParameterClusterStratum(sp.sympify(condition), result),)
        return ()

    if relation < 0:
        # Positive radial order collapses every bounded continuous angular image.
        if angular_denominator != 1:
            return ()
        result = _cluster(
            expr,
            variables,
            target,
            sp.FiniteSet(0),
            sp.S.Zero,
            sp.S.Zero,
            "parameter_positive_radial_order_cluster",
            "bounded homogeneous angular factor times positive radial order has singleton cluster set {0}",
        )
        return (ParameterClusterStratum(sp.sympify(condition), result),)

    # Negative radial order: angular signs determine the complete extended image.
    if angular_denominator != 1:
        return ()
    sign_range = _even_monomial_sign_range(numerator, variables)
    if sign_range is None:
        from .angular_optimization import parameterized_angular_range

        optimized = parameterized_angular_range(numerator, variables)
        if not optimized.certified:
            return ()
        lo, hi, provider = optimized.minimum, optimized.maximum, optimized.provider
        lo_neg = bounded_ask(sp.Q.negative(lo))
        hi_pos = bounded_ask(sp.Q.positive(hi))
        lo_zero = sp.simplify(lo) == 0
        hi_zero = sp.simplify(hi) == 0
    else:
        lo, hi, provider = sign_range
        lo_neg, hi_pos = lo == -1, hi == 1
        lo_zero, hi_zero = lo == 0, hi == 0

    if lo_neg is True and hi_pos is True:
        cluster_set, infimum, supremum = sp.S.Reals, -sp.oo, sp.oo
    elif (lo_zero or lo_neg is False) and hi_pos is True:
        cluster_set, infimum, supremum = sp.Interval(0, sp.oo), 0, sp.oo
    elif lo_neg is True and (hi_zero or hi_pos is False):
        cluster_set, infimum, supremum = sp.Interval(-sp.oo, 0), -sp.oo, 0
    else:
        return ()
    result = _cluster(
        expr,
        variables,
        target,
        cluster_set,
        infimum,
        supremum,
        "parameter_signed_projective_cluster",
        f"complete signed radial-rescaling cluster image from {provider}",
    )
    return (ParameterClusterStratum(sp.sympify(condition), result),)


__all__ = ["ParameterClusterStratum", "parameter_cluster_strata"]
