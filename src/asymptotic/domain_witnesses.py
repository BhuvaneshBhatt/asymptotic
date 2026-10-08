"""Attained approaches certifying closure of small polynomial domains."""

from itertools import product

import sympy as sp

from ._polynomial_bounds import bounded_degree


def attained_ray(domain, variables, target):
    """Return a rational ray in the punctured domain, or None if unresolved.

    Each substituted polynomial has the sign of its first nonzero coefficient
    for all sufficiently small positive t. A finite conjunction therefore has
    a shared positive interval. Failure to find a ray proves nothing about
    closure: curved approaches may still exist.
    """
    variables, target = tuple(variables), tuple(map(sp.sympify, target))
    if not 1 <= len(variables) <= 3 or len(target) != len(variables):
        return None
    if any(a.is_finite is not True or a.is_real is not True for a in target):
        return None
    if sum(sp.count_ops(a) for a in target) > 120:
        return None
    domain = sp.sympify(domain)
    # QQ conversion rationalizes floats; it is not an exact sign certificate
    # for their represented values. Leave approximate inputs to the backend.
    if domain.has(sp.Float) or any(a.has(sp.Float) for a in target):
        return None
    relations = sp.And.make_args(domain)
    if len(relations) > 8 or sp.count_ops(domain) > 120:
        return None
    t = sp.Dummy("ray_parameter", positive=True)
    polynomials = []
    for relation in relations:
        if relation is sp.S.true:
            continue
        if not isinstance(
            relation,
            (
                sp.Equality,
                sp.Unequality,
                sp.GreaterThan,
                sp.StrictGreaterThan,
                sp.LessThan,
                sp.StrictLessThan,
            ),
        ):
            return None
        expression = relation.lhs - relation.rhs
        if isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
            expression = -expression
        if bounded_degree(expression, variables, 16) is None:
            return None
        try:
            p = sp.Poly(expression, *variables, domain=sp.QQ)
        except (sp.PolynomialError, sp.CoercionFailed):
            return None
        if p.total_degree() > 16:
            return None
        polynomials.append((relation, p.as_expr()))
    for direction in product((-1, 0, 1), repeat=len(variables)):
        if not any(direction):
            continue
        substitutions = dict(
            zip(
                variables,
                (a + d * t for a, d in zip(target, direction, strict=True)),
                strict=True,
            )
        )
        for relation, expression in polynomials:
            p = sp.Poly(expression.xreplace(substitutions), t)
            if p.is_zero:
                holds = isinstance(relation, (sp.Equality, sp.GreaterThan, sp.LessThan))
            else:
                coefficient = p.terms()[-1][1]
                holds = (
                    isinstance(relation, sp.Unequality)
                    and coefficient.is_zero is False
                    or not isinstance(relation, (sp.Equality, sp.Unequality))
                    and coefficient.is_positive is True
                )
            if not holds:
                break
        else:
            return direction
    return None


def domain_accumulates(domain, variables, target):
    """Decide closure using an attained ray before general real elimination."""
    variables, target = tuple(variables), tuple(target)
    if attained_ray(domain, variables, target) is not None:
        return True
    from semialg import point_in_closure

    result = point_in_closure(
        domain,
        dict(zip(variables, target, strict=True)),
        variables,
        return_result=True,
    )
    return result.in_closure
