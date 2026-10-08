"""Whole-plane limits through a faithful two-real-coordinate chart."""

from dataclasses import replace

import sympy as sp

from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult
from .limits import limit
from .path_limits import _finite_under_assumptions, _value_or_result


def _regular_germ(expression, variable):
    if not expression.has(variable):
        return expression.is_finite is True
    if expression == variable:
        return True
    if expression.is_Add or expression.is_Mul:
        return all(_regular_germ(arg, variable) for arg in expression.args)
    if expression.is_Pow:
        return (
            expression.exp.is_Integer is True
            and 0 <= expression.exp <= 12
            and _regular_germ(expression.base, variable)
        )
    if expression.func is sp.atan:
        return expression.args[0].subs(variable, 0).is_zero is True and _regular_germ(
            expression.args[0], variable
        )
    return expression.func in (
        sp.sin,
        sp.cos,
        sp.exp,
        sp.sinh,
        sp.cosh,
    ) and _regular_germ(expression.args[0], variable)


def _meromorphic_value(expression, variable, point):
    """Use bounded Taylor valuations only for quotients of regular analytic germs."""
    if sp.count_ops(expression) > 80:
        return None
    expression = (
        expression.rewrite(sp.sin) if expression.has(sp.tan, sp.cot) else expression
    )
    numerator, denominator = sp.fraction(sp.together(expression))
    if not (
        _regular_germ(numerator, variable) and _regular_germ(denominator, variable)
    ):
        return None
    t = sp.Dummy("complex_displacement")
    coefficients = []
    for term in (numerator, denominator):
        if term == 0:
            coefficients.append((sp.oo, sp.S.Zero))
            continue
        expansion = (
            sp.series(term.subs(variable, point + t), t, 0, 9).removeO().expand()
        )
        terms = [
            (j, expansion.coeff(t, j)) for j in range(9) if expansion.coeff(t, j) != 0
        ]
        if (
            not terms
            or terms[0][1].is_zero is not False
            or terms[0][1].is_finite is not True
        ):
            return None
        coefficients.append(terms[0])
    (ni, nc), (di, dc) = coefficients
    if dc == 0:
        return None
    return sp.zoo if ni < di else sp.S.Zero if ni > di else sp.simplify(nc / dc)


def complex_limit(
    expression, variable, point=0, *, assumptions=sp.S.true, return_result=False
):
    """Take a limit over every complex approach to a finite point or infinity.

    The chart ``z = point + u + I*v`` ranges over the punctured plane. A
    spherical pole is returned as ``zoo``; it carries no preferred direction.
    Unresolved branch or domain questions remain structured unknown results.
    """
    expression, point, assumptions = map(sp.sympify, (expression, point, assumptions))
    if not isinstance(variable, sp.Symbol):
        raise TypeError("limit variable must be a symbol")
    if assumptions.has(variable):
        raise ValueError(
            "whole-plane parameter assumptions cannot restrict the approach variable"
        )
    if (
        variable.is_real is not None
        or variable.is_imaginary is not None
        or variable.is_finite is False
    ):
        raise ValueError(
            "whole-plane approach requires an unrestricted complex variable"
        )
    if point.has(variable) or (
        point is not sp.zoo and not _finite_under_assumptions(point, assumptions)
    ):
        raise ValueError(
            "whole-plane limit requires a fixed finite point or spherical infinity"
        )
    if assumptions is sp.S.false:
        result = SimultaneousLimitResult(
            expression,
            (variable,),
            (point,),
            LimitStatus.UNKNOWN,
            evidence=(
                LimitEvidence(
                    "empty_complex_approach",
                    "False assumptions admit no approaches, so no value or nonexistence is certified.",
                ),
            ),
        )
        return _value_or_result(result, return_result)
    if point is sp.zoo:
        from .complex_tail_germs import spherical_growth_certificate

        certificate = spherical_growth_certificate(expression, variable)
        if certificate is not None:
            status, value, item = certificate
            result = SimultaneousLimitResult(
                expression, (variable,), (point,), status, value, (item,)
            )
        else:
            result = SimultaneousLimitResult(
                expression,
                (variable,),
                (point,),
                LimitStatus.UNKNOWN,
                evidence=(
                    LimitEvidence(
                        "spherical_infinity_prerequisite",
                        "Uniform angular growth or decay has not been certified.",
                    ),
                ),
            )
        return _value_or_result(result, return_result)
    displacement = sp.Dummy("complex_displacement")
    centered = expression.subs(variable, point + displacement)
    value = _meromorphic_value(centered, displacement, sp.S.Zero)
    if value is not None:
        result = SimultaneousLimitResult(
            expression,
            (variable,),
            (point,),
            LimitStatus.PROVED,
            value,
            (
                LimitEvidence(
                    "whole_plane_meromorphic_valuation",
                    "A quotient of regular analytic germs has exact nonzero Taylor coefficients at the first recorded orders. Their quotient gives a removable value or an isolated pole whose modulus diverges on every complex approach.",
                    value=value,
                ),
            ),
        )
        return _value_or_result(result, return_result)
    if expression.free_symbols - {variable}:
        result = SimultaneousLimitResult(
            expression,
            (variable,),
            (point,),
            LimitStatus.UNKNOWN,
            evidence=(
                LimitEvidence(
                    "whole_plane_parameter_prerequisite",
                    "Unresolved parameter coefficients or denominators require explicit regular and singular strata before the real-coordinate chart can certify a universal value.",
                ),
            ),
        )
        return _value_or_result(result, return_result)
    unsupported_heads = expression.atoms(sp.Function) - expression.atoms(
        sp.re, sp.im, sp.Abs, sp.sign
    )
    if unsupported_heads or expression.has(sp.sign):
        from .path_limits import complex_ray_limit

        ray_axis = sp.S.One if expression.has(sp.sign) else sp.I

        upper = complex_ray_limit(
            expression,
            variable,
            point,
            ray=ray_axis,
            assumptions=assumptions,
            return_result=True,
        )
        lower = complex_ray_limit(
            expression,
            variable,
            point,
            ray=-ray_axis,
            assumptions=assumptions,
            return_result=True,
        )
        if (
            upper.status is lower.status is LimitStatus.PROVED
            and upper.value.is_finite is True
            and lower.value.is_finite is True
            and sp.simplify(upper.value - lower.value).is_zero is False
        ):
            result = SimultaneousLimitResult(
                expression,
                (variable,),
                (point,),
                LimitStatus.DOES_NOT_EXIST,
                evidence=upper.evidence
                + lower.evidence
                + tuple(
                    LimitEvidence(
                        "attained_complex_ray",
                        "The certified ray is attained at t=1/n; its branch certificate gives eventual domain and pole avoidance.",
                        (
                            (
                                variable,
                                point
                                + side
                                * ray_axis
                                / sp.Dummy("ray_index", integer=True, positive=True),
                            ),
                        ),
                        ray_result.value,
                    )
                    for side, ray_result in ((1, upper), (-1, lower))
                ),
            )
        else:
            result = SimultaneousLimitResult(
                expression,
                (variable,),
                (point,),
                LimitStatus.UNKNOWN,
                evidence=(
                    LimitEvidence(
                        "whole_plane_branch_prerequisite",
                        "A function outside the entire-germ grammar needs a whole-plane branch certificate; agreeing rays do not supply one.",
                    ),
                ),
            )
        return _value_or_result(result, return_result)
    u, v = (
        sp.Dummy("real_displacement", real=True),
        sp.Dummy("imaginary_displacement", real=True),
    )
    chart = point + u + sp.I * v
    pulled = sp.expand_complex(expression.subs(variable, chart))
    result = limit(pulled, (u, v), (0, 0), assumptions=assumptions, return_result=True)
    if not isinstance(result, SimultaneousLimitResult):
        unknown = SimultaneousLimitResult(
            expression,
            (variable,),
            (point,),
            LimitStatus.UNKNOWN,
            evidence=(
                LimitEvidence(
                    "complex_parameter_cells_prerequisite",
                    "Real-coordinate parameter cells require a whole-plane codomain certificate before projection.",
                ),
            ),
        )
        return _value_or_result(unknown, return_result)
    evidence = []
    for item in result.evidence:
        bindings = dict(item.substitutions)
        if u in bindings or v in bindings:
            for coordinate in (u, v):
                if coordinate not in pulled.free_symbols:
                    bindings.setdefault(coordinate, sp.S.Zero)
            item = replace(item, substitutions=((variable, chart.subs(bindings)),))
        evidence.append(item)
    evidence = tuple(evidence)
    if result.status is LimitStatus.DOES_NOT_EXIST:
        from .local_tail_germs import fixed_finite

        values = [
            item.value
            for item in evidence
            if item.value is not None
            and isinstance(item.value, sp.Expr)
            and item.substitutions
            and item.substitutions[0][0] == variable
            and not item.substitutions[0][1].has(u, v)
            and (
                fixed_finite(item.value)
                or _finite_under_assumptions(item.value, assumptions)
            )
        ]
        conflict = False
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                delta = sp.simplify(left - right)
                if delta.is_zero is False or sp.Ne(delta, 0) in sp.And.make_args(
                    assumptions
                ):
                    conflict = True
        if not conflict:
            result = replace(
                result, status=LimitStatus.UNKNOWN, value=None, evidence=()
            )
            evidence = (
                LimitEvidence(
                    "whole_plane_codomain_prerequisite",
                    "The available witnesses do not establish distinct finite attained values on the declared parameter cells. Differing infinite directions represent the same spherical infinity and need a uniform pole certificate.",
                ),
            )
    result = replace(
        result,
        expression=expression,
        variables=(variable,),
        target=(point,),
        reduced_variables=(variable,),
        evidence=(
            LimitEvidence(
                "whole_plane_real_chart",
                "Two independent real displacements cover every complex approach to the puncture.",
            ),
        )
        + evidence,
    )
    return _value_or_result(result, return_result)
