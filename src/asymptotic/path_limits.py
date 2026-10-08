"""One-sided and contour-path limits built on the canonical limit engine."""

from __future__ import annotations

from dataclasses import replace

import sympy as sp
from funcprops import ParametricContour
from funcprops.exactness import certify_identity_equal

from .limit_models import (
    LimitEvidence,
    LimitStatus,
    SimultaneousLimitDoesNotExist,
    SimultaneousLimitResult,
)
from .limits import limit


def _value_or_result(result: SimultaneousLimitResult, return_result: bool):
    from .stratification import AsymptoticStratification

    if isinstance(result, AsymptoticStratification):
        return result if return_result else result.mathematical_value
    if return_result:
        return result
    if result.status is LimitStatus.PROVED:
        return result.value
    if result.status is LimitStatus.DOES_NOT_EXIST:
        raise SimultaneousLimitDoesNotExist(
            "certified path witnesses prove nonexistence"
        )
    return result


def _pullback_result(e, x, p, u, chart, result, method, statement, domain=sp.S.true):
    """Keep public metadata and attained witnesses in the caller's variable."""
    from .stratification import AsymptoticStratification

    if isinstance(result, AsymptoticStratification):
        return replace(
            result,
            strata=tuple(
                replace(
                    cell,
                    result=_pullback_result(
                        e, x, p, u, chart, cell.result, method, statement, domain
                    ),
                )
                for cell in result.strata
            ),
        )
    items = tuple(
        replace(
            ev,
            substitutions=tuple(
                (x, chart.subs(u, value)) if variable == u else (variable, value)
                for variable, value in ev.substitutions
            ),
        )
        for ev in result.evidence
    )
    return replace(
        result,
        expression=e,
        variables=(x,),
        target=(p,),
        reduced_variables=(x,),
        domain=domain,
        evidence=(LimitEvidence(method, statement, value=e.subs(x, chart)),) + items,
    )


def _finite_under_assumptions(value, assumptions):
    if value.is_finite is True:
        return True
    clauses = sp.And.make_args(sp.sympify(assumptions))
    if sp.Q.finite(value) in clauses:
        return True
    replacements = {}
    for clause in sp.And.make_args(sp.sympify(assumptions)):
        if isinstance(clause, (sp.StrictGreaterThan, sp.StrictLessThan)):
            subject = clause.lhs - clause.rhs
            if isinstance(clause, sp.StrictLessThan):
                subject = -subject
            replacements[subject] = sp.Dummy("positive_fixed_parameter", positive=True)
        elif isinstance(clause, sp.Unequality) and clause.rhs == 0:
            replacements[clause.lhs] = sp.Dummy(
                "nonzero_fixed_parameter", zero=False, finite=clause.lhs.is_finite
            )
        elif (
            isinstance(clause, sp.Equality)
            and isinstance(clause.lhs, sp.Symbol)
            and clause.rhs.is_finite is True
        ):
            replacements[clause.lhs] = clause.rhs
        elif (
            clause.func is sp.assumptions.assume.AppliedPredicate
            and clause.function == sp.Q.finite
        ):
            replacements[clause.arguments[0]] = sp.Dummy(
                "finite_fixed_parameter", finite=True
            )
    return value.xreplace(replacements).is_finite is True


def _validate_finite_ray(x, p, d, assumptions=sp.S.true):
    if not isinstance(x, sp.Symbol):
        raise TypeError("limit variable must be a symbol")
    if p.has(x) or not _finite_under_assumptions(p, assumptions):
        raise ValueError("ray limit requires a fixed finite point")
    if d.has(x) or d.is_finite is not True or d.is_zero is not False:
        raise ValueError("ray must be fixed, finite and proved nonzero")
    if x.is_real is True and (p.is_real is not True or d.is_real is not True):
        raise ValueError("ray is incompatible with a real variable")
    if x.is_imaginary is True and (
        (p.is_imaginary is not True and p.is_zero is not True)
        or d.is_imaginary is not True
    ):
        raise ValueError("ray is incompatible with an imaginary variable")
    if x.is_real is False and p.is_real is True and d.is_real is True:
        raise ValueError("ray is incompatible with a nonreal variable")
    if x.is_integer is True or x.is_finite is False:
        raise ValueError(
            "variable domain cannot approach this point on a continuous finite ray"
        )
    if (x.is_nonnegative is True and p.is_negative is True) or (
        x.is_nonpositive is True and p.is_positive is True
    ):
        raise ValueError("target is incompatible with variable sign")
    if p == 0 and (
        (x.is_nonnegative is True and d.is_negative is True)
        or (x.is_nonpositive is True and d.is_positive is True)
    ):
        raise ValueError("ray is incompatible with variable sign")


def _positive_pullback_limit(expression, u, assumptions):
    from .branch_constant_normalization import normalize_logarithmic_angles

    expression = normalize_logarithmic_angles(expression, u)
    if assumptions is sp.S.false:
        return SimultaneousLimitResult(
            expression,
            (u,),
            (sp.S.Zero,),
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "incompatible_pullback_assumptions",
                    "The approach chart does not satisfy the supplied assumptions.",
                ),
            ),
            (u,),
            sp.S.true,
        )
    from .local_inverse_poles import inverse_tangent_pole_certificate

    certificate = inverse_tangent_pole_certificate(expression, u)
    if certificate is not None and not assumptions.has(u):
        status, value, evidence = certificate
        return SimultaneousLimitResult(
            expression,
            (u,),
            (sp.S.Zero,),
            status,
            value,
            (evidence,),
            (u,),
            sp.S.true,
        )
    return limit(expression, u, 0, assumptions=assumptions, return_result=True)


def complex_ray_limit(
    expr, variable, point, *, ray, assumptions=True, return_result=False
):
    """Finite-target limit on z=point+ray*t, t->0+, for a fixed nonzero ray.

    The ray points outward from the target: ray=I approaches from above.
    This certifies one path, not a limit across all complex directions.
    """
    e = sp.sympify(expr)
    x = sp.sympify(variable)
    p = sp.sympify(point)
    d = sp.sympify(ray)
    _validate_finite_ray(x, p, d, assumptions)
    if isinstance(p, sp.Float):
        p = sp.Rational(p)
    u = sp.Dummy("complex_ray_parameter", positive=True)
    chart = p + d * u
    pulled_assumptions = sp.sympify(assumptions).subs(x, chart)
    from .complex_ray_germs import (
        elementary_ray_branch_certificate,
        ray_integral_cut_certificate,
        special_function_ray_certificate,
    )

    transformed = e.subs(x, chart)
    from .fixed_ray_branch_germs import (
        original_bessel_ray_certificate,
        rational_ray_pole_certificate,
        ray_sign_certificate,
        special_cut_ray_certificate,
    )

    certificate = original_bessel_ray_certificate(e, x, p, d)
    for provider in (
        rational_ray_pole_certificate,
        ray_sign_certificate,
        special_cut_ray_certificate,
        ray_integral_cut_certificate,
        elementary_ray_branch_certificate,
        special_function_ray_certificate,
    ):
        if certificate is None:
            certificate = provider(transformed, u)
    if certificate is None:
        from .local_branch_germs import logarithmic_integral_cancellation_certificate

        certificate = logarithmic_integral_cancellation_certificate(
            transformed, u, sp.S.Zero, sp.S.true, pulled_assumptions
        )
        if certificate is not None:
            status, value, items = certificate
            certificate = (status, value, items[0])
    from .variable_order_germs import regular_variable_bessel_certificate

    if certificate is None:
        certificate = regular_variable_bessel_certificate(
            transformed, u, sp.S.Zero, sp.S.true, pulled_assumptions
        )
    supported = {
        sp.exp,
        sp.sin,
        sp.cos,
        sp.sinh,
        sp.cosh,
        sp.erf,
        sp.erfc,
        sp.erfi,
        sp.log,
        sp.arg,
        sp.re,
        sp.im,
        sp.Abs,
        sp.atanh,
        sp.Ei,
        sp.Ci,
        sp.Chi,
        sp.Heaviside,
        sp.besselj,
        sp.bessely,
        sp.besseli,
        sp.besselk,
    }
    unsupported = sp.count_ops(transformed) > 160 or any(
        a.func not in supported for a in transformed.atoms(sp.Function)
    )
    if (
        transformed.has(sp.besselj, sp.bessely, sp.besseli, sp.besselk)
        and certificate is None
    ):
        unsupported = True
    for atom in transformed.atoms(sp.Ei, sp.Ci, sp.Chi):
        center = atom.args[0].subs(u, 0)
        if certificate is None and not (
            center.is_finite is True
            and (center.is_positive is True or center.is_real is False)
        ):
            unsupported = True

    if certificate is not None and pulled_assumptions is sp.S.true:
        status, value, evidence = certificate
        result = SimultaneousLimitResult(
            e.subs(x, chart),
            (u,),
            (sp.S.Zero,),
            status,
            value,
            (evidence,),
            (u,),
            sp.S.true,
        )
    elif unsupported:
        result = SimultaneousLimitResult(
            transformed,
            (u,),
            (sp.S.Zero,),
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "complex_ray_germ_prerequisite",
                    "This ray needs an unsupported function, sector, amplified cut germ or parameter-dependent branch theorem.",
                ),
            ),
            (u,),
            sp.S.true,
        )
    else:
        result = _positive_pullback_limit(transformed, u, pulled_assumptions)
    if (
        result.status is LimitStatus.PROVED
        and result.value.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
        and result.value not in (sp.oo, -sp.oo)
    ):
        result = replace(
            result,
            status=LimitStatus.UNKNOWN,
            value=None,
            evidence=(
                LimitEvidence(
                    "complex_ray_pole_prerequisite",
                    "A noncanonical infinite expression needs an explicit directional pole theorem.",
                ),
            ),
        )
    result = _pullback_result(
        e,
        x,
        p,
        u,
        chart,
        result,
        "complex_ray_pullback",
        "Exact fixed ray z=point+ray*t with t->0+; this is a directional result only.",
        sp.And(sp.Eq(sp.im((x - p) / d), 0), sp.Gt(sp.re((x - p) / d), 0)),
    )
    return _value_or_result(result, return_result)


def one_sided_limit(
    expr, variable, point, *, direction: str, assumptions=True, return_result=False
):
    """Compute a certified univariate limit from one real side.

    ``direction='+'`` approaches from larger real values and ``direction='-'``
    approaches from smaller real values.  The request is reduced exactly to a
    positive local parameter before invoking :func:`asymptotic.limit`.
    """
    if direction not in ("+", "-"):
        raise ValueError("direction must be '+' or '-'")
    e = sp.sympify(expr)
    x = sp.sympify(variable)
    p = sp.sympify(point)
    if isinstance(p, sp.Float):
        p = sp.Rational(p)
    if p in (sp.oo, -sp.oo):
        raise ValueError("one-sided finite-direction semantics require a finite point")
    _validate_finite_ray(x, p, sp.S.One if direction == "+" else -sp.S.One, assumptions)
    u = sp.Dummy("_limit_u", positive=True)
    transformed = e.subs(x, p + (u if direction == "+" else -u))
    pulled_assumptions = sp.sympify(assumptions).subs(
        x, p + (u if direction == "+" else -u)
    )
    result = _positive_pullback_limit(transformed, u, pulled_assumptions)
    ray_domain = sp.And(
        sp.Eq(sp.im(x - p), 0),
        sp.Gt(sp.re(x - p), 0) if direction == "+" else sp.Lt(sp.re(x - p), 0),
    )
    result = _pullback_result(
        e,
        x,
        p,
        u,
        p + (u if direction == "+" else -u),
        result,
        "one_sided_pullback",
        f"{x} -> {p} {direction} u",
        ray_domain,
    )
    return _value_or_result(result, return_result)


def path_limit(
    expr,
    variable,
    point,
    *,
    path: ParametricContour,
    assumptions=True,
    return_result=False,
):
    """Compute a certified limit along the terminal germ of a parametric contour."""
    if not isinstance(path, ParametricContour):
        raise TypeError("path must be a funcprops.ParametricContour")
    e = sp.sympify(expr)
    x = sp.sympify(variable)
    p = sp.sympify(point)
    if certify_identity_equal(path.end, p) is not True:
        raise ValueError("path endpoint must equal the requested limit point")
    from ._symbolic_policy import bounded_simplify

    if (
        path.upper.is_finite is not True
        or path.lower.is_finite is not True
        or p.is_finite is not True
    ):
        raise ValueError("terminal contour germ requires finite bounds and endpoint")
    delta = bounded_simplify(path.upper - path.lower)
    if delta.is_positive is True:
        sign = -1
    elif delta.is_negative is True:
        sign = 1
    else:
        raise ValueError("path orientation must be exactly decidable")
    u = sp.Dummy("_path_limit_u", positive=True)
    transformed = e.subs(x, path.gamma.subs(path.parameter, path.upper + sign * u))
    chart = path.gamma.subs(path.parameter, path.upper + sign * u)
    if chart.has(x) or bounded_simplify(chart - p) == 0:
        raise ValueError("path must approach through a nonconstant punctured germ")
    result = _positive_pullback_limit(
        transformed, u, sp.sympify(assumptions).subs(x, chart)
    )
    result = _pullback_result(
        e,
        x,
        p,
        u,
        chart,
        result,
        "path_pullback",
        "terminal contour germ reduced to u -> 0+",
    )
    return _value_or_result(result, return_result)


__all__ = ["complex_ray_limit", "one_sided_limit", "path_limit"]
