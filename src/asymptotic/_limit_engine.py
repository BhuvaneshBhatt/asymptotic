"""Proof-aware simultaneous real multivariate limits.

A simultaneous Euclidean limit ranges over the full admissible approach domain,
not an iterated sequence of one-variable limits. The solver combines exact
reductions, uniform bounds, path certificates, and local geometric methods;
unsupported cases return an explicit unknown status.
"""

from __future__ import annotations

import sympy as sp

from ._limit_certificates import (
    _semialgebraic_certify_candidate,
)
from ._limit_composition import (
    _complex_log_origin_conflict,
    _domain_ratio_squeeze_certificate,
    _exact_equal,
    _fast_univariate_expansion_limit,
    _growth_comparison_limit_certificate,
    _local_germ_composition_certificate,
    _monotone_infinite_outer_certificate,
    _parameter_exponential_rate_is_unresolved,
    _piecewise_branch_limit_certificate,
    _polygamma_composition_certificate,
    _principal_cut_conflict,
    _regular_composition,
    _resolved_limit_value,
    _special_univariate_limit_certificate,
    _transcendental_composition_certificate,
    _two_sided_limit,
    unresolved_origin_parameter,
)
from ._limit_paths import (
    _candidate_values,
    _expansion_path_conflict,
    _path_conflict,
)
from ._symbolic_policy import bounded_limit, bounded_refine
from .limit_models import (
    LimitEvidence,
    LimitStatus,
    SimultaneousLimitDoesNotExist,
    SimultaneousLimitResult,
)
from .limit_primitives import _normalize_variables, normalize_limit_target
from .proof_obligations import ObligationKind, ProofObligation


def _certified_germ_result(expr, original, variables, target, certificate, domain):
    status, value, evidence = certificate
    active = tuple(v for v in variables if original.has(v))
    pieces = evidence if isinstance(evidence, tuple) else (evidence,)
    if active and active != variables:
        pieces += (
            LimitEvidence(
                "variable_subset_reduction",
                "The expression is independent of passive coordinates; the unrestricted approach domain is cylindrical. The attained certificate also supplies the passive-coordinate approach.",
            ),
        )
    return SimultaneousLimitResult(
        expr, variables, target, status, value, pieces, active or variables, domain
    )


def _limit_impl(
    expr,
    variables,
    target,
    *,
    domain=True,
    assumptions=True,
    return_result: bool = False,
    _recurrence_attempted: bool = False,
):
    """Compute a simultaneous real limit at a finite Euclidean target.

    Unlike nested calls to ``sympy.limit``, all supplied variables approach the
    target together.  The implementation proves regular ambient composition,
    reduces variables absent from the expression, delegates an unrestricted
    single active variable to a two-sided exact limit, and proves nonexistence
    when two exact ambient paths have different limits.  Agreement of sampled
    paths is never treated as proof.

    For unresolved finite candidates, polynomial and rational cases use local CAD closure of the exact epsilon-bad set. Algebraic and semialgebraic function graphs are then reduced by eliminating graph auxiliaries before CAD closure. The general semialgebraic path constructs a reduced epsilon-bad set, verifies that the target lies in the closure of the
    punctured approach domain, and asks semialg quantifier elimination to prove the reduced real
    epsilon-delta sentence. Rational functions are polynomialized directly;
    graph auxiliaries are eliminated before the alternating quantifier block. Exact scalar compositions certify the multivariate inner limit and delegate the outer one-variable behavior, including removable transcendental singularities, to the bounded asymptotic limit machinery. ``domain`` therefore participates in the
    proof rather than merely filtering sampled paths.  In value mode unresolved limits return the structured
    result so uncertainty is not hidden; use ``return_result=True`` for a
    uniform evidence-bearing result.
    """
    from .instrumentation import record_symbolic_event

    record_symbolic_event("full_limit_calls")
    assumptions = sp.sympify(assumptions)
    original_expr = sp.sympify(expr)
    expr = bounded_refine(original_expr, assumptions)
    if (
        isinstance(expr, sp.Expr)
        and expr.has(sp.sin, sp.cos)
        and sp.count_ops(expr) <= 30
    ):
        expr = sp.trigsimp(expr)
    # Canonicalize rational germs before dispatch.  Cancelling a common
    # polynomial factor changes only removable holes in the punctured germ and
    # prevents expensive CAD/Lagrange work on expressions that are already
    # constant after cancellation.
    try:
        if isinstance(expr, sp.Expr) and expr.is_rational_function(
            *_normalize_variables(variables)
        ):
            expr = sp.cancel(expr)
    except (TypeError, ValueError, sp.PolynomialError):
        pass
    domain = bounded_refine(sp.sympify(domain), assumptions)
    variables = _normalize_variables(variables)
    raw_target = normalize_limit_target(variables, target)
    from .periodic_approaches import hyperbolic_pole_conflict

    periodic = hyperbolic_pole_conflict(
        original_expr, variables, raw_target, domain, assumptions
    )
    if periodic is not None:
        from .path_limits import _value_or_result

        result = _certified_germ_result(
            expr, original_expr, variables, raw_target, periodic, domain
        )
        return _value_or_result(result, return_result)
    from .uniform_radial_bounds import (
        radial_limit_certificate,
        radial_pole_certificate,
        signed_power_vanishing_certificate,
    )

    radial = radial_limit_certificate(
        original_expr, variables, raw_target, domain, assumptions
    )
    if radial is None and original_expr.has(sp.sign):
        radial = signed_power_vanishing_certificate(
            original_expr, variables, raw_target, domain, assumptions
        )
    if radial is None and original_expr.has(sp.log):
        radial = radial_pole_certificate(
            original_expr, variables, raw_target, domain, assumptions
        )
    if radial is not None:
        result = _certified_germ_result(
            expr, original_expr, variables, raw_target, radial, domain
        )
        return result if return_result else result.value
    from .trigonometric_removable_germs import trigonometric_removable_certificate

    removable = trigonometric_removable_certificate(
        original_expr, variables, raw_target, domain, assumptions
    )
    if removable is not None:
        result = _certified_germ_result(
            expr, original_expr, variables, raw_target, removable, domain
        )
        return result if return_result else result.value
    from .inverse_pole_composition import inverse_pole_certificate

    inverse = inverse_pole_certificate(
        original_expr, variables, raw_target, domain, assumptions
    )
    if inverse is not None:
        result = _certified_germ_result(
            expr, original_expr, variables, raw_target, inverse, domain
        )
        return result if return_result else result.value
    from .attained_ray_germs import (
        elementary_ray_conflict,
        exponential_linear_pole_conflict,
    )
    from .binomial_pole_witnesses import binomial_pole_conflict
    from .fractional_pole_witnesses import fractional_pole_conflict
    from .inverse_pole_composition import reciprocal_inverse_pole_certificate
    from .real_pole_germs import (
        logarithmic_unit_vanishing_certificate,
        positive_logarithmic_pole_certificate,
        signed_real_pole_certificate,
    )
    from .regular_germ_composition import (
        nonzero_sign_composition,
        rational_zero_composition,
        signed_root_composition,
    )

    for certificate_function in (
        signed_real_pole_certificate,
        positive_logarithmic_pole_certificate,
        logarithmic_unit_vanishing_certificate,
        fractional_pole_conflict,
        reciprocal_inverse_pole_certificate,
        nonzero_sign_composition,
        signed_root_composition,
        rational_zero_composition,
        exponential_linear_pole_conflict,
        binomial_pole_conflict,
        elementary_ray_conflict,
    ):
        certificate = certificate_function(
            original_expr, variables, raw_target, domain, assumptions
        )
        if certificate is not None:
            from .path_limits import _value_or_result

            result = _certified_germ_result(
                expr, original_expr, variables, raw_target, certificate, domain
            )
            return _value_or_result(result, return_result)
    from .parameter_tail_germs import binomial_joint_pole_certificate

    joint_pole = binomial_joint_pole_certificate(
        expr, variables, raw_target, domain, assumptions
    )
    if joint_pole is None:
        from .singular_curve_witnesses import signed_root_pole_witness

        joint_pole = signed_root_pole_witness(
            expr, variables, raw_target, domain, assumptions
        )
    if joint_pole is not None:
        status, value, evidence = joint_pole
        result = SimultaneousLimitResult(
            expr, variables, raw_target, status, value, evidence, variables, domain
        )
        if return_result:
            return result
        raise SimultaneousLimitDoesNotExist("attained singular-corner limits disagree")
    if len(variables) == 1 and isinstance(expr, sp.Expr):
        from .limit_certificates import scalar_limit_certificate

        x, p = variables[0], raw_target[0]
        if domain == (x > p):
            u = sp.Dummy("audit_positive_side", positive=True)
            certificate = scalar_limit_certificate(
                expr.subs(x, p + u), u, 0, sp.S.true, assumptions.subs(x, p + u)
            )
        else:
            certificate = scalar_limit_certificate(expr, x, p, domain, assumptions)
        if certificate is not None:
            status, value, evidence = certificate
            certificate_evidence = (
                evidence if isinstance(evidence, tuple) else (evidence,)
            )
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                status,
                value,
                certificate_evidence,
                variables,
                domain,
            )
            if return_result or status is LimitStatus.UNKNOWN:
                return result
            if status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(
                    "; ".join(e.statement for e in certificate_evidence)
                )
            return value
    from .local_path_witnesses import parameter_monomial_conflict

    parameter_conflict = parameter_monomial_conflict(
        original_expr, variables, raw_target, domain, assumptions
    )
    if parameter_conflict is not None:
        result = SimultaneousLimitResult(
            expr,
            variables,
            raw_target,
            LimitStatus.DOES_NOT_EXIST,
            None,
            parameter_conflict,
            variables,
            domain,
        )
        if return_result:
            return result
        raise SimultaneousLimitDoesNotExist(
            "attained parameter-family path limits disagree"
        )
    from .analytic_parameter_corners import analytic_corner_stratification
    from .parameter_conditions import (
        displaced_pole_stratification,
        rational_corner_stratification,
        scaled_corner_stratification,
    )

    for discover in (
        displaced_pole_stratification,
        scaled_corner_stratification,
        analytic_corner_stratification,
        rational_corner_stratification,
    ):
        corner_strata = discover(expr, variables, raw_target, domain, assumptions)
        if corner_strata is not None:
            return corner_strata if return_result else corner_strata.mathematical_value
    from .multivariate_pole_bounds import (
        positive_sum_vanishing_certificate,
        radical_quotient_certificate,
        regular_numerator_pole_certificate,
        removable_germ_certificate,
    )

    elementary = radical_quotient_certificate(
        original_expr, variables, raw_target, domain, assumptions
    )
    if elementary is None:
        elementary = removable_germ_certificate(
            original_expr, variables, raw_target, domain, assumptions
        )
    if elementary is None:
        elementary = positive_sum_vanishing_certificate(
            original_expr, variables, raw_target, domain, assumptions
        )
    if elementary is None:
        from .weighted_monomial_bounds import weighted_quotient_certificate

        elementary = weighted_quotient_certificate(
            original_expr, variables, raw_target, domain, assumptions
        )
    if elementary is None:
        from .trigonometric_pole_germs import trigonometric_pole_certificate

        elementary = trigonometric_pole_certificate(
            original_expr, variables, raw_target, domain, assumptions
        )
    if elementary is None:
        elementary = regular_numerator_pole_certificate(
            expr, variables, raw_target, domain, assumptions
        )
    if elementary is not None:
        status, value, evidence = elementary
        result = SimultaneousLimitResult(
            expr, variables, raw_target, status, value, (evidence,), variables, domain
        )
        return result if return_result else result.value
    from .local_path_witnesses import (
        local_path_conflict,
        polynomial_pole_conflict,
        reciprocal_phase_conflict,
    )

    local_conflict = polynomial_pole_conflict(
        original_expr, variables, raw_target, domain, assumptions
    )
    if local_conflict is None:
        local_conflict = reciprocal_phase_conflict(
            original_expr, variables, raw_target, domain, assumptions
        )
    if local_conflict is None:
        local_conflict = local_path_conflict(
            original_expr, variables, raw_target, domain, assumptions
        )
    if local_conflict is not None:
        result = SimultaneousLimitResult(
            expr,
            variables,
            raw_target,
            LimitStatus.DOES_NOT_EXIST,
            None,
            local_conflict,
            variables,
            domain,
        )
        if return_result:
            return result
        raise SimultaneousLimitDoesNotExist(
            "attained local subsequences have different limits"
        )

    from .limit_planner import plan_limit

    proof_plan = plan_limit(expr, variables, raw_target)
    if len(variables) > 1 and any(point in (sp.oo, -sp.oo) for point in raw_target):
        raise NotImplementedError(
            "simultaneous limits at infinite target points are not implemented"
        )
    # A positive limit certificate is meaningful only when the punctured
    # approach domain actually accumulates at the target.  Check this before
    # ambient/order shortcuts so restricted domains cannot inherit an ambient
    # proof for an empty local germ.  Unsupported closure reasoning declines.
    if domain is not sp.S.true:
        distance2 = sp.Add(
            *(
                (variable - point) ** 2
                for variable, point in zip(variables, raw_target, strict=True)
            )
        )
        punctured_domain = sp.And(domain, sp.Gt(distance2, 0))
        try:
            from .domain_witnesses import domain_accumulates

            closure = domain_accumulates(
                punctured_domain,
                variables,
                raw_target,
            )
        except (
            ImportError,
            ModuleNotFoundError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
        ):
            closure = None
        if closure is not None and not closure:
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.UNKNOWN,
                None,
                (
                    LimitEvidence(
                        "semialgebraic_approach_closure",
                        "target is not in the closure of the punctured approach domain",
                    ),
                ),
                variables,
                domain,
                obligations=(
                    ProofObligation(
                        ObligationKind.DOMAIN_ACCUMULATION,
                        "establish a nonempty punctured approach germ at the target",
                        provider="limit",
                        expression=str(domain),
                    ),
                ),
            )
            return result
    # A finite-dimensional vector limit exists iff every component limit exists.
    # Keep the common domain/target for every component; never infer a joint
    # cluster set as a Cartesian product from scalar cluster information.
    if isinstance(expr, (sp.Tuple, tuple)):
        component_results = tuple(
            limit(
                component,
                variables,
                raw_target,
                domain=domain,
                assumptions=assumptions,
                return_result=True,
            )
            for component in expr
        )
        if any(r.status is LimitStatus.DOES_NOT_EXIST for r in component_results):
            failed_index, failed_component = next(
                (i, r)
                for i, r in enumerate(component_results)
                if r.status is LimitStatus.DOES_NOT_EXIST
            )
            result = SimultaneousLimitResult(
                sp.Tuple(*expr),
                variables,
                raw_target,
                LimitStatus.DOES_NOT_EXIST,
                None,
                (
                    LimitEvidence(
                        "vector_component_disagreement",
                        f"component {failed_index} has a certified non-existent limit; its attained witnesses follow",
                    ),
                    *failed_component.evidence,
                ),
                variables,
                domain,
            )
        elif all(r.status is LimitStatus.PROVED for r in component_results):
            value = sp.Tuple(*(r.value for r in component_results))
            symbolic_zero_power = any(
                power.base == 0 and bool(power.exp.free_symbols)
                for component in value
                for power in sp.sympify(component).atoms(sp.Pow)
            )
            if symbolic_zero_power:
                result = SimultaneousLimitResult(
                    sp.Tuple(*expr),
                    variables,
                    raw_target,
                    LimitStatus.UNKNOWN,
                    None,
                    (),
                    variables,
                    domain,
                )
                return result
            result = SimultaneousLimitResult(
                sp.Tuple(*expr),
                variables,
                raw_target,
                LimitStatus.PROVED,
                value,
                (
                    LimitEvidence(
                        "vector_componentwise_limit",
                        "all components converge on the same approach domain",
                        value=value,
                    ),
                ),
                variables,
                domain,
            )
        else:
            result = SimultaneousLimitResult(
                sp.Tuple(*expr),
                variables,
                raw_target,
                LimitStatus.UNKNOWN,
                None,
                (),
                variables,
                domain,
            )
        if return_result:
            return result
        if result.status is LimitStatus.PROVED:
            return result.value
        if result.status is LimitStatus.DOES_NOT_EXIST:
            raise SimultaneousLimitDoesNotExist(
                "a vector component has no simultaneous limit"
            )
        return result

    # Exact branch and complete-cluster theorems precede path search and QE.
    if len(variables) >= 2 and isinstance(expr, sp.Expr):
        if domain is sp.S.true and expr.func is sp.atanh:
            from .multivariate_certificates import inverse_branch_limit_certificate

            certificate = inverse_branch_limit_certificate(expr, variables, raw_target)
            if certificate.certified:
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    raw_target,
                    LimitStatus.PROVED,
                    certificate.value,
                    (
                        LimitEvidence(
                            certificate.method,
                            certificate.statement,
                            value=certificate.value,
                        ),
                    ),
                    variables,
                    domain,
                )
                return result if return_result else result.value
        # Reciprocal trigonometric phases can make radial reconstruction much
        # more expensive than computing their complete cluster set directly.
        if expr.is_Add and expr.has(sp.sin, sp.cos):
            target_substitution = dict(zip(variables, raw_target, strict=True))
            singular_phase = False
            for atom in expr.atoms(sp.sin, sp.cos):
                _, denominator = sp.fraction(atom.args[0])
                if denominator == 1:
                    continue
                value_at_target = denominator.subs(target_substitution)
                if value_at_target == 0:
                    singular_phase = True
                    break
            if singular_phase:
                from .cluster_semantics import (
                    ClusterLimitStatus as _ClusterStatus,
                )
                from .cluster_semantics import (
                    cluster_limit_semantics,
                )
                from .multivariate_limits_advanced import oscillatory_cluster_set

                cluster = oscillatory_cluster_set(
                    expr, variables, raw_target, domain=domain
                )
                semantics = cluster_limit_semantics(cluster)
                if semantics.status is _ClusterStatus.DOES_NOT_EXIST:
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        raw_target,
                        LimitStatus.DOES_NOT_EXIST,
                        None,
                        (LimitEvidence(cluster.provider, cluster.statement),),
                        variables,
                        domain,
                    )
                    if return_result:
                        return result
                    raise SimultaneousLimitDoesNotExist(
                        "complete singular oscillatory cluster set is non-singleton"
                    )

    # Preserve exact multivariate branch/cluster certificates above, then
    # allow the same bounded recurrence fallback before generic geometry.
    if len(variables) >= 2 and not _recurrence_attempted and isinstance(expr, sp.Expr):
        from dataclasses import replace

        from .recurrence_simplification import simplify_recurrences

        simplified = simplify_recurrences(
            expr,
            assumptions=assumptions,
            variables=variables,
            target=raw_target,
            require_defined_germ=True,
            return_result=True,
        )
        if simplified.evidence:
            result = _limit_impl(
                simplified.expression,
                variables,
                raw_target,
                domain=domain,
                assumptions=assumptions,
                return_result=True,
                _recurrence_attempted=True,
            )
            result = replace(
                result, expression=expr, evidence=simplified.evidence + result.evidence
            )
            if return_result or result.status is LimitStatus.UNKNOWN:
                return result
            if result.status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(
                    "recurrence-normalized simultaneous limit has incompatible paths"
                )
            return result.value

    if len(variables) >= 2:
        from ._recurrence_families import uncertified_germ_reason

        reason = uncertified_germ_reason(expr, variables, raw_target)
        if reason is not None:
            return SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.UNKNOWN,
                None,
                (LimitEvidence("recurrence_family_germ_prerequisite", reason),),
                variables,
                domain,
            )

    if len(variables) == 1 and isinstance(expr, sp.Expr) and expr.func is sp.exp:
        from .univariate_branch_certificates import complex_exponential_certificate

        exponential = complex_exponential_certificate(
            expr, variables[0], raw_target[0], domain
        )
        if exponential is not None:
            status, value, evidence = exponential
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                status,
                value,
                (evidence,),
                variables,
                domain,
            )
            if return_result:
                return result
            if status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(evidence.statement)
            return value

    if len(variables) == 1 and domain is not sp.S.true:
        variable = variables[0]
        point = raw_target[0]
        special = _special_univariate_limit_certificate(
            expr, variable, point, domain, assumptions
        )
        if special is not None:
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.PROVED,
                special[0],
                (special[1],),
                variables,
                domain,
            )
            return result if return_result else result.value
        if _parameter_exponential_rate_is_unresolved(
            expr, variable, point, domain, assumptions
        ):
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.UNKNOWN,
                None,
                (
                    LimitEvidence(
                        "parameter_exponential_rate_unresolved",
                        "the sign of a symbolic exponential rate controls the limit",
                    ),
                ),
                variables,
                domain,
            )
            return result
        direction = None
        clauses = sp.And.make_args(domain)
        for clause in clauses:
            if (
                isinstance(clause, (sp.StrictLessThan, sp.LessThan))
                and clause.lhs == variable
                and clause.rhs == point
            ):
                direction = "-"
            elif (
                isinstance(clause, (sp.StrictGreaterThan, sp.GreaterThan))
                and clause.lhs == variable
                and clause.rhs == point
            ):
                direction = "+"
        if direction is not None:
            try:
                approach = sp.Dummy("_one_sided_t", positive=True)
                replacement = point + approach if direction == "+" else point - approach
                one_sided_expr = bounded_refine(
                    expr.subs(variable, replacement), sp.Q.positive(approach)
                )
                value = bounded_limit(
                    one_sided_expr, approach, 0, direction="+", allow_general=True
                )
            except (TypeError, ValueError, NotImplementedError, RecursionError):
                value = None
            if value is not None and _resolved_limit_value(value, variables):
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    raw_target,
                    LimitStatus.PROVED,
                    value,
                    (
                        LimitEvidence(
                            "one_sided_domain_limit",
                            f"approach domain selects the {direction} one-sided germ",
                            value=value,
                        ),
                    ),
                    variables,
                    domain,
                )
                return result if return_result else value

    if (
        domain is sp.S.true
        and getattr(expr, "func", None) is sp.tan
        and not any(
            v.is_positive is True and p == 0 for v, p in zip(variables, raw_target)
        )
    ):
        phase = expr.args[0]
        target_substitution = dict(zip(variables, raw_target, strict=True))
        phase_target = sp.simplify(phase.subs(target_substitution))
        if sp.simplify(sp.cos(phase_target)) == 0:
            transverse = any(
                sp.simplify(sp.diff(phase, variable).subs(target_substitution)) != 0
                for variable in variables
            )
            if transverse:
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    raw_target,
                    LimitStatus.DOES_NOT_EXIST,
                    None,
                    (
                        LimitEvidence(
                            "transverse_tangent_pole",
                            "the tangent phase crosses a simple pole transversely at the target",
                        ),
                    ),
                    variables,
                    domain,
                )
                if return_result:
                    return result
                raise SimultaneousLimitDoesNotExist(
                    "the tangent phase crosses a simple pole at the target"
                )

    if domain is sp.S.true and isinstance(expr, sp.Expr) and expr.is_Mul:
        target_substitution = dict(zip(variables, raw_target, strict=True))
        factor_limits = []
        separable = True
        for factor in expr.args:
            active = tuple(variable for variable in variables if factor.has(variable))
            direct = _regular_composition(factor, variables, raw_target)
            if direct is not None:
                factor_limits.append(direct)
                continue
            if len(active) != 1:
                separable = False
                break
            variable = active[0]
            try:
                value = _two_sided_limit(
                    factor, variable, target_substitution[variable]
                )
            except (TypeError, ValueError, NotImplementedError, RecursionError):
                separable = False
                break
            if not _resolved_limit_value(value, variables):
                separable = False
                break
            factor_limits.append(value)
        if separable:
            value = sp.Mul(*factor_limits)
            if _resolved_limit_value(value, variables):
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    raw_target,
                    LimitStatus.PROVED,
                    value,
                    (
                        LimitEvidence(
                            "separable_product",
                            "factor limits combine without an indeterminate product",
                            value=value,
                        ),
                    ),
                    variables,
                    domain,
                )
                return result if return_result else value

    log_conflict = _complex_log_origin_conflict(expr, variables, raw_target)
    if log_conflict is not None:
        result = SimultaneousLimitResult(
            expr,
            variables,
            raw_target,
            LimitStatus.DOES_NOT_EXIST,
            None,
            (log_conflict,),
            variables,
            domain,
        )
        if return_result:
            return result
        raise SimultaneousLimitDoesNotExist(
            "principal complex-log phase is direction-dependent"
        )

    if domain is sp.S.true and len(variables) >= 2 and isinstance(expr, sp.Expr):
        from .multivariate_certificates import discontinuity_branch_certificate

        discontinuity = discontinuity_branch_certificate(expr, variables, raw_target)
        if discontinuity.certified:
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.DOES_NOT_EXIST,
                None,
                (
                    LimitEvidence(
                        discontinuity.method,
                        discontinuity.statement,
                        value=None,
                    ),
                ),
                variables,
                domain,
            )
            if return_result:
                return result
            raise SimultaneousLimitDoesNotExist(discontinuity.statement)

    if (
        domain is sp.S.true
        and len(variables) >= 2
        and isinstance(expr, sp.Expr)
        and proof_plan.rational
    ):
        # Positive rational certificates are much cheaper than enumerating
        # negative-witness paths and, unlike path agreement, prove full local
        # coverage.  Run them first; path search is only needed if they decline.
        from ._multivariate_germ import candidate_rational_limit_certificate

        rational_certificate = candidate_rational_limit_certificate(
            expr, variables, raw_target
        )
        if rational_certificate.certified:
            is_dne = rational_certificate.method.endswith("_dne")
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.DOES_NOT_EXIST if is_dne else LimitStatus.PROVED,
                None if is_dne else rational_certificate.value,
                (
                    LimitEvidence(
                        rational_certificate.method,
                        rational_certificate.statement,
                        value=None if is_dne else rational_certificate.value,
                    ),
                ),
                variables,
                domain,
            )
            if return_result:
                return result
            if is_dne:
                raise SimultaneousLimitDoesNotExist(
                    "certified rational paths give incompatible limits"
                )
            return result.value

    if (
        domain is sp.S.true
        and 2 <= len(variables) <= 3
        and isinstance(expr, sp.Expr)
        and proof_plan.rational
    ):
        conflict = _path_conflict(expr, variables, raw_target, domain=domain)
        if conflict is not None:
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.DOES_NOT_EXIST,
                None,
                conflict,
                variables,
                domain,
            )
            if return_result:
                return result
            raise SimultaneousLimitDoesNotExist(
                "exact paths give incompatible simultaneous limits"
            )

    domain_squeeze = _domain_ratio_squeeze_certificate(
        expr, variables, raw_target, domain
    )
    if domain_squeeze is not None:
        value, evidence = domain_squeeze
        result = SimultaneousLimitResult(
            expr,
            variables,
            raw_target,
            LimitStatus.PROVED,
            value,
            (evidence,),
            variables,
            domain,
        )
        return result if return_result else value

    # Cheap certificates run before geometric reconstruction so unsupported
    # expressions do not pay for coordinate changes or quantifier elimination.
    if (
        len(variables) >= 2
        and isinstance(expr, sp.Expr)
        and all(a not in (sp.oo, -sp.oo) for a in raw_target)
    ):
        from .multivariate_limits_advanced import (
            AdvancedLimitStatus as _AdvancedStatus,
        )
        from .multivariate_limits_advanced import (
            cheap_uniform_vanishing_limit,
        )

        early = cheap_uniform_vanishing_limit(
            expr, variables, raw_target, domain=domain
        )
        if early.status is not _AdvancedStatus.CERTIFIED:
            from .multivariate_limits_advanced import (
                analytic_jet_radial_vanishing_limit,
            )

            early = analytic_jet_radial_vanishing_limit(
                expr, variables, raw_target, domain=domain
            )
        if early.status is _AdvancedStatus.CERTIFIED:
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.PROVED,
                early.value,
                (LimitEvidence(early.provider, early.statement, value=early.value),),
                variables,
                domain,
            )
            return result if return_result else result.value

    radial_reduction = _exact_radial_reduction(expr, variables, raw_target, domain)
    if radial_reduction is not None:
        radial, radius_squared = radial_reduction
        value = bounded_limit(
            radial, radius_squared, 0, direction="+", allow_general=True
        )
        if _resolved_limit_value(value, variables):
            result = SimultaneousLimitResult(
                expr,
                variables,
                raw_target,
                LimitStatus.PROVED,
                value,
                (
                    LimitEvidence(
                        "radial_reduction",
                        "the expression and approach domain depend only on squared distance "
                        "from the target",
                    ),
                ),
                variables,
                domain,
            )
            return result if return_result else result.value

    # Infinite real coordinates are first normalized into reciprocal
    # projective charts.  This reuses the finite-target proof engine while
    # preserving the one-sided end (+oo versus -oo) in the transformed domain.
    raw_target = target
    try:
        target_items = (
            tuple(raw_target)
            if isinstance(raw_target, (tuple, list, sp.Tuple))
            else (raw_target,)
        )
    except TypeError:
        target_items = (raw_target,)
    if any(sp.sympify(a) in (sp.oo, -sp.oo) for a in target_items):
        from .multivariate_limits_advanced import projective_limit_chart

        chart = projective_limit_chart(expr, variables, raw_target, domain=domain)
        result = limit(
            chart.expression,
            chart.variables,
            chart.target,
            domain=chart.domain,
            assumptions=assumptions,
            return_result=True,
        )
        return (
            result
            if return_result
            else (result.value if result.status is LimitStatus.PROVED else result)
        )
    target = normalize_limit_target(variables, target)

    # The uniform-zero prepass already ran before exact radial reduction.
    # Do not repeat the same potentially declining certificate here.

    from .parameter_conditions import conditional_limit_stratification

    conditioned = conditional_limit_stratification(
        expr, variables, target, domain=domain
    )
    if conditioned is not None:
        if return_result:
            # Structured mode preserves the parameter stratification itself;
            # value mode below is the universal ConditionalExpression/Piecewise
            # projection shared by every parameter-aware proof mechanism.
            return conditioned
        return conditioned.mathematical_value

    if sp.sympify(assumptions) is sp.S.true:
        prerequisite = unresolved_origin_parameter(expr, variables, raw_target)
        if prerequisite is not None:
            return SimultaneousLimitResult(
                sp.sympify(expr),
                variables,
                raw_target,
                LimitStatus.UNKNOWN,
                None,
                (
                    LimitEvidence(
                        prerequisite,
                        "Free origin parameters require a proved parameter stratification before generic path comparison.",
                    ),
                ),
                variables,
                sp.sympify(domain),
            )

    # Relative domains and Piecewise branch geometry can accumulate through
    # coordinates absent from the branch value itself.  Dropping such a
    # coordinate changes the punctured germ (for example a value supported on
    # the line y=0 still accumulates through x).  Preserve the full ambient
    # coordinates whenever local-domain geometry participates.
    # A variable may be omitted only when it is absent from both the value
    # expression and the local-domain germ.  This is the conservative
    # cylindrical-lift condition: passive coordinates then contribute a full
    # punctured cylinder and cannot restrict the active approach set.
    piecewise_geometry = isinstance(expr, sp.Piecewise) or expr.has(
        sp.sign, sp.Heaviside, sp.floor, sp.ceiling, sp.Mod
    )
    active_pairs = tuple(
        (v, a)
        for v, a in zip(variables, target, strict=True)
        if piecewise_geometry
        or expr.has(v)
        or (domain is not sp.S.true and domain.has(v))
    )
    active_variables = tuple(v for v, _ in active_pairs)
    active_target = tuple(a for _, a in active_pairs)
    reduction = ()
    if len(active_variables) != len(variables):
        reduction = (
            LimitEvidence(
                "variable_subset_reduction",
                "removed limit variables absent from the expression",
            ),
        )

    early_piecewise = (
        _piecewise_branch_limit_certificate(
            expr, active_variables, active_target, domain, assumptions
        )
        if active_variables and isinstance(expr, sp.Piecewise)
        else None
    )

    if early_piecewise is not None:
        _pw_status, _pw_value, _pw_evidence = early_piecewise
        result = SimultaneousLimitResult(
            expr,
            variables,
            target,
            LimitStatus.DOES_NOT_EXIST if _pw_status == "dne" else LimitStatus.PROVED,
            None if _pw_status == "dne" else _pw_value,
            reduction + _pw_evidence,
            active_variables,
            domain,
        )
    elif not active_variables:
        result = SimultaneousLimitResult(
            expr,
            variables,
            target,
            LimitStatus.PROVED,
            expr,
            reduction,
            active_variables,
            domain,
        )
    else:
        branch_conflict = _principal_cut_conflict(expr, active_variables, active_target)
        if branch_conflict is not None and domain is sp.S.true:
            result = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.DOES_NOT_EXIST,
                None,
                reduction + branch_conflict,
                active_variables,
                domain,
            )
            if return_result:
                return result
            raise SimultaneousLimitDoesNotExist(
                "conflicting principal-branch side germs prove nonexistence"
            )
        germ_composed = _local_germ_composition_certificate(
            expr, active_variables, active_target, domain, assumptions
        )
        polygamma_composed = _polygamma_composition_certificate(
            expr, active_variables, active_target, domain, assumptions
        )
        special_univariate = (
            _special_univariate_limit_certificate(
                expr, active_variables[0], active_target[0], domain, assumptions
            )
            if len(active_variables) == 1
            else None
        )
        unresolved_rate = (
            _parameter_exponential_rate_is_unresolved(
                expr, active_variables[0], active_target[0], domain, assumptions
            )
            if len(active_variables) == 1
            else False
        )
        composed = (
            None
            if unresolved_rate
            else _regular_composition(expr, active_variables, active_target)
        )
        if germ_composed is not None:
            result = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.PROVED,
                germ_composed[0],
                reduction + (germ_composed[1],),
                active_variables,
                domain,
            )
        elif polygamma_composed is not None:
            result = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.PROVED,
                polygamma_composed[0],
                reduction + (polygamma_composed[1],),
                active_variables,
                domain,
            )
        elif special_univariate is not None:
            result = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.PROVED,
                special_univariate[0],
                reduction + (special_univariate[1],),
                active_variables,
                domain,
            )
        elif unresolved_rate:
            result = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.UNKNOWN,
                None,
                reduction
                + (
                    LimitEvidence(
                        "parameter_exponential_rate_unresolved",
                        "the sign of a symbolic exponential rate controls the limit",
                    ),
                ),
                active_variables,
                domain,
            )
        elif composed is not None and domain is sp.S.true:
            result = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.PROVED,
                composed,
                reduction
                + (
                    LimitEvidence(
                        "composition",
                        "continuous composition at target",
                        value=composed,
                    ),
                ),
                active_variables,
                domain,
            )
        elif len(active_variables) == 1 and (
            domain is sp.S.true or domain == (active_variables[0] > active_target[0])
        ):
            # Complete one-dimensional oscillation lifts unchanged through a
            # genuine cylindrical germ.  Two distinct accumulating values of
            # sin/cos with a phase tending to infinity certify DNE; this is
            # before generic scalar Limit, which often declines.
            from .tail_cancellation_germs import monomial_phase_subsequences

            attained = monomial_phase_subsequences(
                expr, active_variables[0], active_target[0], domain
            )
            if attained is not None:
                status, value, witnesses = attained
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    status,
                    value,
                    reduction + witnesses,
                    active_variables,
                    domain,
                )
                if return_result:
                    return result
                raise SimultaneousLimitDoesNotExist(
                    "attained cylindrical subsequences have different limits"
                )
            from .multivariate_limits_advanced import oscillatory_phase_cluster_theorem

            _cyl_cluster = oscillatory_phase_cluster_theorem(
                expr, active_variables, active_target, domain=domain
            )
            if _cyl_cluster is not None and _cyl_cluster.certified:
                from .cluster_semantics import (
                    ClusterLimitStatus,
                    cluster_limit_semantics,
                )

                _cyl_sem = cluster_limit_semantics(_cyl_cluster)
                if _cyl_sem.status is ClusterLimitStatus.DOES_NOT_EXIST:
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        target,
                        LimitStatus.DOES_NOT_EXIST,
                        None,
                        reduction
                        + (
                            LimitEvidence(
                                "cylindrical_univariate_cluster",
                                "complete active-coordinate cluster set lifts through passive cylindrical coordinates",
                            ),
                        ),
                        active_variables,
                        domain,
                    )
                    if return_result:
                        return result
                    raise SimultaneousLimitDoesNotExist(
                        "complete cylindrical univariate cluster set is non-singleton"
                    )
            expansion_value = (
                _fast_univariate_expansion_limit(
                    expr, active_variables[0], active_target[0]
                )
                if domain is sp.S.true and sp.count_ops(expr) <= 100
                else None
            )
            growth = _growth_comparison_limit_certificate(
                expr, active_variables[0], active_target[0], assumptions, domain
            )
            # SymPy can return AccumBounds here.  A degenerate bound is a
            # certified scalar limit; a nondegenerate one is a complete
            # univariate cluster witness and therefore certifies DNE.
            value = (
                expansion_value
                if expansion_value is not None
                else growth[0]
                if growth is not None
                else (
                    _two_sided_limit(expr, active_variables[0], active_target[0])
                    if domain is sp.S.true
                    else bounded_limit(
                        expr,
                        active_variables[0],
                        active_target[0],
                        direction="+",
                        allow_general=True,
                    )
                )
            )
            if isinstance(value, sp.AccumBounds):
                # The bounded symbolic policy over-approximates
                # some finite singular compositions (e.g. atan(1/|x|)).  Ask
                # SymPy's exact one-sided scalar engine before interpreting a
                # non-singleton accumulation interval.
                try:
                    _r = bounded_limit(
                        expr,
                        active_variables[0],
                        active_target[0],
                        direction="+",
                        allow_general=True,
                    )
                    _l = bounded_limit(
                        expr,
                        active_variables[0],
                        active_target[0],
                        direction="-",
                        allow_general=True,
                    )
                except (TypeError, ValueError, NotImplementedError, RecursionError):
                    _r = _l = None
                if (
                    _r is not None
                    and _l is not None
                    and not isinstance(_r, (sp.Limit, sp.AccumBounds))
                    and not isinstance(_l, (sp.Limit, sp.AccumBounds))
                    and _exact_equal(_r, _l) is True
                ):
                    value = _r
                elif _exact_equal(value.min, value.max) is True:
                    value = value.min
                else:
                    # An accumulation enclosure from termwise interval algebra
                    # need not be the attained cluster set. Correlated phases
                    # can cancel, so an enclosure alone never certifies DNE.
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        target,
                        LimitStatus.UNKNOWN,
                        None,
                        reduction
                        + (
                            LimitEvidence(
                                "accumulation_enclosure_only",
                                "a non-singleton symbolic enclosure requires attained subsequence witnesses",
                            ),
                        ),
                        active_variables,
                        domain,
                    )
                    return result
            if value is not None and _resolved_limit_value(value, active_variables):
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.PROVED,
                    value,
                    reduction
                    + (
                        (growth[1],)
                        if growth is not None
                        else (
                            LimitEvidence(
                                "univariate_reduction",
                                "two-sided univariate limit",
                                value=value,
                            ),
                        )
                    ),
                    active_variables,
                    domain,
                )
            else:
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.UNKNOWN,
                    None,
                    reduction,
                    active_variables,
                    domain,
                )
        else:
            from .multivariate_limits_advanced import (
                AdvancedLimitStatus as _EarlyAdvancedStatus,
            )
            from .multivariate_limits_advanced import (
                cheap_uniform_vanishing_limit as _cheap_uniform_zero,
            )
            from .multivariate_limits_advanced import (
                coercive_outer_divergence_limit as _cheap_outer_divergence,
            )

            # The same uniform-zero theorem already ran on the full germ.  Run
            # it again only when cylindrical reduction changed the active germ.
            _early_zero = (
                _cheap_uniform_zero(
                    expr, active_variables, active_target, domain=domain
                )
                if active_variables != variables
                else None
            )
            _early_div = _cheap_outer_divergence(
                expr, active_variables, active_target, domain=domain
            )
            if (
                _early_zero is not None
                and _early_zero.status is _EarlyAdvancedStatus.CERTIFIED
            ) or _early_div.status is _EarlyAdvancedStatus.CERTIFIED:
                _ec = (
                    _early_zero
                    if _early_zero is not None
                    and _early_zero.status is _EarlyAdvancedStatus.CERTIFIED
                    else _early_div
                )
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.PROVED,
                    _ec.value,
                    reduction
                    + (LimitEvidence(_ec.provider, _ec.statement, value=_ec.value),),
                    active_variables,
                    domain,
                )
                return result if return_result else result.value
            # Selected theorem-level outer certificates must precede generic
            # parity/path machinery: they are cheaper and can also prevent
            # unreliable generalized path limits from being used as evidence.
            _early_outer_inf = _monotone_infinite_outer_certificate(
                expr, active_variables, active_target, domain
            )
            if _early_outer_inf is not None:
                _oi_value, _oi_evidence = _early_outer_inf
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.PROVED,
                    _oi_value,
                    reduction + (_oi_evidence,),
                    active_variables,
                    domain,
                )
                return result if return_result else result.value
            # Branch/singular certificates whose local theorems are stronger
            # than generalized path series must run first.  This prevents
            # spurious negative-ray Bessel-Y expansions and avoids expensive
            # path work for an already certified variable-power branch conflict.
            _early_branch_cert = None
            if expr.has(sp.bessely) or any(
                isinstance(node, sp.Pow) and node.exp.has(*active_variables)
                for node in sp.preorder_traversal(expr)
            ):
                from .multivariate_certificates import fast_multivariate_certificate

                _early_branch_cert = fast_multivariate_certificate(
                    expr, active_variables, active_target, domain=domain
                )
            if (
                _early_branch_cert is not None
                and _early_branch_cert.certified
                and _early_branch_cert.method
                in {
                    "singular_polylog_germ",
                    "special_function_germ",
                    "variable_power_branch_dne",
                }
            ):
                _early_dne = _early_branch_cert.method.endswith("_dne")
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.DOES_NOT_EXIST if _early_dne else LimitStatus.PROVED,
                    None if _early_dne else _early_branch_cert.value,
                    reduction
                    + (
                        LimitEvidence(
                            _early_branch_cert.method,
                            _early_branch_cert.statement,
                            value=None if _early_dne else _early_branch_cert.value,
                        ),
                    ),
                    active_variables,
                    domain,
                )
                if return_result:
                    return result
                if _early_dne:
                    raise SimultaneousLimitDoesNotExist(_early_branch_cert.statement)
                return result.value

            # Run inexpensive proof-producing parity/growth certificates before
            # path analysis: transcendental path simplification can be much more
            # expensive than a direct radial growth proof.
            from .multivariate_certificates import fast_multivariate_certificate

            prepiecewise = _piecewise_branch_limit_certificate(
                expr, active_variables, active_target, domain
            )
            prefast = (
                fast_multivariate_certificate(
                    expr, active_variables, active_target, domain=domain
                )
                if prepiecewise is None and not isinstance(expr, sp.Piecewise)
                else None
            )
            if prefast is None and prepiecewise is not None:
                _pw_status, _pw_value, _pw_evidence = prepiecewise
                prefast = type(
                    "_PiecewiseCertificate",
                    (),
                    {
                        "method": "piecewise_accumulation_dne"
                        if _pw_status == "dne"
                        else "piecewise_accumulation",
                        "statement": (
                            "conflicting accumulating Piecewise branch germs"
                            if _pw_status == "dne"
                            else "all accumulating Piecewise branch germs have the same certified limit"
                        ),
                        "value": _pw_value,
                    },
                )()
            # Removable scalar compositions (sinc-like germs, expm1/log1p
            # quotients, etc.) are proof-producing and should run before path
            # analysis.  Delaying them made simple removable germs pay for
            # expensive path/CAD work first.
            precomposition = (
                _transcendental_composition_certificate(
                    expr, active_variables, active_target, domain
                )
                if prefast is None
                else None
            )
            if prefast is None and precomposition is not None:
                _pc_value, _pc_evidence = precomposition
                prefast = type(
                    "_CompositionCertificate",
                    (),
                    {
                        "method": "removable_transcendental_composition",
                        "statement": "certified inner germ and removable/continuous outer scalar limit",
                        "value": _pc_value,
                    },
                )()
            # Singular primitive combinations may cancel only after local
            # expansion. Conflicting certified path germs remain a cheap
            # negative witness after stronger branch/germ theorems decline.
            expansion_probe = False
            if expr.has(sp.Function) and sp.count_ops(expr) <= 80:
                try:
                    target_value = expr.subs(
                        dict(zip(active_variables, active_target, strict=True)),
                        simultaneous=True,
                    )
                    expansion_probe = bool(
                        target_value.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
                    )
                except (TypeError, ValueError, NotImplementedError, RecursionError):
                    expansion_probe = True
            if prefast is None and expansion_probe:
                expansion_conflict = _expansion_path_conflict(
                    expr, active_variables, active_target, domain
                )
                if expansion_conflict is not None:
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        target,
                        LimitStatus.DOES_NOT_EXIST,
                        None,
                        reduction + expansion_conflict,
                        active_variables,
                        domain,
                    )
                    if return_result:
                        return result
                    raise SimultaneousLimitDoesNotExist(
                        "certified local expansions disagree along exact paths"
                    )

            # Uniform radial-order and monotone outer-divergence theorems are
            # stronger than heuristic candidate generation and may correct a
            # candidate produced by a less uniform reduction.
            from .multivariate_limits_advanced import (
                AdvancedLimitStatus as _StrongAdvancedStatus,
            )
            from .multivariate_limits_advanced import (
                coercive_outer_divergence_limit as _strong_outer_divergence,
            )
            from .multivariate_limits_advanced import (
                radial_order_vanishing_limit as _strong_radial_zero,
            )

            if prefast is None:
                _srz = _strong_radial_zero(
                    expr, active_variables, active_target, domain=domain
                )
                _sod = _strong_outer_divergence(
                    expr, active_variables, active_target, domain=domain
                )
                _strong = (
                    _srz if _srz.status is _StrongAdvancedStatus.CERTIFIED else _sod
                )
                if _strong.status is _StrongAdvancedStatus.CERTIFIED:
                    prefast = type(
                        "_UniformTheoremCertificate",
                        (),
                        {
                            "method": _strong.provider,
                            "statement": _strong.statement,
                            "value": _strong.value,
                        },
                    )()
            if (
                prefast is not None
                and not prefast.method.endswith("_dne")
                and not _resolved_limit_value(prefast.value, active_variables)
            ):
                prefast = None
            if prefast is None:
                from .multivariate_limits_advanced import (
                    AdvancedLimitStatus as _AdvancedLimitStatus,
                )
                from .multivariate_limits_advanced import (
                    bounded_factor_envelope_limit as _bounded_factor_envelope_limit,
                )

                _bounded = _bounded_factor_envelope_limit(
                    expr, active_variables, active_target, domain=domain
                )
                if _bounded.status is _AdvancedLimitStatus.CERTIFIED:
                    prefast = type(
                        "_BoundedEnvelopeCertificate",
                        (),
                        {
                            "method": "bounded_factor_envelope",
                            "statement": _bounded.statement,
                            "value": _bounded.value,
                        },
                    )()
            if prefast is None:
                from .multivariate_limits_advanced import (
                    coercive_outer_divergence_limit as _coercive_outer,
                )
                from .multivariate_limits_advanced import (
                    denominator_term_vanishing_limit as _denominator_term_zero,
                )
                from .multivariate_limits_advanced import (
                    norm_comparison_divergence_limit as _norm_divergence,
                )
                from .multivariate_limits_advanced import (
                    positive_quotient_divergence_limit as _positive_quotient,
                )
                from .multivariate_limits_advanced import (
                    power_log_vanishing_limit as _power_log_zero,
                )
                from .multivariate_limits_advanced import (
                    radial_order_vanishing_limit as _radial_vanishing,
                )
                from .multivariate_limits_advanced import (
                    uniform_vanishing_amplitude_limit as _uniform_vanishing,
                )

                _uv = _radial_vanishing(
                    expr, active_variables, active_target, domain=domain
                )
                if _uv.status is not _AdvancedLimitStatus.CERTIFIED:
                    _uv = _uniform_vanishing(
                        expr, active_variables, active_target, domain=domain
                    )
                if _uv.status is _AdvancedLimitStatus.CERTIFIED:
                    prefast = type(
                        "_UniformAmplitudeCertificate",
                        (),
                        {
                            "method": _uv.provider,
                            "statement": _uv.statement,
                            "value": _uv.value,
                        },
                    )()
                else:
                    _cd = _coercive_outer(
                        expr, active_variables, active_target, domain=domain
                    )
                    if _cd.status is _AdvancedLimitStatus.CERTIFIED:
                        prefast = type(
                            "_CoerciveOuterCertificate",
                            (),
                            {
                                "method": _cd.provider,
                                "statement": _cd.statement,
                                "value": _cd.value,
                            },
                        )()
                    else:
                        _pq = _positive_quotient(
                            expr, active_variables, active_target, domain=domain
                        )
                        if _pq.status is _AdvancedLimitStatus.CERTIFIED:
                            prefast = type(
                                "_PositiveQuotientCertificate",
                                (),
                                {
                                    "method": _pq.provider,
                                    "statement": _pq.statement,
                                    "value": _pq.value,
                                },
                            )()
                        else:
                            _nd = _norm_divergence(
                                expr, active_variables, active_target, domain=domain
                            )
                            if _nd.status is _AdvancedLimitStatus.CERTIFIED:
                                prefast = type(
                                    "_NormDivergenceCertificate",
                                    (),
                                    {
                                        "method": _nd.provider,
                                        "statement": _nd.statement,
                                        "value": _nd.value,
                                    },
                                )()
                            else:
                                _dz = _denominator_term_zero(
                                    expr, active_variables, active_target, domain=domain
                                )
                                if _dz.status is not _AdvancedLimitStatus.CERTIFIED:
                                    _dz = _power_log_zero(
                                        expr,
                                        active_variables,
                                        active_target,
                                        domain=domain,
                                    )
                                if _dz.status is _AdvancedLimitStatus.CERTIFIED:
                                    prefast = type(
                                        "_DominatedZeroCertificate",
                                        (),
                                        {
                                            "method": _dz.provider,
                                            "statement": _dz.statement,
                                            "value": _dz.value,
                                        },
                                    )()
            if prefast is None:
                _outer_inf = _monotone_infinite_outer_certificate(
                    expr, active_variables, active_target, domain
                )
                if _outer_inf is not None:
                    _oi_value, _oi_evidence = _outer_inf
                    prefast = type(
                        "_InfiniteOuterCertificate",
                        (),
                        {
                            "method": _oi_evidence.method,
                            "statement": _oi_evidence.statement,
                            "value": _oi_value,
                        },
                    )()
            # Exact conflicting path witnesses are negative certificates and
            # take precedence over positive fast-path certificates that do not
            # themselves carry complete cluster-set coverage.
            from .limit_planner import measure_limit_stage

            with measure_limit_stage("negative_witness"):
                conflict = _path_conflict(
                    expr, active_variables, active_target, domain=domain
                )
            if conflict is not None:
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.DOES_NOT_EXIST,
                    None,
                    reduction + conflict,
                    active_variables,
                    domain,
                )
            elif prefast is not None:
                _prefast_dne = prefast.method.endswith("_dne")
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.DOES_NOT_EXIST if _prefast_dne else LimitStatus.PROVED,
                    None if _prefast_dne else prefast.value,
                    reduction
                    + (
                        LimitEvidence(
                            prefast.method,
                            prefast.statement,
                            value=None if _prefast_dne else prefast.value,
                        ),
                    ),
                    active_variables,
                    domain,
                )
            else:
                # Complete oscillatory cluster sets with a uniformly vanishing
                # additive perturbation are cheaper than candidate generation,
                # Newton fans, or CAD/QE.  Try this narrowly before those
                # fallbacks; incomplete cluster descriptions still decline.
                from .cluster_semantics import (
                    ClusterLimitStatus as _EarlyClusterStatus,
                )
                from .cluster_semantics import (
                    cluster_limit_semantics as _early_cluster_semantics,
                )
                from .multivariate_limits_advanced import (
                    oscillatory_cluster_set as _early_osc_cluster,
                )

                _early_cluster = _early_osc_cluster(
                    expr, active_variables, active_target, domain=domain
                )
                _early_sem = _early_cluster_semantics(_early_cluster)
                if _early_sem.status is _EarlyClusterStatus.DOES_NOT_EXIST:
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        target,
                        LimitStatus.DOES_NOT_EXIST,
                        None,
                        reduction
                        + (
                            LimitEvidence(
                                _early_cluster.provider, _early_cluster.statement
                            ),
                        ),
                        active_variables,
                        domain,
                    )
                    if return_result:
                        return result
                    raise SimultaneousLimitDoesNotExist(
                        "complete early oscillatory cluster set is non-singleton"
                    )
                candidate_evidence: list[LimitEvidence] = []
                proved = None
                # The executable planner orders the finite Newton atlas before
                # candidate/QE work; positive Newton coverage is normally much
                # cheaper than constructing epsilon-delta CAD sentences.
                if "newton_fan" in proof_plan.stages:
                    from .limit_planner import measure_limit_stage
                    from .multivariate_limits_advanced import (
                        AdvancedLimitStatus,
                        newton_fan_limit,
                    )

                    try:
                        with measure_limit_stage("newton_fan"):
                            fan = newton_fan_limit(
                                expr, active_variables, active_target, domain=domain
                            )
                    except (TypeError, ValueError, NotImplementedError, sp.PoleError):
                        fan = None
                    if fan is not None and fan.status is AdvancedLimitStatus.CERTIFIED:
                        proved = fan.value
                        candidate_evidence.append(
                            LimitEvidence(
                                "newton_fan_atlas", fan.statement, value=fan.value
                            )
                        )
                if proved is None and "semialgebraic" in proof_plan.stages:
                    for candidate, evidence in _candidate_values(
                        expr, active_variables, active_target, domain
                    ):
                        candidate_evidence.append(evidence)
                        from .limit_planner import measure_limit_stage

                        with measure_limit_stage("semialgebraic"):
                            certification = _semialgebraic_certify_candidate(
                                expr, active_variables, active_target, candidate, domain
                            )
                        if certification is None:
                            continue
                        verdict, cert_evidence = certification
                        candidate_evidence.append(cert_evidence)
                        if verdict is True:
                            proved = candidate
                            break
                        if verdict is False:
                            break
                if proved is None:
                    transcendental = _transcendental_composition_certificate(
                        expr, active_variables, active_target, domain
                    )
                    if transcendental is not None:
                        proved, composition_evidence = transcendental
                        candidate_evidence.extend(composition_evidence)
                if proved is None:
                    from .multivariate_limits_advanced import (
                        AdvancedLimitStatus,
                        bounded_factor_envelope_limit,
                    )

                    oscillatory = bounded_factor_envelope_limit(
                        expr, active_variables, active_target, domain=domain
                    )
                    if oscillatory.status is AdvancedLimitStatus.CERTIFIED:
                        proved = oscillatory.value
                        candidate_evidence.append(
                            LimitEvidence(
                                "bounded_oscillation_envelope",
                                oscillatory.statement,
                                value=oscillatory.value,
                            )
                        )
                if proved is not None:
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        target,
                        LimitStatus.PROVED,
                        proved,
                        reduction + tuple(candidate_evidence),
                        active_variables,
                        domain,
                    )
                else:
                    # Unified cluster-set semantics is the final proof layer.
                    # Only a certified complete cluster set may prove a limit
                    # (singleton) or DNE (non-singleton); partial atlases remain
                    # UNKNOWN.
                    from .cluster_semantics import cluster_semantics_to_limit_result
                    from .multivariate_limits_advanced import newton_fan_cluster_set

                    cluster = newton_fan_cluster_set(
                        expr, active_variables, active_target, domain=domain
                    )
                    cluster_result = cluster_semantics_to_limit_result(cluster)
                    result = SimultaneousLimitResult(
                        expr,
                        variables,
                        target,
                        cluster_result.status,
                        cluster_result.value,
                        reduction + tuple(candidate_evidence) + cluster_result.evidence,
                        active_variables,
                        domain,
                    )

    # Parameter-dependent Newton certificates must expose their sufficient
    # coefficient regime instead of promoting a generic-parameter
    # candidate to an unconditional value.
    free_parameters = expr.free_symbols - set(variables)
    if (
        result.status is LimitStatus.PROVED
        and free_parameters
        and assumptions is sp.S.true
        and any(ev.method == "newton_fan_atlas" for ev in result.evidence)
    ):
        try:
            den = sp.fraction(sp.cancel(expr))[1]
            poly = sp.Poly(den, *variables)
            coeffs = poly.coeffs()
            conditions = []
            for c in coeffs:
                if c.free_symbols & free_parameters:
                    conditions.append(sp.Gt(c, 0))
                elif c.is_nonnegative is not True:
                    conditions = []
                    break
            if conditions:
                result = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.UNKNOWN,
                    None,
                    result.evidence
                    + (
                        LimitEvidence(
                            "parameter_coefficient_regime",
                            "Newton certificate is conditional on positive parameter-dependent denominator coefficients",
                        ),
                    ),
                    active_variables,
                    domain,
                    result.value,
                    sp.And(*conditions),
                )
        except (TypeError, ValueError, sp.PolynomialError):
            pass

    if result.status is LimitStatus.PROVED and not _resolved_limit_value(
        result.value, active_variables
    ):
        result = SimultaneousLimitResult(
            expr,
            variables,
            target,
            LimitStatus.UNKNOWN,
            None,
            result.evidence
            + (
                LimitEvidence(
                    "unresolved_symbolic_value",
                    "candidate contains assumption-dependent or unevaluated symbolic singular data",
                ),
            ),
            active_variables,
            domain,
        )
    if return_result:
        return result
    if result.status is LimitStatus.PROVED:
        return result.value
    if result.status is LimitStatus.DOES_NOT_EXIST:
        raise SimultaneousLimitDoesNotExist(
            "conflicting exact path limits prove nonexistence"
        )
    return result


def _expression_size_within(
    expr: sp.Basic, *, nodes: int = 180, depth: int = 24
) -> bool:
    stack = [(expr, 0)]
    seen = 0
    while stack:
        node, level = stack.pop()
        seen += 1
        if seen > nodes or level > depth:
            return False
        stack.extend((arg, level + 1) for arg in node.args)
    return True


def _exact_expression_identity(
    left: sp.Expr, right: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> bool:
    if left == right:
        return True
    if not (_expression_size_within(left) and _expression_size_within(right)):
        return False
    try:
        if left.is_rational_function(*variables) and right.is_rational_function(
            *variables
        ):
            return sp.cancel(left - right) == 0
    except (TypeError, ValueError, sp.PolynomialError):
        return False
    return False


def _exact_domain_identity(left: sp.Expr, right: sp.Expr) -> bool:
    if left is sp.S.true or left == right:
        return True
    if not (_expression_size_within(left) and _expression_size_within(right)):
        return False
    if isinstance(left, sp.Expr) and isinstance(right, sp.Expr):
        return sp.expand(left - right) == 0
    return False


def _exact_radial_reduction(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    target: tuple[sp.Expr, ...],
    domain: sp.Expr,
) -> tuple[sp.Expr, sp.Symbol] | None:
    from .instrumentation import record_symbolic_event

    if sp.sympify(expr).has(*variables):
        record_symbolic_event("radial_reduction_calls")
    if len(variables) < 2 or any(value in (sp.oo, -sp.oo) for value in target):
        return None
    if not _expression_size_within(expr):
        return None
    if domain is not sp.S.true and not _expression_size_within(domain):
        return None

    radius_squared = sp.Dummy("radius_squared", positive=True)
    offsets = tuple(
        variable - value for variable, value in zip(variables, target, strict=True)
    )
    squared_distance = sum(offset**2 for offset in offsets)
    probe = dict(zip(variables, target, strict=True))
    probe[variables[0]] = target[0] + sp.sqrt(radius_squared)

    radial = expr.subs(probe)
    if radial.has(*variables) or not _expression_size_within(radial):
        return None
    reconstructed = radial.xreplace({radius_squared: squared_distance})
    if not _exact_expression_identity(expr, reconstructed, variables):
        return None

    if domain is not sp.S.true:
        radial_domain = domain.subs(probe)
        if radial_domain.has(*variables) or not _expression_size_within(radial_domain):
            return None
        reconstructed_domain = radial_domain.xreplace(
            {radius_squared: squared_distance}
        )
        if not _exact_domain_identity(domain, reconstructed_domain):
            return None
    return radial, radius_squared


def _mixed_infinity_separable_limit(expr, variables, target, domain):
    """Certify mixed finite/infinite ends when additive terms separate."""
    if sp.sympify(domain) != sp.S.true:
        return None
    infinite = tuple(i for i, a in enumerate(target) if a in (sp.oo, -sp.oo))
    finite = tuple(i for i, a in enumerate(target) if a not in (sp.oo, -sp.oo))
    if not infinite or not finite:
        return None
    values = []
    for term in sp.Add.make_args(sp.sympify(expr)):
        inf_used = tuple(i for i in infinite if variables[i] in term.free_symbols)
        fin_used = tuple(i for i in finite if variables[i] in term.free_symbols)
        if inf_used and fin_used:
            return None
        if inf_used:
            value = _simultaneous_infinity_limit(
                term,
                tuple(variables[i] for i in inf_used),
                tuple(target[i] for i in inf_used),
                sp.S.true,
            )
        else:
            try:
                value = sp.simplify(
                    term.subs(
                        {variables[i]: target[i] for i in fin_used}, simultaneous=True
                    )
                )
            except (TypeError, ValueError, NotImplementedError):
                return None
        if value is None:
            return None
        values.append(value)
    value = sp.Add(*values)
    return None if value in (sp.nan, sp.zoo) else value


def _simultaneous_infinity_limit(expr, variables, target, domain):
    """Certify elementary growth and decay at a product of real infinities.

    The certificate is restricted to unrestricted approach domains and expressions
    whose factors have coordinatewise monotone asymptotic bounds.  This covers
    sums/products of powers, logarithms, and exponentials without reducing a
    simultaneous product-end to iterated limits.
    """
    if sp.sympify(domain) != sp.S.true:
        return None
    signs = []
    positive_vars = []
    substitutions = {}
    for variable, point in zip(variables, target, strict=True):
        if point not in (sp.oo, -sp.oo):
            return None
        sign = 1 if point is sp.oo else -1
        signs.append(sign)
        u = sp.Dummy(f"_infinity_abs_{variable}", positive=True)
        positive_vars.append(u)
        substitutions[variable] = sign * u
    transformed = sp.sympify(expr).subs(substitutions, simultaneous=True)

    def positive_polynomial_growth(poly_expr):
        """Return a positive coercive lower-bound scale for a simple polynomial."""
        try:
            poly = sp.Poly(sp.expand(poly_expr), *positive_vars)
        except sp.PolynomialError:
            return None
        terms = poly.terms()
        if not terms:
            return None
        # A sum of nonnegative monomials is coercive on the product end when
        # at least one positive-degree monomial is present.
        if all(coeff.is_nonnegative is True for _, coeff in terms):
            degrees = [
                sum(mon)
                for mon, coeff in terms
                if coeff.is_positive is True and sum(mon) > 0
            ]
            # Coercivity on a product end only needs a positive-degree term:
            # every positive coordinate tends to infinity, though at unrelated rates.
            if degrees:
                return min(degrees)
        # A positive-definite quadratic leading form is coercive even when its
        # expanded polynomial contains negative cross coefficients.
        if poly.total_degree() == 2 and all(sum(mon) == 2 for mon, _ in terms):
            hessian = sp.hessian(poly.as_expr(), positive_vars) / 2
            if hessian.is_positive_definite is True:
                return 2
        return None

    if transformed.func == sp.exp:
        decay_degree = positive_polynomial_growth(-sp.expand(transformed.args[0]))
        if decay_degree is not None:
            return sp.S.Zero
        growth_degree = positive_polynomial_growth(sp.expand(transformed.args[0]))
        if growth_degree is not None:
            return sp.oo

    def classify(node):
        node = sp.sympify(node)
        if node.is_Mul:
            exponential_factors = [
                factor for factor in node.args if factor.func == sp.exp
            ]
            if len(exponential_factors) > 1:
                combined_exponent = sp.Add(
                    *(factor.args[0] for factor in exponential_factors)
                )
                other_factors = [
                    factor for factor in node.args if factor.func != sp.exp
                ]
                node = sp.Mul(*other_factors, sp.exp(combined_exponent))
        if node.func == sp.exp:
            arg = sp.expand(node.args[0])
            degree = positive_polynomial_growth(arg)
            if degree is not None:
                return ("posinf", None)
            degree = positive_polynomial_growth(-arg)
            if degree is not None:
                return ("expzero", degree)
        node = sp.powsimp(sp.factor_terms(node))
        if node.is_Number:
            if node == 0:
                return ("finite", sp.S.Zero)
            return ("finite", node)
        if node in positive_vars:
            return ("posinf", None)
        if node.is_Pow and node.base in positive_vars and node.exp.is_real:
            if node.exp.is_positive:
                return ("posinf", None)
            if node.exp.is_negative:
                return ("zero", sp.S.Zero)
        if node.func == sp.exp:
            arg = sp.expand(node.args[0])
            degree = positive_polynomial_growth(arg)
            if degree is not None:
                return ("posinf", None)
            degree = positive_polynomial_growth(-arg)
            if degree is not None:
                return ("expzero", degree)
        if node.func == sp.log and node.args[0] in positive_vars:
            return ("posinf", None)
        if node.is_Mul:
            parts = [classify(arg) for arg in node.args]
            if any(kind is None for kind, _ in parts):
                return (None, None)
            exp_decay = [data for kind, data in parts if kind == "expzero"]
            growing = [kind for kind, _ in parts if kind == "posinf"]
            zero = [kind for kind, _ in parts if kind == "zero"]
            finite = [data for kind, data in parts if kind == "finite"]
            if exp_decay and all(
                kind in {"expzero", "posinf", "zero", "finite"} for kind, _ in parts
            ):
                # Exponential decay dominates polynomial, logarithmic, and
                # algebraic factors on the product end.
                return ("zero", sp.S.Zero)
            if (
                growing
                and not zero
                and all(kind in {"posinf", "finite"} for kind, _ in parts)
            ):
                coeff = sp.sympify(sp.prod(finite))
                if coeff.is_positive is True:
                    return ("posinf", None)
                if coeff.is_negative is True:
                    return ("neginf", None)
            if (
                zero
                and not growing
                and all(kind in {"zero", "finite"} for kind, _ in parts)
            ):
                return ("zero", sp.S.Zero)
        if node.is_Add:
            parts = [classify(arg) for arg in node.args]
            if all(kind == "zero" for kind, _ in parts):
                return ("zero", sp.S.Zero)
            if all(kind in {"posinf", "zero", "finite"} for kind, _ in parts) and any(
                kind == "posinf" for kind, _ in parts
            ):
                finite_values = [data for kind, data in parts if kind == "finite"]
                if all(v.is_finite is True for v in finite_values):
                    return ("posinf", None)
        degree = positive_polynomial_growth(node)
        if degree is not None:
            return ("posinf", None)
        return (None, None)

    kind, value = classify(transformed)
    if kind == "zero":
        return sp.S.Zero
    if kind == "posinf":
        return sp.oo
    if kind == "neginf":
        return -sp.oo
    if kind == "finite":
        return value
    return None


def _public_limit_impl(
    expr,
    variables,
    target,
    *,
    domain=True,
    assumptions=True,
    return_result=False,
    _recurrence_attempted=False,
):
    """Compute a proof-aware univariate or joint multivariate limit.

    A scalar variable/target pair denotes a one-dimensional limit. A sequence
    of variables and matching target coordinates denotes their joint limit; it
    is never interpreted as an iterated limit. Unsupported symbolic geometry
    is represented as UNKNOWN with evidence rather than escaping this API.
    """
    if isinstance(variables, sp.Symbol):
        variables = (variables,)
        target = (target,)
    vs = _normalize_variables(variables)
    from .multivariate_limits_advanced import (
        _normalize_extended_target,
        projective_limit_chart,
    )

    extended_target = _normalize_extended_target(vs, target)
    # Preserve the exact represented Float point during symbolic substitution.
    # Premature numeric evaluation can reverse the sign of tiny cancellations.
    extended_target = tuple(
        sp.Rational(p) if isinstance(p, sp.Float) else p for p in extended_target
    )
    variables, target = vs, extended_target
    from .periodic_approaches import square_wave_infinite_conflict

    periodic = square_wave_infinite_conflict(
        sp.sympify(expr),
        vs,
        extended_target,
        sp.sympify(domain),
        sp.sympify(assumptions),
    )
    if periodic is not None:
        from .path_limits import _value_or_result

        result = _certified_germ_result(
            sp.sympify(expr),
            sp.sympify(expr),
            vs,
            extended_target,
            periodic,
            sp.sympify(domain),
        )
        return _value_or_result(result, return_result)
    if any(
        (v.is_real is True and p.is_finite is True and p.is_real is False)
        or (v.is_nonnegative is True and p.is_negative is True)
        or (v.is_nonpositive is True and p.is_positive is True)
        for v, p in zip(vs, extended_target, strict=True)
    ):
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            evidence=(
                LimitEvidence(
                    "incompatible_finite_target",
                    "The coordinate type or sign has no approach to the target.",
                ),
            ),
            domain=sp.sympify(domain),
        )
    if len(vs) == 1:
        from .parameter_special_germs import special_parameter_limit

        special = special_parameter_limit(
            sp.sympify(expr),
            vs[0],
            extended_target[0],
            sp.sympify(domain),
            sp.sympify(assumptions),
        )
        if special is not None:
            from .path_limits import _value_or_result

            return _value_or_result(special, return_result)

    if len(vs) == 1 and extended_target[0] is sp.zoo:
        from .complex_limits import complex_limit

        if (
            vs[0].is_real is None
            and vs[0].is_imaginary is None
            and sp.sympify(domain) is sp.S.true
        ):
            return complex_limit(
                expr,
                vs[0],
                sp.zoo,
                assumptions=assumptions,
                return_result=return_result,
            )

    from .parameter_conditions import elementary_parameter_stratification

    elementary_strata = elementary_parameter_stratification(
        sp.sympify(expr),
        vs,
        extended_target,
        sp.sympify(domain),
        sp.sympify(assumptions),
    )
    if elementary_strata is not None:
        return (
            elementary_strata if return_result else elementary_strata.mathematical_value
        )
    from .fixed_ray_branch_germs import DirectionalInfinity

    if len(vs) == 1 and isinstance(extended_target[0], DirectionalInfinity):
        from .path_limits import (
            _pullback_result,
            _validate_finite_ray,
            complex_ray_limit,
        )

        variable = vs[0]
        direction = extended_target[0].args[0]
        try:
            _validate_finite_ray(variable, sp.S.Zero, direction, assumptions)
            compatible = True
        except ValueError:
            compatible = False
        if (
            sp.sympify(domain) is not sp.S.true
            or sp.sympify(assumptions) is sp.S.false
            or not compatible
        ):
            return SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                LimitStatus.UNKNOWN,
                None,
                (
                    LimitEvidence(
                        "directional_tail_contract",
                        "The ray must satisfy the original variable domain and approach assumptions.",
                    ),
                ),
                vs,
                sp.sympify(domain),
            )
        from .perturbation_scale_germs import (
            principal_lambert_analytic_scale_certificate,
        )

        certificate = (
            principal_lambert_analytic_scale_certificate(
                sp.sympify(expr), variable, extended_target[0]
            )
            if sp.sympify(domain) is sp.S.true and sp.sympify(assumptions) is sp.S.true
            else None
        )
        if certificate is None and sp.sympify(assumptions) is sp.S.true:
            from .lambert_small_germs import reciprocal_lambert_germ_certificate

            certificate = reciprocal_lambert_germ_certificate(
                sp.sympify(expr),
                variable,
                extended_target[0],
                sp.sympify(domain),
                sp.sympify(assumptions),
            )
        if certificate is not None and variable.is_real is not True:
            status, value, evidence = certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                evidence if isinstance(evidence, tuple) else (evidence,),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else result.value
        from .complex_tail_germs import imaginary_tail_certificate

        tail_certificate = imaginary_tail_certificate(
            sp.sympify(expr), variable, direction
        )
        if tail_certificate is not None and sp.sympify(assumptions) is sp.S.true:
            status, value, evidence = tail_certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                evidence,
                vs,
                sp.sympify(domain),
            )
            if return_result:
                return result
            if status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(
                    "Attained imaginary-tail phases disagree"
                )
            return value
        reciprocal = sp.Dummy("reciprocal_ray", positive=True)
        chart = direction / reciprocal
        result = complex_ray_limit(
            sp.sympify(expr).subs(variable, chart),
            reciprocal,
            0,
            ray=1,
            assumptions=sp.sympify(assumptions).subs(variable, chart),
            return_result=True,
        )
        result = _pullback_result(
            sp.sympify(expr),
            variable,
            extended_target[0],
            reciprocal,
            chart,
            result,
            "reciprocal_directional_chart",
            "z=direction/t with t>0 approaches the declared directional infinity; witnesses are pulled back to the original variable.",
            sp.sympify(domain),
        )
        if return_result or result.status is LimitStatus.UNKNOWN:
            return result
        if result.status is LimitStatus.DOES_NOT_EXIST:
            raise SimultaneousLimitDoesNotExist(
                "directional tail subsequences disagree"
            )
        return result.value
    if sp.sympify(assumptions) is sp.S.false or sp.sympify(domain) is sp.S.false:
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "empty_approach_contract",
                    "Explicitly false assumptions or domain admit no approach points; no limit or nonexistence is certified.",
                ),
            ),
            vs,
            sp.sympify(domain),
        )
    from .parameter_conditions import conditional_growth_certificate

    growth = (
        conditional_growth_certificate(
            sp.sympify(expr), vs[0], extended_target[0], sp.sympify(assumptions)
        )
        if len(vs) == 1
        else None
    )
    if growth is not None and (
        sp.sympify(domain) is sp.S.true
        or sp.sympify(domain) == (vs[0] > extended_target[0])
    ):
        status, value, evidence = growth
        result = SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            status,
            value,
            (evidence,),
            vs,
            sp.sympify(domain),
        )
        return result if return_result else value
    if isinstance(sp.sympify(expr), sp.Piecewise):
        piece = sp.sympify(expr)
        if all(not (condition.free_symbols & set(vs)) for _, condition in piece.args):
            branches = []
            evidence = []
            previous = sp.S.false
            for value, condition in piece.args:
                cell = sp.And(sp.sympify(assumptions), condition, sp.Not(previous))
                previous = sp.Or(previous, condition)
                if cell is sp.S.false:
                    continue
                branch = _public_limit_impl(
                    value,
                    vs,
                    extended_target,
                    domain=domain,
                    assumptions=cell,
                    return_result=True,
                    _recurrence_attempted=_recurrence_attempted,
                )
                if (
                    not isinstance(branch, SimultaneousLimitResult)
                    or branch.status is not LimitStatus.PROVED
                ):
                    return SimultaneousLimitResult(
                        piece,
                        vs,
                        extended_target,
                        LimitStatus.UNKNOWN,
                        evidence=(
                            LimitEvidence(
                                "piecewise_parameter_strata_prerequisite",
                                "Every active fixed-parameter cell must be proved; an unresolved cell cannot be discarded as nonaccumulating.",
                            ),
                        ),
                        domain=sp.sympify(domain),
                    )
                branches.append((branch.value, condition))
                evidence.extend(branch.evidence)
            result = SimultaneousLimitResult(
                piece,
                vs,
                extended_target,
                LimitStatus.PROVED,
                sp.Piecewise(*branches),
                tuple(evidence),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else result.value
    root_parameters = set()
    for piece in sp.sympify(expr).atoms(sp.Piecewise):
        for _, condition in piece.args:
            if condition.has(sp.im):
                root_parameters.update(
                    p
                    for p in condition.free_symbols - set(vs)
                    if p.is_real is not True
                    and sp.Q.real(p) not in sp.And.make_args(sp.sympify(assumptions))
                )
    if root_parameters:
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            evidence=(
                LimitEvidence(
                    "parameter_branch_prerequisite",
                    "A real/complex Piecewise branch depends on unresolved fixed-parameter reality. Its principal branch cannot replace its real-root branch.",
                ),
            ),
            domain=sp.sympify(domain),
        )
    from .function_normalization import normalize_functions

    normalized, translation = normalize_functions(
        expr, vs, extended_target, sp.sympify(assumptions)
    )
    if translation:
        from dataclasses import replace

        result = _public_limit_impl(
            normalized,
            vs,
            extended_target,
            domain=domain,
            assumptions=assumptions,
            return_result=True,
            _recurrence_attempted=_recurrence_attempted,
        )
        if isinstance(result, SimultaneousLimitResult):
            result = replace(
                result,
                expression=sp.sympify(expr),
                evidence=translation + result.evidence,
            )
            if return_result or result.status is LimitStatus.UNKNOWN:
                return result
            if result.status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(
                    "definition-normalized limit has incompatible attained limits"
                )
            return result.value
        return result
    from .weierstrass_origin import (
        WeierstrassSigma,
        WeierstrassZeta,
        weierstrass_origin_certificate,
    )

    if sp.sympify(expr).has(WeierstrassZeta, WeierstrassSigma):
        certificate = (
            weierstrass_origin_certificate(
                sp.sympify(expr),
                vs[0],
                extended_target[0],
                sp.sympify(domain),
                sp.sympify(assumptions),
            )
            if len(vs) == 1
            else None
        )
        if certificate is not None:
            status, value, evidence = certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                (evidence,),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else value
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "weierstrass_germ_prerequisite",
                    "The origin remainder certificate does not establish this parameter, domain, target or amplified cancellation.",
                ),
            ),
            vs,
            sp.sympify(domain),
        )
    from .function_normalization import TriangleWave

    if sp.sympify(expr).has(TriangleWave):
        from .rational_pullback_germs import (
            triangle_reciprocal_phase_certificate,
        )

        certificate = (
            triangle_reciprocal_phase_certificate(
                sp.sympify(expr),
                vs[0],
                extended_target[0],
                sp.sympify(domain),
                sp.sympify(assumptions),
            )
            if len(vs) == 1
            else None
        )
        if certificate is not None:
            status, value, evidence = certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                evidence,
                vs,
                sp.sympify(domain),
            )
            if return_result:
                return result
            raise SimultaneousLimitDoesNotExist(
                "triangle-wave phases attain distinct limits"
            )
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "triangle_wave_prerequisite",
                    "Only certified real reciprocal-phase subsequences are implemented; unsupported domains, complex arguments and poles remain explicit.",
                ),
            ),
            vs,
            sp.sympify(domain),
        )
    from .function_normalization import NearestInteger

    if sp.sympify(expr).has(NearestInteger):
        from .rational_pullback_germs import rounded_tail_certificate

        certificate = None
        if (
            len(vs) == 1
            and extended_target[0] is sp.oo
            and vs[0].is_real is not False
            and sp.sympify(domain) is sp.S.true
            and not sp.sympify(assumptions).has(vs[0])
            and sp.sympify(assumptions) is not sp.S.false
        ):
            positive = sp.Dummy("round_positive_tail", positive=True)
            certificate = rounded_tail_certificate(
                sp.sympify(expr).subs(vs[0], positive), positive, sp.oo
            )
        if certificate is not None:
            status, value, evidence = certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                (evidence,),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else value
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "rounding_germ_prerequisite",
                    "The nearest-integer error certificate does not establish this target, domain or amplified discontinuity.",
                ),
            ),
            vs,
            sp.sympify(domain),
        )
    from sympy.core.function import AppliedUndef

    from .function_normalization import InverseRegularizedGamma, RealRoot
    from .reference_normalization import Normal, Series
    from .step_factorials import StepFactorialPower

    if sp.sympify(expr).has(
        AppliedUndef,
        RealRoot,
        InverseRegularizedGamma,
        StepFactorialPower,
        Normal,
        Series,
    ):
        result = SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "undefined_function_prerequisite",
                    "undefined function heads have no supplied local/asymptotic theorem",
                ),
            ),
            vs,
            sp.sympify(domain),
        )
        return result
    if (
        len(vs) == 1
        and extended_target[0] is sp.oo
        and sp.sympify(domain) is sp.S.true
        and sp.sympify(assumptions) is sp.S.true
        and sp.sympify(expr).func not in (sp.Min, sp.Max)
        and sp.sympify(expr).has(sp.Min, sp.Max)
    ):
        from .univariate_perturbation_limits import elementary_minmax_tail_certificate

        positive = sp.Dummy("minmax_positive_tail", positive=True)
        certificate = elementary_minmax_tail_certificate(
            sp.sympify(expr).subs(vs[0], positive), positive, sp.oo
        )
        if certificate is not None:
            status, value, evidence = certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                (evidence,),
                vs,
                sp.S.true,
            )
            return result if return_result else value
    # Exact constant parameter equalities are substitutions, not asymptotic
    # growth questions. Apply them before invoking generic infinity machinery.
    equality_substitution = {}
    for clause in sp.And.make_args(sp.sympify(assumptions)):
        if isinstance(clause, sp.Equality):
            for parameter, value in (
                (clause.lhs, clause.rhs),
                (clause.rhs, clause.lhs),
            ):
                if (
                    isinstance(parameter, sp.Symbol)
                    and parameter not in vs
                    and value.is_number is True
                    and value.is_finite is True
                ):
                    if (
                        parameter in equality_substitution
                        and equality_substitution[parameter] != value
                    ):
                        equality_substitution = {}
                        break
                    equality_substitution[parameter] = value
    reduced_assumptions = sp.sympify(assumptions).xreplace(equality_substitution)
    if equality_substitution and reduced_assumptions is not sp.S.false:
        from dataclasses import replace

        r = _public_limit_impl(
            sp.sympify(expr).xreplace(equality_substitution),
            vs,
            tuple(p.xreplace(equality_substitution) for p in extended_target),
            domain=sp.sympify(domain).xreplace(equality_substitution),
            assumptions=reduced_assumptions,
            return_result=True,
            _recurrence_attempted=_recurrence_attempted,
        )
        if isinstance(r, SimultaneousLimitResult):
            r = replace(
                r,
                expression=sp.sympify(expr),
                target=extended_target,
                domain=sp.sympify(domain),
            )
            if return_result or r.status is LimitStatus.UNKNOWN:
                return r
            if r.status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(
                    "parameter-equality-normalized limit has incompatible sides"
                )
            return r.value
        return r
    # Assumption-proved signs are encoded in temporary parameters so native
    # local algebra cannot discard them. Map all output data back.
    if sp.sympify(assumptions) is not sp.S.true:
        from dataclasses import replace

        from ._symbolic_policy import bounded_ask

        parameters = (
            sp.sympify(expr).free_symbols | sp.Tuple(*extended_target).free_symbols
        ) - set(vs)
        substitution = {}
        assumption_clauses = sp.And.make_args(sp.sympify(assumptions))
        for parameter in parameters:
            if parameter.is_positive is not True and (
                sp.Gt(parameter, 0) in assumption_clauses
                or (
                    sp.Ge(parameter, 0) in assumption_clauses
                    and sp.Ne(parameter, 0) in assumption_clauses
                )
                or bounded_ask(sp.Q.positive(parameter), assumptions) is True
            ):
                substitution[parameter] = sp.Dummy(str(parameter), positive=True)
            elif parameter.is_negative is not True and (
                sp.Lt(parameter, 0) in assumption_clauses
                or bounded_ask(sp.Q.negative(parameter), assumptions) is True
            ):
                substitution[parameter] = sp.Dummy(str(parameter), negative=True)
            elif (
                parameter.is_real is not True
                and sp.Q.real(parameter) in assumption_clauses
            ):
                substitution[parameter] = (
                    sp.Dummy(str(parameter), real=True, nonzero=True)
                    if sp.Ne(parameter, 0) in assumption_clauses
                    else sp.Dummy(str(parameter), real=True)
                )
        if substitution:
            inverse = {v: k for k, v in substitution.items()}
            r = _public_limit_impl(
                sp.sympify(expr).xreplace(substitution),
                vs,
                tuple(p.xreplace(substitution) for p in extended_target),
                domain=sp.sympify(domain).xreplace(substitution),
                assumptions=sp.sympify(assumptions).xreplace(substitution),
                return_result=True,
                _recurrence_attempted=_recurrence_attempted,
            )
            if isinstance(r, SimultaneousLimitResult):
                r = replace(
                    r,
                    expression=sp.sympify(expr),
                    target=extended_target,
                    value=r.value.xreplace(inverse) if r.value is not None else None,
                    domain=sp.sympify(domain),
                    evidence=tuple(
                        replace(
                            e,
                            value=e.value.xreplace(inverse)
                            if isinstance(e.value, sp.Basic)
                            else e.value,
                            substitutions=tuple(
                                (
                                    inverse.get(v, v),
                                    s.xreplace(inverse)
                                    if isinstance(s, sp.Basic)
                                    else s,
                                )
                                for v, s in e.substitutions
                            ),
                        )
                        for e in r.evidence
                    ),
                )
                if return_result or r.status is LimitStatus.UNKNOWN:
                    return r
                if r.status is LimitStatus.DOES_NOT_EXIST:
                    raise SimultaneousLimitDoesNotExist(
                        "assumption-aware scalar germs disagree"
                    )
                return r.value
            return r
    if (
        len(vs) == 1
        and extended_target[0].is_finite is True
        and extended_target[0].is_real is False
    ):
        from .univariate_complex_rational import finite_complex_rational_certificate

        certificate = finite_complex_rational_certificate(
            sp.sympify(expr),
            vs[0],
            extended_target[0],
            sp.sympify(domain),
            sp.sympify(assumptions),
        )
        if certificate is None:
            from .special_function_limit_germs import finite_complex_entire_certificate

            certificate = finite_complex_entire_certificate(
                sp.sympify(expr),
                vs[0],
                extended_target[0],
                sp.sympify(domain),
                sp.sympify(assumptions),
            )
        if (
            certificate is None
            and sp.sympify(domain) is sp.S.true
            and sp.sympify(assumptions) is sp.S.true
            and vs[0].is_real is None
            and vs[0].is_imaginary is None
            and not (sp.sympify(expr).free_symbols - {vs[0]})
        ):
            from .complex_limits import complex_limit

            whole = complex_limit(expr, vs[0], extended_target[0], return_result=True)
            if whole.status is not LimitStatus.UNKNOWN:
                from .path_limits import _value_or_result

                return _value_or_result(whole, return_result)
        if certificate is not None:
            status, value, evidence = certificate
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                (evidence,),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else value
        return SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            extended_target,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "complex_target_prerequisite",
                    "This complex target requires a branch, pole, or approach-domain theorem outside finite rational continuity.",
                ),
            ),
            vs,
            sp.sympify(domain),
        )
    if len(vs) == 1 and isinstance(sp.sympify(expr), sp.Expr):
        from .limit_certificates import scalar_limit_certificate

        certificate = scalar_limit_certificate(
            sp.sympify(expr),
            vs[0],
            extended_target[0],
            sp.sympify(domain),
            sp.sympify(assumptions),
        )
        if certificate is not None:
            status, value, evidence = certificate
            certificate_evidence = (
                evidence if isinstance(evidence, tuple) else (evidence,)
            )
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                certificate_evidence,
                vs,
                sp.sympify(domain),
            )
            if return_result:
                return result
            if status is LimitStatus.DOES_NOT_EXIST:
                raise SimultaneousLimitDoesNotExist(
                    "; ".join(e.statement for e in certificate_evidence)
                )
            return value
    if len(vs) == 1 and extended_target[0] not in (sp.oo, -sp.oo):
        # Substitution on a special-function cut does not establish either
        # approached boundary value. Preserve the missing branch theorem.
        coordinate = sp.Dummy("special_cut_coordinate", real=True)
        for atom in sp.sympify(expr).atoms(sp.li, sp.uppergamma):
            argument = atom.args[-1]
            boundary = sp.simplify(argument.subs(vs[0], extended_target[0]))
            local_argument = argument.subs(vs[0], extended_target[0] + coordinate)
            if boundary.is_negative is True and sp.simplify(sp.im(local_argument)) != 0:
                return SimultaneousLimitResult(
                    sp.sympify(expr),
                    vs,
                    extended_target,
                    LimitStatus.UNKNOWN,
                    None,
                    (
                        LimitEvidence(
                            "special_function_cut_boundary_prerequisite",
                            "a non-real approach to a negative special-function cut requires its signed boundary germ",
                        ),
                    ),
                    vs,
                    sp.sympify(domain),
                )
    # Only after the original normalized expression's certified scalar routes
    # decline do we spend the recurrence budget. Retry those same routes once;
    # general analysis follows only after both certificate passes decline.
    if (
        len(vs) == 1
        and not _recurrence_attempted
        and isinstance(sp.sympify(expr), sp.Expr)
    ):
        from dataclasses import replace

        from .recurrence_simplification import simplify_recurrences

        simplified = simplify_recurrences(
            expr,
            assumptions=assumptions,
            variables=vs,
            target=extended_target,
            require_defined_germ=True,
            return_result=True,
        )
        if simplified.evidence:
            result = _public_limit_impl(
                simplified.expression,
                vs,
                extended_target,
                domain=domain,
                assumptions=assumptions,
                return_result=True,
                _recurrence_attempted=True,
            )
            if isinstance(result, SimultaneousLimitResult):
                result = replace(
                    result,
                    expression=sp.sympify(expr),
                    evidence=simplified.evidence + result.evidence,
                )
                if return_result or result.status is LimitStatus.UNKNOWN:
                    return result
                if result.status is LimitStatus.DOES_NOT_EXIST:
                    raise SimultaneousLimitDoesNotExist(
                        "recurrence-normalized limit has incompatible attained limits"
                    )
                return result.value
            return result
    if len(vs) == 1:
        if sp.sympify(assumptions) is sp.S.true:
            prerequisite = unresolved_origin_parameter(expr, vs, extended_target)
            if prerequisite is not None:
                return SimultaneousLimitResult(
                    sp.sympify(expr),
                    vs,
                    extended_target,
                    LimitStatus.UNKNOWN,
                    None,
                    (
                        LimitEvidence(
                            prerequisite,
                            "Free origin parameters require a proved parameter stratification before generic path comparison.",
                        ),
                    ),
                    vs,
                    sp.sympify(domain),
                )
    if len(vs) == 1:
        from ._recurrence_families import uncertified_germ_reason

        reason = uncertified_germ_reason(sp.sympify(expr), vs, extended_target)
        if reason is not None:
            return SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                LimitStatus.UNKNOWN,
                None,
                (LimitEvidence("recurrence_family_germ_prerequisite", reason),),
                vs,
                sp.sympify(domain),
            )
    if len(vs) == 1:
        # An unbounded periodic pole term cannot be discarded by a growth
        # comparison on the unrestricted tail. Its accumulating poles require
        # avoidance bounds or explicit subsequence certificates.
        if sp.sympify(domain) is sp.S.true and extended_target[0] in (sp.oo, -sp.oo):
            pole_atoms = sp.sympify(expr).atoms(sp.tan, sp.cot, sp.sec, sp.csc)
            for atom in pole_atoms:
                phase = atom.args[0]
                phase_value = bounded_limit(
                    phase, vs[0], extended_target[0], allow_general=True
                )
                if phase_value in (sp.oo, -sp.oo) or phase.has(
                    sp.tan, sp.cot, sp.sec, sp.csc
                ):
                    result = SimultaneousLimitResult(
                        sp.sympify(expr),
                        vs,
                        extended_target,
                        LimitStatus.UNKNOWN,
                        None,
                        (
                            LimitEvidence(
                                "periodic_pole_avoidance_prerequisite",
                                "an unrestricted tail with accumulating periodic poles needs explicit avoidance or attained-subsequence bounds",
                            ),
                        ),
                        vs,
                        sp.sympify(domain),
                    )
                    return result
    if (
        len(vs) == 1
        and extended_target == (sp.oo,)
        and sp.sympify(expr).has(
            sp.airyai,
            sp.airybi,
            sp.besselj,
            sp.bessely,
            sp.erf,
            sp.erfc,
            sp.gamma,
            sp.zeta,
        )
    ):
        from .univariate_engine import analyze_univariate

        try:
            analysis = analyze_univariate(
                sp.sympify(expr), vs[0], point=sp.oo, direction="+"
            )
            special = analysis.infinity_asymptotic
            candidate = None
            if special is not None:
                for attr in ("limit", "value", "leading_value"):
                    candidate = getattr(special, attr, None)
                    if candidate is not None:
                        break
            if candidate is not None and _resolved_limit_value(candidate, vs):
                result = SimultaneousLimitResult(
                    sp.sympify(expr),
                    vs,
                    extended_target,
                    LimitStatus.PROVED,
                    candidate,
                    (
                        LimitEvidence(
                            "univariate_engine",
                            "package-native special-function infinity analysis certified the scalar limit",
                            value=candidate,
                        ),
                    ),
                    vs,
                    sp.sympify(domain),
                )
                return result if return_result else result.value
        except (
            TypeError,
            ValueError,
            NotImplementedError,
            RecursionError,
            sp.PoleError,
        ):
            pass
    if len(vs) == 1 and extended_target == (sp.oo,):
        from .uniform_integration import uniform_limit_value

        uniform_result = uniform_limit_value(sp.sympify(expr), vs[0], sp.oo)
        if uniform_result is not None:
            uniform_value, _uniform_expansion = uniform_result
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                LimitStatus.PROVED,
                uniform_value,
                (
                    LimitEvidence(
                        "uniform_asymptotic_regime",
                        "theorem-selected uniform expansion with vanishing propagated remainder",
                        value=uniform_value,
                    ),
                ),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else result.value
    if len(vs) > 1 and any(value in (sp.oo, -sp.oo) for value in extended_target):
        from .attained_periodic_families import independent_periodic_tail_witness

        periodic = independent_periodic_tail_witness(
            sp.sympify(expr), vs, extended_target, domain, assumptions
        )
        if periodic is not None:
            status, value, evidence = periodic
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                status,
                value,
                evidence,
                vs,
                sp.sympify(domain),
            )
            if return_result:
                return result
            raise SimultaneousLimitDoesNotExist(
                "attained periodic tail values disagree"
            )
        infinity_value = _simultaneous_infinity_limit(expr, vs, extended_target, domain)
        if infinity_value is None:
            infinity_value = _mixed_infinity_separable_limit(
                expr, vs, extended_target, domain
            )
        if infinity_value is not None:
            result = SimultaneousLimitResult(
                sp.sympify(expr),
                vs,
                extended_target,
                LimitStatus.PROVED,
                infinity_value,
                (
                    LimitEvidence(
                        "simultaneous_infinity_growth",
                        "coordinatewise growth bounds certify the product-end limit",
                        value=infinity_value,
                    ),
                ),
                vs,
                sp.sympify(domain),
            )
            return result if return_result else result.value
        # Mixed finite/infinite targets use the reciprocal chart below.  The
        # finite coordinates retain their ordinary local-germ semantics.
    if any(value in (sp.oo, -sp.oo) for value in extended_target):
        chart = projective_limit_chart(expr, vs, extended_target, domain=domain)
        if chart.expression.is_rational_function(*chart.variables):
            from .multivariate_certificates import candidate_rational_limit_certificate

            parameter = sp.Dummy("projective_ray", positive=True)
            witnesses = []
            directions = [tuple(1 for _ in chart.variables)]
            directions.extend(
                tuple(2 if i == j else 1 for i in range(len(chart.variables)))
                for j in range(len(chart.variables))
            )
            for direction in directions:
                ray_substitution = dict(
                    zip(
                        chart.variables, (c * parameter for c in direction), strict=True
                    )
                )
                if sp.simplify(chart.domain.subs(ray_substitution)) is not sp.S.true:
                    continue
                along = sp.cancel(chart.expression.subs(ray_substitution))
                if along.has(sp.nan, sp.zoo):
                    continue
                value = bounded_limit(along, parameter, 0)
                if value is None or value.is_finite is not True:
                    continue
                evidence = LimitEvidence(
                    "projective_rational_ray",
                    "The positive reciprocal ray satisfies the chart domain; its rational denominator is eventually nonzero.",
                    tuple(
                        (variable, image.subs(ray_substitution, simultaneous=True))
                        for variable, image in chart.substitutions
                    ),
                    value,
                )
                if any(
                    sp.simplify(value - previous.value).is_zero is False
                    for previous in witnesses
                ):
                    result = SimultaneousLimitResult(
                        sp.sympify(expr),
                        vs,
                        extended_target,
                        LimitStatus.DOES_NOT_EXIST,
                        None,
                        tuple(witnesses) + (evidence,),
                        vs,
                        sp.sympify(domain),
                    )
                    if return_result:
                        return result
                    raise SimultaneousLimitDoesNotExist(
                        "attained projective rays disagree"
                    )
                witnesses.append(evidence)
            if witnesses:
                certificate = candidate_rational_limit_certificate(
                    chart.expression, chart.variables, chart.target
                )
                if (
                    certificate.certified
                    and certificate.value is not None
                    and certificate.value.is_finite is True
                ):
                    result = SimultaneousLimitResult(
                        sp.sympify(expr),
                        vs,
                        extended_target,
                        LimitStatus.PROVED,
                        certificate.value,
                        (
                            LimitEvidence(
                                "reciprocal_product_chart",
                                "The reciprocal coordinates map the declared product end to zero, with pulled domain "
                                + str(chart.domain),
                                tuple(chart.substitutions),
                            ),
                            LimitEvidence(
                                certificate.method,
                                certificate.statement,
                                value=certificate.value,
                            ),
                        ),
                        vs,
                        sp.sympify(domain),
                    )
                    return result if return_result else result.value
        substitution = dict(chart.substitutions)
        transformed_assumptions = sp.sympify(assumptions).subs(
            substitution, simultaneous=True
        )
        result = _limit_impl(
            chart.expression,
            chart.variables,
            chart.target,
            domain=chart.domain,
            assumptions=transformed_assumptions,
            return_result=True,
        )
        if return_result:
            return result
        if result.status is LimitStatus.PROVED:
            return result.value
        if result.status is LimitStatus.DOES_NOT_EXIST:
            raise SimultaneousLimitDoesNotExist(
                "projective directions give incompatible limits"
            )
        return result
    try:
        return _limit_impl(
            expr,
            variables,
            target,
            domain=domain,
            assumptions=assumptions,
            return_result=return_result,
        )
    except (
        sp.PolynomialError,
        sp.PoleError,
        NotImplementedError,
        RecursionError,
        ZeroDivisionError,
    ) as exc:
        vs = _normalize_variables(variables)
        tar = normalize_limit_target(vs, target)
        result = SimultaneousLimitResult(
            sp.sympify(expr),
            vs,
            tar,
            LimitStatus.UNKNOWN,
            None,
            (
                LimitEvidence(
                    "unsupported_symbolic_geometry", f"{type(exc).__name__}: {exc}"
                ),
            ),
            vs,
            sp.sympify(domain),
            obligations=(
                ProofObligation(
                    ObligationKind.THEOREM_PREREQUISITE,
                    f"resolve unsupported symbolic geometry: {type(exc).__name__}",
                    provider="limit",
                    expression=str(sp.sympify(expr)),
                ),
            ),
        )
        return result


def limit(
    expr, variables, target, *, domain=True, assumptions=True, return_result=False
):
    """Compute a proof-aware limit within a shared call-scoped analysis context."""
    from .limit_planner import limit_analysis_context

    with limit_analysis_context():
        return _public_limit_impl(
            expr,
            variables,
            target,
            domain=domain,
            assumptions=assumptions,
            return_result=return_result,
        )
