"""General exhaustive parameter stratification for limit proofs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from .conditional import mathematical_result
from .parameter_auto import parameter_symbols
from .stratification import AsymptoticStratification, ParameterStratum


@dataclass(frozen=True)
class ConditionalProof:
    """A certified mathematical result on one parameter condition."""

    value: sp.Expr
    condition: sp.Expr
    method: str
    statement: str = ""


ConditionalDiscoverer = Callable[..., tuple[ConditionalProof, ...]]


def _certificate_conditions(expr, variables, target, *, domain=True):
    """Adapt conditional data emitted by exact certificate families."""
    if domain is not sp.S.true:
        return ()
    from ._multivariate_analytic import parameter_weighted_order_certificate

    proofs = []
    for discover in (parameter_weighted_order_certificate,):
        cert = discover(expr, variables, target)
        if cert.certified or not cert.data or len(cert.data) < 2:
            continue
        value, condition = cert.data[:2]
        condition = sp.simplify(sp.sympify(condition))
        if condition is sp.S.false:
            continue
        proofs.append(
            ConditionalProof(
                sp.sympify(value),
                condition,
                cert.method,
                cert.statement or "certified result on a parameter regime",
            )
        )
    return tuple(proofs)


_DISCOVERERS: tuple[ConditionalDiscoverer, ...] = (_certificate_conditions,)


def _limit_result(expr, variables, target, domain, proof):
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult

    return SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.PROVED,
        proof.value,
        (
            LimitEvidence(
                proof.method,
                proof.statement,
                value=proof.value,
            ),
        ),
        variables,
        domain,
    )


def _unknown_result(expr, variables, target, domain, condition):
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult

    return SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.UNKNOWN,
        None,
        (
            LimitEvidence(
                "parameter_stratum_unresolved",
                f"no proof mechanism certified this parameter stratum: {condition}",
            ),
        ),
        variables,
        domain,
    )


def _complement_strata(condition):
    """Split a simple strict inequality into boundary and opposite open region."""
    condition = sp.sympify(condition)
    if isinstance(condition, sp.StrictLessThan):
        return (
            sp.Eq(condition.lhs, condition.rhs),
            sp.Gt(condition.lhs, condition.rhs),
        )
    if isinstance(condition, sp.StrictGreaterThan):
        return (
            sp.Eq(condition.lhs, condition.rhs),
            sp.Lt(condition.lhs, condition.rhs),
        )
    complement = sp.simplify_logic(sp.Not(condition))
    return () if complement is sp.S.false else (complement,)


def _resolve_equality_stratum(expr, variables, target, domain, condition, parameters):
    """Evaluate a parameter equality after exact specialization."""
    if not isinstance(condition, sp.Equality):
        return None
    from .limits import limit
    from .parameter_auto import specialize_expression

    specialized = specialize_expression(expr, condition, parameters=parameters)
    if specialized == expr:
        return None
    # A generic result with other unresolved free parameters is not uniform on
    # this equality stratum. Let parameter-aware cluster geometry refine those
    # remaining parameters instead of accepting a generic path verdict.
    if specialized.free_symbols - set(variables):
        return None
    result = limit(
        specialized,
        variables,
        target,
        domain=domain,
        return_result=True,
    )
    # A specialization must eliminate the equality's solved parameter. Refuse
    # nested parameter stratifications here; they belong to a later refinement.
    if isinstance(result, AsymptoticStratification):
        return None
    return result


def conditional_limit_stratification(
    expr, variables, target, *, domain=True, branch_budget: int = 3
):
    """Collect proof regimes and explicitly cover their unresolved complement.

    The returned stratification is exhaustive whenever any conditional proof is
    discovered. Certified regimes retain their values; every remaining
    parameter point is represented by an explicit UNKNOWN stratum rather than
    being omitted.
    """
    expr = sp.sympify(expr)
    variables = tuple(variables)
    target = tuple(target)
    domain = sp.sympify(domain)
    parameters = parameter_symbols(expr, variables)
    if not parameters:
        return None

    proofs = []
    for discoverer in _DISCOVERERS:
        proofs.extend(discoverer(expr, variables, target, domain=domain))
    if not proofs:
        return None

    strata = [
        ParameterStratum(
            sp.simplify(proof.condition),
            _limit_result(expr, variables, target, domain, proof),
        )
        for proof in proofs
    ]
    covered = sp.simplify_logic(sp.Or(*(proof.condition for proof in proofs)))
    complements = (
        _complement_strata(proofs[0].condition)
        if len(proofs) == 1
        else _complement_strata(covered)
    )
    if len(strata) + len(complements) > branch_budget:
        return None
    for complement in complements:
        resolved = _resolve_equality_stratum(
            expr, variables, target, domain, complement, parameters
        )
        from .limit_models import LimitStatus

        if resolved is None or resolved.status is LimitStatus.UNKNOWN:
            from .cluster_semantics import cluster_semantics_to_limit_result
            from .parameter_clusters import parameter_cluster_strata

            cluster_strata = parameter_cluster_strata(
                expr, variables, target, complement, domain=domain
            )
            if cluster_strata:
                for cluster_stratum in cluster_strata:
                    strata.append(
                        ParameterStratum(
                            cluster_stratum.condition,
                            cluster_semantics_to_limit_result(
                                cluster_stratum.cluster_result
                            ),
                        )
                    )
                continue
        strata.append(
            ParameterStratum(
                complement,
                resolved
                if resolved is not None
                else _unknown_result(expr, variables, target, domain, complement),
            )
        )
    return AsymptoticStratification(
        parameters,
        tuple(strata),
        exhaustive=True,
    )


def conditional_limit_value(expr, variables, target, *, domain=True):
    """Return the mathematical value over all certified parameter strata."""
    stratification = conditional_limit_stratification(
        expr, variables, target, domain=domain
    )
    if stratification is None:
        return None
    return mathematical_result(stratification)


__all__ = [
    "ConditionalProof",
    "conditional_limit_stratification",
    "conditional_limit_value",
]


def elementary_parameter_stratification(expr, variables, target, domain, assumptions):
    """Separate fixed real growth rates, retaining unresolved complex parameters."""
    if len(variables) != 1 or domain is not sp.S.true or assumptions is not sp.S.true:
        return None
    x, point = variables[0], target[0]
    if (
        x.is_real is False
        or (point == sp.oo and x.is_nonpositive is True)
        or (point == -sp.oo and x.is_nonnegative is True)
    ):
        return None
    expr = sp.sympify(expr)
    parameter = None
    if (
        expr.is_Pow
        and expr.base == x
        and point == sp.oo
        and isinstance(expr.exp, sp.Symbol)
    ):
        parameter = expr.exp
        values = (sp.S.Zero, sp.S.One, sp.oo)
    elif expr.func is sp.exp and point in (sp.oo, -sp.oo):
        slope = sp.diff(expr.args[0], x)
        if isinstance(slope, sp.Symbol) and expr.args[0] == slope * x:
            parameter = slope
            values = (
                (sp.S.Zero, sp.S.One, sp.oo)
                if point == sp.oo
                else (sp.oo, sp.S.One, sp.S.Zero)
            )
    if parameter is None or parameter == x or parameter.is_finite is False:
        return None
    if (
        parameter.is_positive is True
        or parameter.is_negative is True
        or parameter.is_zero is True
    ):
        return None
    real = sp.S.true if parameter.is_real is True else sp.Q.real(parameter)
    conditions = (
        sp.And(real, parameter < 0),
        sp.Eq(parameter, 0),
        sp.And(real, parameter > 0),
    )
    strata = []
    for condition, value in zip(conditions, values, strict=True):
        proof = ConditionalProof(
            value,
            condition,
            "fixed_parameter_growth_strata",
            "A fixed finite real rate determines exponential growth/decay, or the sign of the real logarithmic power exponent. The zero-rate expression is identically one.",
        )
        strata.append(
            ParameterStratum(
                condition, _limit_result(expr, variables, target, domain, proof)
            )
        )
    if real is not sp.S.true:
        strata.append(
            ParameterStratum(
                sp.Not(real),
                _unknown_result(expr, variables, target, domain, sp.Not(real)),
            )
        )
    return AsymptoticStratification((parameter,), tuple(strata), exhaustive=True)


def conditional_growth_certificate(expression, variable, point, assumptions):
    """Use explicit growth inequalities without repeatedly querying assumptions."""
    if (
        assumptions is sp.S.true
        or variable.is_real is False
        or sp.count_ops(expression) > 65
    ):
        return None
    from .limit_models import LimitEvidence, LimitStatus
    from .local_tail_germs import fixed_finite

    clauses = sp.And.make_args(assumptions)
    if point == sp.oo:
        for atom in expression.atoms(sp.Pow):
            base, exponent = atom.args
            rate = sp.diff(exponent, variable)
            if (
                base.has(variable)
                or not fixed_finite(base)
                or rate.is_positive is not True
                or rate.has(variable)
                or exponent != rate * variable
            ):
                continue
            if (
                sp.Gt(sp.log(base), 0) in clauses
                and sp.Gt(base**rate, 1) in clauses
                and expression == atom / (base**rate - 1)
            ):
                return (
                    LimitStatus.PROVED,
                    sp.oo,
                    LimitEvidence(
                        "declared_positive_power_growth",
                        "A strictly positive real principal logarithm implies a fixed real base above one. The stated denominator inequality makes the fixed denominator positive, so the positive real power grows without bound.",
                        value=sp.oo,
                    ),
                )
    if point == 0 and len(expression.free_symbols - {variable}) == 1:
        a = next(iter(expression.free_symbols - {variable}))
        if sp.Gt(a, 1) in clauses and expression == sp.sinh(a / variable**2) * sp.csch(
            1 / variable**2
        ):
            return (
                LimitStatus.PROVED,
                sp.oo,
                LimitEvidence(
                    "fixed_hyperbolic_ratio_growth",
                    "On either real side u=1/x^2 tends to positive infinity. The exact exponential definitions give sinh(a*u)/sinh(u)=exp((a-1)*u)*(1-exp(-2*a*u))/(1-exp(-2*u)); a>1 proves positive divergence and eventual denominator avoidance.",
                    value=sp.oo,
                ),
            )
    return None


def rational_corner_stratification(expr, variables, target, domain, assumptions):
    """Separate the removable parameter from attained rational corner limits."""
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult

    if (
        len(variables) != 2
        or tuple(target) != (0, 0)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 50
    ):
        return None
    parameters = expr.free_symbols - set(variables)
    if len(parameters) != 1:
        return None
    a = next(iter(parameters))
    if not isinstance(a, sp.Symbol) or a.is_finite is not True:
        return None
    for x, y in (variables, variables[::-1]):
        for q in range(1, 5):
            vanishing = x * y * y / (x ** (2 * q) + a * y * y)
            if expr == vanishing and a.is_real is True:
                j = sp.Dummy(
                    "attained_parameter_pole_index", positive=True, integer=True
                )
                y_curve = sp.Piecewise(
                    (j ** (-(2 * q - 1)), sp.Eq(a, 0)),
                    (j ** (-2 * q) * (1 + j ** (-2)) / sp.sqrt(-a), True),
                )
                curve_value = sp.Piecewise((1, sp.Eq(a, 0)), (1 / (2 * a), True))
                proved = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.PROVED,
                    sp.S.Zero,
                    (
                        LimitEvidence(
                            "positive_rational_corner_bound",
                            "For a>0, abs(x*y**2/(x**(2*q)+a*y**2)) <= abs(x)/a "
                            "on the defined domain, uniformly tending to zero.",
                            value=sp.S.Zero,
                        ),
                    ),
                    variables,
                    domain,
                )
                disagreement = SimultaneousLimitResult(
                    expr,
                    variables,
                    target,
                    LimitStatus.DOES_NOT_EXIST,
                    None,
                    (
                        LimitEvidence(
                            "rational_pole_axis_subsequence",
                            "At x=1/j**2,y=0 the denominator is positive and the value is zero.",
                            ((x, j ** (-2)), (y, 0)),
                            sp.S.Zero,
                        ),
                        LimitEvidence(
                            "rational_parameter_pole_subsequence",
                            "At a=0, y=j**(-(2*q-1)) with x=j**(-2) gives exactly one. "
                            "At a<0, y=x**q*(1+x)/sqrt(-a) makes the denominator "
                            "x**(2*q)*(1-(1+x)**2), nonzero for every x>0. The ratio "
                            "tends to 1/(2*a). These nonzero values differ from the axis limit.",
                            ((x, j ** (-2)), (y, y_curve)),
                            curve_value,
                        ),
                    ),
                    variables,
                    domain,
                )
                return AsymptoticStratification(
                    (a,),
                    (
                        ParameterStratum(a > 0, proved),
                        ParameterStratum(a <= 0, disagreement),
                    ),
                    exhaustive=True,
                )
            template = (x ** (2 * q) + y * y) / (x ** (2 * q) + a * y * y)
            if expr != template:
                continue
            j = sp.Dummy("attained_rational_corner_index", positive=True, integer=True)
            c = sp.Piecewise((2, sp.Eq(a, -1)), (1, True))
            value = (1 + c * c) / (1 + a * c * c)
            proved = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.PROVED,
                sp.S.One,
                (
                    LimitEvidence(
                        "removable_rational_parameter",
                        "At a=1 numerator and denominator coincide; the denominator "
                        "x**(2*q)+y**2 is positive away from the origin.",
                        value=sp.S.One,
                    ),
                ),
                variables,
                domain,
            )
            disagreement = SimultaneousLimitResult(
                expr,
                variables,
                target,
                LimitStatus.DOES_NOT_EXIST,
                None,
                (
                    LimitEvidence(
                        "rational_corner_axis_subsequence",
                        "For x=1/j,y=0 the denominator is positive and the value is one.",
                        ((x, 1 / j), (y, 0)),
                        sp.S.One,
                    ),
                    LimitEvidence(
                        "rational_corner_curve_subsequence",
                        "For y=c*x**q choose c=2 at a=-1 and c=1 otherwise. "
                        "Then 1+a*c**2 is nonzero on every finite parameter cell. "
                        "The denominator is (1+a*c**2)*x**(2*q), so every "
                        "point is defined. The constant value differs from one "
                        "exactly when a!=1.",
                        ((x, 1 / j), (y, c / j**q)),
                        value,
                    ),
                ),
                variables,
                domain,
            )
            return AsymptoticStratification(
                (a,),
                (
                    ParameterStratum(sp.Eq(a, 1), proved),
                    ParameterStratum(sp.Ne(a, 1), disagreement),
                ),
                exhaustive=True,
            )
    return None


def displaced_pole_stratification(expr, variables, target, domain, assumptions):
    """Separate a fixed displaced rational pole from its singular corner.

    Nonzero finite complex displacements give a regular rational germ.
    The zero displacement has two attained real rays with distinct values.
    """
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult

    if (
        len(variables) != 2
        or tuple(target) != (0, 0)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 20
        # The attained paths use positive coordinates and an axis containing zero.
        or any(
            v.is_real is not True
            or v.is_integer is True
            or v.is_nonpositive is True
            or v.is_nonzero is True
            for v in variables
        )
    ):
        return None
    parameters = expr.free_symbols - set(variables)
    if len(parameters) != 1:
        return None
    a = next(iter(parameters))
    if (
        not isinstance(a, sp.Symbol)
        or a.is_finite is False
        or a.is_nonzero is True
        or a.is_zero is True
    ):
        return None
    match = next(
        (
            (x, y)
            for x, y in (variables, variables[::-1])
            if expr == sp.cancel(x * y / ((x - a) ** 2 + y**2))
        ),
        None,
    )
    if match is None:
        return None
    x, y = match
    regular = SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.PROVED,
        sp.S.Zero,
        (
            LimitEvidence(
                "displaced_pole_continuity",
                "For each fixed finite complex a!=0 the denominator tends to a**2!=0. "
                "Rational continuity gives zero on a full neighborhood of the origin.",
                value=sp.S.Zero,
            ),
        ),
        variables,
        domain,
    )
    j = sp.Dummy("attained_displaced_pole_index", positive=True, integer=True)
    singular = SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "displaced_pole_axis_subsequence",
                "At a=0, x=1/j,y=0 gives zero with denominator 1/j**2>0.",
                ((x, 1 / j), (y, sp.S.Zero)),
                sp.S.Zero,
            ),
            LimitEvidence(
                "displaced_pole_diagonal_subsequence",
                "At a=0, x=y=1/j gives 1/2 with denominator 2/j**2>0. "
                "Both attained sequences approach the origin and their limits differ.",
                ((x, 1 / j), (y, 1 / j)),
                sp.Rational(1, 2),
            ),
        ),
        variables,
        domain,
    )
    return AsymptoticStratification(
        (a,),
        (
            ParameterStratum(sp.Ne(a, 0), regular),
            ParameterStratum(sp.Eq(a, 0), singular),
        ),
        exhaustive=True,
    )


def scaled_corner_stratification(expr, variables, target, domain, assumptions):
    """Retain coefficient-zero cells of two radial angular corner families.

    The axis and a balanced curve attain distinct finite limits, so the proof
    does not need a signed or complex infinity convention.
    """
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult

    if (
        len(variables) != 2
        or tuple(target) != (0, 0)
        or domain is not sp.S.true
        or assumptions is not sp.S.true
        or not isinstance(expr, sp.Expr)
        or sp.count_ops(expr) > 40
        # The attained paths use positive coordinates and an axis containing zero.
        or any(
            v.is_real is not True
            or v.is_integer is True
            or v.is_nonpositive is True
            or v.is_nonzero is True
            for v in variables
        )
    ):
        return None
    coefficient, core = expr.as_independent(*variables, as_Add=False)
    parameters = coefficient.free_symbols
    if len(parameters) != 1:
        return None
    a = next(iter(parameters))
    if not isinstance(a, sp.Symbol) or a.is_finite is False:
        return None
    _, denominator = sp.fraction(coefficient)
    if denominator != 1 or coefficient.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return None
    entire = {sp.sin, sp.cos, sp.sinh, sp.cosh, sp.exp}
    if any(
        atom.func not in entire or not atom.args[0].is_polynomial(a)
        for atom in coefficient.atoms(sp.Function)
    ):
        return None
    if any(
        power.exp.is_integer is not True or power.exp.is_nonnegative is not True
        for power in coefficient.atoms(sp.Pow)
        if power.base.has(a)
    ):
        return None
    matched = None
    for x, y in (variables, variables[::-1]):
        radius_squared = x * x + y * y
        if core == sp.log(1 + y * y / radius_squared) / sp.sqrt(radius_squared):
            matched = (x, y, sp.S.Zero, sp.S.One)
        elif core == 1 + y * y / radius_squared ** sp.Rational(3, 2):
            matched = (x, y, sp.S.One, sp.Integer(2))
        if matched is not None:
            break
    if matched is None:
        return None
    x, y, axis, curve = matched
    zero = SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.PROVED,
        sp.S.Zero,
        (
            LimitEvidence(
                "zero_corner_coefficient",
                "When the fixed finite coefficient is zero, every defined approach value is zero.",
                value=sp.S.Zero,
            ),
        ),
        variables,
        domain,
    )
    j = sp.Dummy("attained_scaled_corner_index", positive=True, integer=True)
    conflict = SimultaneousLimitResult(
        expr,
        variables,
        target,
        LimitStatus.DOES_NOT_EXIST,
        None,
        (
            LimitEvidence(
                "scaled_corner_axis_subsequence",
                "The attained axis x=1/j**2,y=0 has positive radial denominator.",
                ((x, j ** (-2)), (y, sp.S.Zero)),
                coefficient * axis,
            ),
            LimitEvidence(
                "scaled_corner_balanced_subsequence",
                "The attained curve x=1/j**2,y=1/j**3 has positive squared radius "
                "j**(-4)*(1+j**(-2)). With t=j**(-2), the logarithmic corner is "
                "log(1+t/(1+t))/(t*sqrt(1+t)), tending to one by log1p(s)/s->1. "
                "The rational angular corner is 1+(1+t)**(-3/2), tending to two. "
                "Both denominators are nonzero, and the logarithm argument is positive. "
                "The two finite attained limits differ whenever the fixed coefficient is nonzero.",
                ((x, j ** (-2)), (y, j ** (-3))),
                coefficient * curve,
            ),
        ),
        variables,
        domain,
    )
    return AsymptoticStratification(
        (a,),
        (
            ParameterStratum(sp.Eq(coefficient, 0), zero),
            ParameterStratum(sp.Ne(coefficient, 0), conflict),
        ),
        exhaustive=True,
    )
