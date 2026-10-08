"""Parameter stratification for multivariate limits."""

from __future__ import annotations

import sympy as sp


def parameter_stratified_limit(expr, variables, target, parameters, *, domain=True):
    """Return an explicit :class:`AsymptoticStratification` for target fibers.

    This conservative layer certifies direct nonvanishing-denominator fibers and
    represents unresolved singular fibers with an ordinary UNKNOWN limit result.
    It never returns the tuple representation.
    """
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult
    from .limit_primitives import _normalize_variables, normalize_limit_target
    from .stratification import AsymptoticStratification, ParameterStratum

    expr = sp.sympify(expr)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    parameters = tuple(parameters)
    domain = sp.sympify(domain)
    subs = dict(zip(variables, target, strict=True))
    num, den = sp.fraction(sp.cancel(expr))
    n0, d0 = sp.simplify(num.subs(subs)), sp.simplify(den.subs(subs))

    def result(status, value=None, evidence=()):
        return SimultaneousLimitResult(
            expr, variables, target, status, value, tuple(evidence), variables, domain
        )

    unknown = result(
        LimitStatus.UNKNOWN,
        evidence=(
            LimitEvidence(
                "parameter_target_singular",
                "target denominator vanishes on this parameter fiber; direct substitution is inconclusive",
            ),
        ),
    )
    if d0.is_zero is True:
        return AsymptoticStratification(
            parameters,
            (
                ParameterStratum(
                    sp.S.true,
                    unknown,
                    complete=False,
                    limitations=(
                        "singular target fiber unresolved by direct substitution",
                    ),
                ),
            ),
            exhaustive=True,
        )
    value = sp.cancel(n0 / d0)
    proved = result(
        LimitStatus.PROVED,
        value,
        (
            LimitEvidence(
                "parameter_target_substitution",
                "denominator is nonzero at the target on this parameter fiber",
                value=value,
            ),
        ),
    )
    if d0.is_zero is False:
        return AsymptoticStratification(
            parameters, (ParameterStratum(sp.S.true, proved),), exhaustive=True
        )
    return AsymptoticStratification(
        parameters,
        (
            ParameterStratum(sp.Ne(d0, 0), proved),
            ParameterStratum(
                sp.Eq(d0, 0),
                unknown,
                complete=False,
                limitations=(
                    "singular target fiber unresolved by direct substitution",
                ),
            ),
        ),
        exhaustive=True,
    )


def recursive_parameter_stratified_limit(
    expr,
    variables,
    target,
    parameters,
    *,
    domain=sp.S.true,
    assumptions=sp.S.true,
    max_splits=8,
):
    """Recursively stratify parameter space and evaluate singular fibers.

    Structural zero/nonzero coefficients are split recursively by the package's
    exhaustive parameter stratifier; each leaf is evaluated by the ordinary
    certified multivariate engine under that leaf's assumptions.
    """
    from .limits import limit
    from .parameter_auto import (
        automatic_parameter_stratification,
        specialize_expression,
    )

    parameters = tuple(parameters)

    def evaluator(condition):
        specialized = specialize_expression(expr, condition, parameters=parameters)
        return limit(specialized, variables, target, domain=domain, return_result=True)

    structural = [sp.sympify(expr)]
    # Equal-degree rational leading forms can change from direction-dependent
    # to removable when their coefficient vectors become proportional.  Add
    # the 2x2 coefficient minors as exact parameter split conditions.
    try:
        num, den = sp.fraction(sp.cancel(sp.sympify(expr)))
        pn, pd = sp.Poly(num, *variables), sp.Poly(den, *variables)
        mons = sorted(set(pn.monoms()) | set(pd.monoms()))
        cn = [pn.coeff_monomial(m) for m in mons]
        cd = [pd.coeff_monomial(m) for m in mons]
        for i in range(len(mons)):
            for j in range(i + 1, len(mons)):
                minor = sp.factor(cn[i] * cd[j] - cn[j] * cd[i])
                if minor != 0 and minor.free_symbols & set(parameters):
                    structural.append(minor)
    except (sp.PolynomialError, TypeError, ValueError):
        pass
    strat = automatic_parameter_stratification(
        tuple(structural),
        evaluator,
        parameters=parameters,
        assumptions=assumptions,
        max_splits=max_splits,
        provenance_source="asymptotic.recursive_parameter_stratified_limit",
    )
    if strat is not None:
        return strat

    # A parameter may survive only as the value of a limit rather than as a
    # structure-changing condition (for example a removable quotient tending
    # to ``a`` for every real ``a``).  Evaluate that all-parameter fiber before
    # falling back to target-denominator splitting; the latter is # too weak when the denominator vanishes identically at the target.
    direct = limit(expr, variables, target, domain=domain, return_result=True)
    if direct.status.name == "PROVED":
        from .stratification import AsymptoticStratification, ParameterStratum

        return AsymptoticStratification(
            parameters,
            (ParameterStratum(sp.S.true, direct),),
            assumptions=assumptions,
            exhaustive=True,
        )
    return parameter_stratified_limit(
        expr, variables, target, parameters, domain=domain
    )
