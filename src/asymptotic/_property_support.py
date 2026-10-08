"""Asymptotic-specific adapters over the shared :mod:`funcprops` reasoning layer."""

from __future__ import annotations

import sympy as sp
from funcprops import (
    PropertyDecision,
    PropertyKnowledge,
    PropertyProvenance,
    analytic,
    decide,
    entails,
    normalize_assumptions,
    require_decision,
    singularities,
)
from funcprops.results import Analyticity

from ._symbolic_errors import SYMBOLIC_ERRORS


def _provenance(expr: sp.Expr, note: str) -> tuple[PropertyProvenance, ...]:
    return (PropertyProvenance("funcprops", reference=str(expr.func), note=note),)


def analytic_at_decision(
    expr: sp.Expr,
    argument: sp.Symbol,
    value: sp.Expr,
    *,
    assumptions: sp.Expr | bool = True,
) -> PropertyDecision:
    """Return a local complex-analyticity decision using funcprops singularities."""
    expr = sp.sympify(expr)
    argument = sp.sympify(argument)
    value = sp.sympify(value)
    assumptions = normalize_assumptions(assumptions)
    predicate = sp.Symbol(f"analytic_at({sp.sstr(expr)},{sp.sstr(value)})")
    provenance = _provenance(expr, "local complex analyticity")

    # Principal log and square root are not holomorphic across their standard
    # cut on the nonpositive real axis.  A singularity-locus query alone does
    # not encode that branch boundary.
    principal_root = expr.is_Pow and expr.exp == sp.Rational(1, 2)
    if (
        (expr.func is sp.log or principal_root)
        and value.is_real is True
        and value.is_nonpositive is True
    ):
        return PropertyDecision(
            predicate,
            False,
            assumptions,
            PropertyKnowledge.EXACT,
            provenance,
            ("value lies on the principal branch cut",),
        )

    # Unknown function heads must remain conservative even if the generic
    # singularity locus happens to be empty.
    try:
        analyticity_result = analytic(
            expr,
            argument,
            domain=sp.S.Complexes,
            assumptions=assumptions,
            return_result=True,
        )
        analyticity = analyticity_result.resolved
    except (TypeError, ValueError, NotImplementedError):
        analyticity = Analyticity.UNKNOWN
    if analyticity is Analyticity.UNKNOWN:
        return PropertyDecision(
            predicate,
            None,
            assumptions,
            PropertyKnowledge.PARTIAL,
            provenance,
            ("funcprops could not establish holomorphy",),
        )
    if analyticity is Analyticity.NOT_ANALYTIC:
        return PropertyDecision(
            predicate,
            False,
            assumptions,
            PropertyKnowledge.EXACT,
            provenance,
            ("funcprops classifies the expression as nonholomorphic",),
        )

    try:
        locus = singularities(
            expr,
            argument,
            sp.S.Complexes,
            assumptions=assumptions,
        )
        at_value = sp.sympify(locus).subs(argument, value)
        on_locus = entails(at_value, assumptions)
    except (TypeError, ValueError, NotImplementedError):
        on_locus = None
        at_value = sp.Symbol("unknown_singularity_locus")
    if on_locus is True:
        return PropertyDecision(
            predicate,
            False,
            assumptions,
            PropertyKnowledge.EXACT,
            provenance,
            (f"value lies on funcprops singularity locus: {at_value}",),
        )
    if on_locus is False or at_value is sp.S.false:
        return PropertyDecision(
            predicate,
            True,
            assumptions,
            PropertyKnowledge.EXACT,
            provenance,
            ("funcprops singularity locus is excluded at the expansion center",),
        )
    return PropertyDecision(
        predicate,
        None,
        assumptions,
        PropertyKnowledge.PARTIAL,
        provenance,
        (f"could not exclude funcprops singularity locus: {at_value}",),
    )


def branch_safe_substitution_decision(expr, argument, value, *, assumptions=True):
    decision = analytic_at_decision(expr, argument, value, assumptions=assumptions)
    return PropertyDecision(
        sp.Symbol(f"branch_safe({sp.sstr(expr)},{sp.sstr(value)})"),
        decision.verdict,
        decision.assumptions,
        decision.knowledge,
        decision.provenance,
        decision.reasons,
    )


def nested_branch_safety_decisions(expr, argument, value, *, assumptions=True):
    concrete = sp.sympify(expr)
    assumptions = normalize_assumptions(assumptions)
    decisions: list[PropertyDecision] = []
    seen: set[tuple[str, str]] = set()

    def visit(node: sp.Expr) -> None:
        for child in node.args:
            if isinstance(child, sp.Basic) and child.has(argument):
                visit(sp.sympify(child))
        if not node.has(argument):
            return
        principal_root = node.is_Pow and node.exp == sp.Rational(1, 2)
        if principal_root:
            inner = sp.sympify(node.base)
        elif len(node.args) == 1:
            inner = sp.sympify(node.args[0])
        else:
            return
        try:
            inner_value = sp.simplify(inner.subs(argument, value))
        except SYMBOLIC_ERRORS:
            inner_value = inner.subs(argument, value)
        key = (sp.srepr(node.func), sp.srepr(inner_value))
        if key in seen:
            return
        seen.add(key)
        probe = sp.Dummy("branch_arg")
        outer = sp.sqrt(probe) if principal_root else node.func(probe)
        decision = analytic_at_decision(
            outer, probe, inner_value, assumptions=assumptions
        )
        # Skip heads for which funcprops has no knowledge. They are not branch
        # evidence, but a top-level composition will still reject an unknown
        # outer function when it must prove analyticity.
        if decision.verdict is not None or node.func in (sp.log, sp.sqrt):
            decisions.append(decision)

    visit(concrete)
    return tuple(decisions)


def nested_branch_safe_substitution_decision(
    expr, argument, value, *, assumptions=True
):
    assumptions = normalize_assumptions(assumptions)
    decisions = nested_branch_safety_decisions(
        expr, argument, value, assumptions=assumptions
    )
    verdict: bool | None = True
    if any(d.verdict is False for d in decisions):
        verdict = False
    elif any(d.verdict is None for d in decisions):
        verdict = None
    return PropertyDecision(
        sp.Symbol(f"nested_branch_safe({sp.sstr(expr)},{sp.sstr(value)})"),
        verdict,
        assumptions,
        PropertyKnowledge.EXACT if verdict is not None else PropertyKnowledge.PARTIAL,
        tuple(p for d in decisions for p in d.provenance),
        tuple(r for d in decisions for r in d.reasons)
        or ("no nested funcprops branch obstruction",),
    )


__all__ = [
    "PropertyDecision",
    "PropertyKnowledge",
    "PropertyProvenance",
    "analytic_at_decision",
    "branch_safe_substitution_decision",
    "decide",
    "entails",
    "nested_branch_safe_substitution_decision",
    "nested_branch_safety_decisions",
    "normalize_assumptions",
    "require_decision",
]
