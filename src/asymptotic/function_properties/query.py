"""Public query surface for mathematical function properties."""

from __future__ import annotations

import sympy as sp

from .._symbolic_errors import SYMBOLIC_ERRORS
from .model import DomainProperties, FunctionProperties, SingularityProperties
from .registry import DEFAULT_REGISTRY, FunctionPropertyRegistry, PropertyBuilder
from .semantics import (
    PropertyDecision,
    PropertyKnowledge,
    PropertyProvenance,
    PropertyRule,
    entails,
)


def expression_head(expr: sp.Expr) -> object:
    if isinstance(expr, sp.Pow) and expr.exp == sp.Rational(1, 2):
        return "sqrt"
    return expr.func


def register_function_properties(
    function: object, builder: PropertyBuilder, *, registry=DEFAULT_REGISTRY, replace: bool = True
) -> None:
    """Register or replace a reviewed function-property builder in a registry."""
    registry.register(function, builder, replace=replace)


def function_properties(expr: sp.Expr, *, registry=DEFAULT_REGISTRY) -> FunctionProperties | None:
    """Return reviewed mathematical properties for *expr*, or ``None`` if unknown."""
    concrete = sp.sympify(expr)
    builder = registry.builder_for(expression_head(concrete))
    return None if builder is None else builder(concrete)


def domain_properties(expr: sp.Expr, *, registry=DEFAULT_REGISTRY) -> DomainProperties | None:
    """Return the registered domain component for *expr*, when one is available."""
    properties = function_properties(expr, registry=registry)
    return None if properties is None else properties.domain


def singularity_properties(
    expr: sp.Expr, *, registry=DEFAULT_REGISTRY
) -> SingularityProperties | None:
    """Return reviewed singularity and branch-locus information for *expr*."""
    properties = function_properties(expr, registry=registry)
    return None if properties is None else properties.singularities


def _provenance(expr: sp.Expr, note: str | None = None) -> PropertyProvenance:
    return PropertyProvenance(
        "asymptotic.function_properties", reference=str(expression_head(expr)), note=note
    )


def analytic_at_decision(
    expr: sp.Expr,
    argument: sp.Expr,
    value: sp.Expr,
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    registry: FunctionPropertyRegistry = DEFAULT_REGISTRY,
) -> PropertyDecision:
    """Return an auditable tri-state analyticity decision.

    Parameter conditions attached to singularity loci are enforced.  A registry
    entry marked globally locally analytic is accepted after all registered
    singular loci/cuts have been certified absent at the requested value.
    """

    assumptions = sp.sympify(assumptions)
    properties = function_properties(expr, registry=registry)
    predicate = sp.Symbol(f"analytic_at({sp.sstr(expr)},{sp.sstr(value)})")
    prov = (_provenance(expr, "local analyticity and singularity loci"),)
    if properties is None or len(properties.arguments) != 1:
        return PropertyDecision(
            predicate,
            None,
            assumptions,
            PropertyKnowledge.PARTIAL,
            prov,
            ("no reviewed unary property entry",),
        )
    source_arg = properties.arguments[0]
    singularities = properties.singularities
    if singularities is None:
        return PropertyDecision(
            predicate, None, assumptions, PropertyKnowledge.PARTIAL, prov, ("no singularity data",)
        )

    undecided = []
    for label, group in (
        ("pole", singularities.poles),
        ("essential singularity", singularities.essential),
        ("branch point", singularities.branch_points),
        ("definition cut", singularities.definition_cuts),
        ("branch cut", singularities.branch_cuts),
    ):
        if group is None:
            undecided.append(f"{label} data unregistered")
            continue
        for locus in group:
            loc = sp.And(sp.sympify(locus.parameter_condition), sp.sympify(locus.condition))
            loc = sp.simplify(loc.subs(source_arg, value))
            verdict = entails(loc, assumptions)
            if verdict is True:
                return PropertyDecision(
                    predicate,
                    False,
                    assumptions,
                    PropertyKnowledge.EXACT,
                    prov,
                    (f"value lies on registered {label}: {loc}",),
                )
            if verdict is None:
                undecided.append(f"could not exclude {label}: {loc}")

    if undecided:
        return PropertyDecision(
            predicate, None, assumptions, PropertyKnowledge.PARTIAL, prov, tuple(undecided)
        )
    if singularities.locally_analytic is True:
        return PropertyDecision(
            predicate,
            True,
            assumptions,
            PropertyKnowledge.EXACT,
            prov,
            ("all registered singular loci excluded",),
        )
    if singularities.locally_analytic is False:
        # False here means not globally locally analytic, not that every point is bad.
        # Empty/excluded loci still certify an ordinary local point.
        return PropertyDecision(
            predicate,
            True,
            assumptions,
            PropertyKnowledge.SUFFICIENT,
            prov,
            ("all registered local obstructions excluded",),
        )
    return PropertyDecision(
        predicate,
        None,
        assumptions,
        PropertyKnowledge.PARTIAL,
        prov,
        ("local analyticity flag unknown",),
    )


def analytic_at(
    expr, argument, value, *, assumptions=sp.S.true, registry=DEFAULT_REGISTRY
) -> bool | None:
    """Return a tri-state verdict for local analyticity after substituting *value*."""
    return analytic_at_decision(
        expr, argument, value, assumptions=assumptions, registry=registry
    ).verdict


def domain_contains_decision(
    expr: sp.Expr,
    argument: sp.Expr,
    value: sp.Expr,
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    real: bool = False,
    registry: FunctionPropertyRegistry = DEFAULT_REGISTRY,
) -> PropertyDecision:
    assumptions = sp.sympify(assumptions)
    properties = function_properties(expr, registry=registry)
    predicate = sp.Symbol(f"domain_contains({sp.sstr(expr)},{sp.sstr(value)})")
    prov = (_provenance(expr, "domain membership"),)
    if properties is None or properties.domain is None or len(properties.arguments) != 1:
        return PropertyDecision(
            predicate,
            None,
            assumptions,
            PropertyKnowledge.PARTIAL,
            prov,
            ("domain data unavailable",),
        )
    source_arg = properties.arguments[0]
    condition = properties.domain.real_domain if real else properties.domain.complex_domain
    if condition is None:
        return PropertyDecision(
            predicate,
            None,
            assumptions,
            PropertyKnowledge.PARTIAL,
            prov,
            ("requested domain not registered",),
        )
    condition = sp.sympify(condition).subs(source_arg, value)
    verdict = entails(condition, assumptions)
    return PropertyDecision(
        predicate,
        verdict,
        assumptions,
        PropertyKnowledge.EXACT,
        prov,
        (f"domain condition: {condition}",),
    )


def branch_safe_substitution_decision(
    expr: sp.Expr,
    argument: sp.Expr,
    value: sp.Expr,
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    registry: FunctionPropertyRegistry = DEFAULT_REGISTRY,
) -> PropertyDecision:
    """Certify that substitution at ``value`` avoids registered branch obstructions."""

    decision = analytic_at_decision(
        expr, argument, value, assumptions=assumptions, registry=registry
    )
    return PropertyDecision(
        sp.Symbol(f"branch_safe({sp.sstr(expr)},{sp.sstr(value)})"),
        decision.verdict,
        decision.assumptions,
        decision.knowledge,
        decision.provenance,
        decision.reasons,
    )


def function_property_rules(expr: sp.Expr, *, registry=DEFAULT_REGISTRY):
    properties = function_properties(expr, registry=registry)
    if properties is None or properties.assumptions is None:
        return ()
    provenance = _provenance(expr, "value predicate")
    rules = []
    for name in (
        "integer",
        "rational",
        "real",
        "real_if_defined",
        "positive",
        "negative",
        "nonpositive",
        "nonnegative",
    ):
        condition = getattr(properties.assumptions, name)
        if condition is not None:
            rules.append(PropertyRule(condition, name, PropertyKnowledge.SUFFICIENT, provenance))
    return tuple(rules)


def nested_branch_safety_decisions(
    expr: sp.Expr,
    argument: sp.Symbol,
    value: sp.Expr,
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    registry: FunctionPropertyRegistry = DEFAULT_REGISTRY,
) -> tuple[PropertyDecision, ...]:
    """Trace principal-branch safety through every nested unary composition.

    Each registered unary node depending on ``argument`` is checked at the
    exact value reached by its own inner argument.  Inner nodes are visited
    first, so a caller receives a replayable branch path rather than a single
    top-level yes/no flag.
    """
    concrete = sp.sympify(expr)
    assumptions = sp.sympify(assumptions)
    decisions = []
    seen = set()

    def visit(node: sp.Expr) -> None:
        for child in node.args:
            if isinstance(child, sp.Basic) and child.has(argument):
                visit(sp.sympify(child))
        if not node.has(argument):
            return
        head = expression_head(node)
        if head == "sqrt":
            inner = sp.sympify(node.base)
        elif len(node.args) == 1:
            inner = sp.sympify(node.args[0])
        else:
            return
        probe = sp.Dummy("branch_arg")
        registered = registry.builder_for(head)
        if registered is None:
            return
        try:
            inner_value = sp.simplify(inner.subs(argument, value))
        except SYMBOLIC_ERRORS:
            inner_value = inner.subs(argument, value)
        key = (sp.srepr(node.func), sp.srepr(inner_value))
        if key in seen:
            return
        seen.add(key)
        outer = sp.sqrt(probe) if head == "sqrt" else node.func(probe)
        decisions.append(
            branch_safe_substitution_decision(
                outer,
                probe,
                inner_value,
                assumptions=assumptions,
                registry=registry,
            )
        )

    visit(concrete)
    return tuple(decisions)


def nested_branch_safe_substitution_decision(
    expr: sp.Expr,
    argument: sp.Symbol,
    value: sp.Expr,
    *,
    assumptions: sp.Expr | bool = sp.S.true,
    registry: FunctionPropertyRegistry = DEFAULT_REGISTRY,
) -> PropertyDecision:
    """Aggregate :func:`nested_branch_safety_decisions` into one tri-state proof."""
    decisions = nested_branch_safety_decisions(
        expr, argument, value, assumptions=assumptions, registry=registry
    )
    verdict: bool | None = True
    if any(decision.verdict is False for decision in decisions):
        verdict = False
    elif any(decision.verdict is None for decision in decisions):
        verdict = None
    return PropertyDecision(
        sp.Symbol(f"nested_branch_safe({sp.sstr(expr)},{sp.sstr(value)})"),
        verdict,
        sp.sympify(assumptions),
        PropertyKnowledge.EXACT if verdict is not None else PropertyKnowledge.PARTIAL,
        tuple(prov for decision in decisions for prov in decision.provenance),
        tuple(reason for decision in decisions for reason in decision.reasons)
        or ("no registered nested branch obstruction",),
    )
