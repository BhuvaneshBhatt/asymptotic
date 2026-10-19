"""Compatibility adapter to the package-independent exprtools recurrence engine."""

from dataclasses import dataclass

import sympy as sp
from exprtools.recurrence import simplify_recurrences as _simplify_recurrences

from ._recurrence_families import FAMILIES
from .limit_models import LimitEvidence


@dataclass(frozen=True)
class RecurrenceSimplificationResult:
    original: sp.Expr
    expression: sp.Expr
    evidence: tuple[LimitEvidence, ...] = ()
    steps: int = 0
    stopped: str | None = None


def simplify_recurrences(
    expr,
    *,
    assumptions=True,
    variables=(),
    target=(),
    max_shift=4,
    max_steps=24,
    max_ops=240,
    budget_seconds=0.05,
    require_defined_germ=False,
    return_result=False,
):
    """Retain the legacy signature, function conventions, evidence and counters."""
    from .instrumentation import record_symbolic_event

    result = _simplify_recurrences(
        expr,
        assumptions=assumptions,
        variables=variables,
        target=target,
        max_shift=max_shift,
        max_steps=max_steps,
        max_ops=max_ops,
        budget_seconds=budget_seconds,
        require_defined_germ=require_defined_germ,
        return_result=return_result,
        families=FAMILIES,
        record_event=record_symbolic_event,
    )
    if not return_result:
        return result
    evidence = tuple(
        LimitEvidence(e.method, e.statement, e.substitutions, e.value)
        for e in result.evidence
    )
    return RecurrenceSimplificationResult(
        result.original,
        result.expression,
        evidence,
        result.steps,
        result.stopped,
    )
