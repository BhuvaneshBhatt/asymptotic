"""Knowledge semantics, provenance and enforceable tri-state decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import sympy as sp

from .._symbolic_policy import bounded_assumption_entails, bounded_simplify
from ..canonical import canonical_key


class PropertyKnowledge(str, Enum):
    EXACT = "exact"
    SUFFICIENT = "sufficient"
    NECESSARY = "necessary"
    PARTIAL = "partial"


@dataclass(frozen=True)
class PropertyProvenance:
    source: str
    reference: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class PropertyRule:
    condition: sp.Expr
    value: object
    knowledge: PropertyKnowledge = PropertyKnowledge.EXACT
    provenance: PropertyProvenance | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", sp.sympify(self.condition))


@dataclass(frozen=True)
class PropertyDecision:
    """A tri-state mathematical decision together with auditable evidence."""

    predicate: sp.Expr
    verdict: bool | None
    assumptions: sp.Expr = sp.S.true
    knowledge: PropertyKnowledge = PropertyKnowledge.EXACT
    provenance: tuple[PropertyProvenance, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", sp.sympify(self.predicate))
        object.__setattr__(self, "assumptions", sp.sympify(self.assumptions))

    @property
    def certified(self) -> bool:
        return self.verdict is not None and self.knowledge is not PropertyKnowledge.PARTIAL


class PropertyEnforcementError(ValueError):
    """Raised when a required mathematical precondition is false or unresolved."""

    def __init__(self, decision: PropertyDecision, *, operation: str | None = None):
        self.decision = decision
        self.operation = operation
        state = "false" if decision.verdict is False else "unresolved"
        prefix = f"{operation}: " if operation else ""
        super().__init__(f"{prefix}required property is {state}: {decision.predicate}")


def _structural_entailment(condition: sp.Expr, assumptions: sp.Expr) -> bool | None:
    """Resolve cheap propositional cases without invoking assumptions/SAT."""

    if condition is sp.S.true or assumptions is sp.S.false:
        return True
    if condition is sp.S.false:
        return False
    if assumptions is sp.S.true:
        return None

    ckey = canonical_key(condition)
    if ckey == canonical_key(assumptions):
        return True
    neg = sp.Not(condition, evaluate=False)
    nkey = canonical_key(neg)

    clauses = sp.And.make_args(assumptions)
    clause_keys = {canonical_key(clause) for clause in clauses}
    if ckey in clause_keys:
        return True
    if nkey in clause_keys:
        return False

    if isinstance(condition, sp.And):
        values = tuple(_structural_entailment(arg, assumptions) for arg in condition.args)
        if all(value is True for value in values):
            return True
        if any(value is False for value in values):
            return False
    elif isinstance(condition, sp.Or):
        values = tuple(_structural_entailment(arg, assumptions) for arg in condition.args)
        if any(value is True for value in values):
            return True
        if all(value is False for value in values):
            return False
    return None


@lru_cache(maxsize=2048)
def _entails_cached(condition: sp.Expr, assumptions: sp.Expr) -> bool | None:
    structural = _structural_entailment(condition, assumptions)
    if structural is not None:
        return structural

    condition = bounded_simplify(condition)
    assumptions = bounded_simplify(assumptions)
    structural = _structural_entailment(condition, assumptions)
    if structural is not None:
        return structural
    return bounded_assumption_entails(condition, assumptions)


def clear_entailment_cache() -> None:
    """Clear cached bounded implication decisions."""

    _entails_cached.cache_clear()


def entailment_cache_info():
    """Return cache statistics for bounded implication decisions."""

    return _entails_cached.cache_info()


def entails(condition: sp.Expr, assumptions: sp.Expr | bool = sp.S.true) -> bool | None:
    """Return whether explicit assumptions entail ``condition``.

    Cheap structural implications are handled first; expensive SymPy
    assumptions/SAT fallbacks are bounded by a process-local LRU cache.
    """

    return _entails_cached(sp.sympify(condition), sp.sympify(assumptions))


def decide(
    predicate: sp.Expr,
    assumptions: sp.Expr | bool = sp.S.true,
    *,
    knowledge: PropertyKnowledge = PropertyKnowledge.EXACT,
    provenance: tuple[PropertyProvenance, ...] = (),
    reasons: tuple[str, ...] = (),
) -> PropertyDecision:
    return PropertyDecision(
        predicate,
        entails(predicate, assumptions),
        sp.sympify(assumptions),
        knowledge,
        provenance,
        reasons,
    )


def require_decision(
    decision: PropertyDecision,
    *,
    operation: str | None = None,
    allow_unknown: bool = False,
) -> PropertyDecision:
    if decision.verdict is False or (decision.verdict is None and not allow_unknown):
        raise PropertyEnforcementError(decision, operation=operation)
    return decision


def applicable_rule(rules, assumptions: sp.Expr | bool = sp.S.true) -> PropertyRule | None:
    candidates = sorted(rules, key=lambda rule: rule.priority, reverse=True)
    return next((rule for rule in candidates if entails(rule.condition, assumptions) is True), None)
