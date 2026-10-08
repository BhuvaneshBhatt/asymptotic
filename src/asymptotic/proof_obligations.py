"""Structured unresolved proof obligations for conservative declines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ObligationKind(Enum):
    DOMAIN_ACCUMULATION = "domain_accumulation"
    PARAMETER_ORDER = "parameter_order"
    ORDER_RELATION = "order_relation"
    SYMBOLIC_BUDGET = "symbolic_budget"
    PROJECTIVE_COVERAGE = "projective_coverage"
    BRANCH_COVERAGE = "branch_coverage"
    COMPLEX_BRANCH = "complex_branch"
    THEOREM_PREREQUISITE = "theorem_prerequisite"


@dataclass(frozen=True)
class ProofObligation:
    kind: ObligationKind
    statement: str
    provider: str | None = None
    expression: str | None = None
