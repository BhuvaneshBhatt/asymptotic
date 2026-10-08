"""Shared result types for limit computations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .proof_obligations import ProofObligation


class LimitStatus(Enum):
    """Logical status of a simultaneous-limit computation."""

    PROVED = "proved"
    DOES_NOT_EXIST = "does_not_exist"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LimitEvidence:
    """One replayable piece of evidence used by a simultaneous-limit result."""

    method: str
    statement: str
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()
    value: sp.Expr | None = None


@dataclass(frozen=True)
class SimultaneousLimitResult:
    """Structured result for a simultaneous real Euclidean limit.

    ``value`` is populated only when the limit is proved.  A nonexistence
    result carries at least two conflicting path witnesses in ``evidence``.
    Unknown is an intentional result and never means that sampled paths agree.
    """

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    status: LimitStatus
    value: sp.Expr | None = None
    evidence: tuple[LimitEvidence, ...] = ()
    reduced_variables: tuple[sp.Symbol, ...] = ()
    domain: sp.Expr = sp.S.true
    conditional_value: sp.Expr | None = None
    condition: sp.Expr | None = None
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def is_conditional(self) -> bool:
        return self.condition is not None

    @property
    def mathematical_value(self):
        """Return the ordinary or conditional mathematical value, if known."""
        if self.status is LimitStatus.PROVED:
            return self.value
        if self.conditional_value is not None and self.condition is not None:
            from .conditional import conditional_expression

            return conditional_expression(self.conditional_value, self.condition)
        return None

    @property
    def exists(self) -> bool | None:
        """Return True, False, or None according to the proved logical status."""
        if self.status is LimitStatus.PROVED:
            return True
        if self.status is LimitStatus.DOES_NOT_EXIST:
            return False
        return None


class SimultaneousLimitDoesNotExist(ValueError):
    """Raised by value-mode when exact path witnesses prove nonexistence."""
