"""Declarative theorem selection shared by asymptotic subsystems."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import sympy as sp


class CertificationStrength(IntEnum):
    FORMAL = 0
    CONDITIONAL = 1
    CERTIFIED = 2
    EXACT = 3


@dataclass(frozen=True)
class Theorem:
    name: str
    family: str
    applicability: Callable[[Any], bool | None]
    hypotheses: Callable[[Any], tuple[sp.Expr, ...]]
    constructor: Callable[[Any], Any]
    certification: CertificationStrength
    priority: int = 0


@dataclass(frozen=True)
class TheoremDecision:
    theorem: Theorem
    applicable: bool | None
    hypotheses: tuple[sp.Expr, ...]
    proved: tuple[bool | None, ...]


class TheoremRegistry:
    """Ordered theorem collection; selection never executes constructors speculatively."""

    def __init__(self, theorems: Iterable[Theorem] = ()) -> None:
        self._theorems = list(theorems)

    def register(self, theorem: Theorem) -> None:
        if any(t.name == theorem.name for t in self._theorems):
            raise ValueError(f"duplicate theorem name: {theorem.name}")
        self._theorems.append(theorem)

    def decisions(
        self, problem: Any, *, context: Any = None
    ) -> tuple[TheoremDecision, ...]:
        out = []
        for theorem in self._theorems:
            key = (
                theorem.family,
                theorem.name,
                sp.srepr(problem) if isinstance(problem, sp.Basic) else repr(problem),
            )
            applicable = (
                context.cached_applicability(
                    key, lambda t=theorem: t.applicability(problem)
                )
                if context is not None
                else theorem.applicability(problem)
            )
            hypotheses = theorem.hypotheses(problem) if applicable is not False else ()
            proved = tuple(
                context.entails(h) if context is not None else None for h in hypotheses
            )
            out.append(TheoremDecision(theorem, applicable, hypotheses, proved))
        return tuple(out)

    def candidates(
        self, problem: Any, *, context: Any = None
    ) -> tuple[TheoremDecision, ...]:
        """Return eligible theorems in deterministic route order.

        ``priority`` preserves dispatcher semantics; certification strength is a
        tie-breaker rather than a reason to bypass an earlier exact structural
        route. Constructors are not executed here.
        """
        eligible = [
            d
            for d in self.decisions(problem, context=context)
            if d.applicable is True and all(v is True for v in d.proved)
        ]
        return tuple(
            sorted(
                eligible,
                key=lambda d: (
                    d.theorem.priority,
                    -int(d.theorem.certification),
                    d.theorem.name,
                ),
            )
        )

    def select(self, problem: Any, *, context: Any = None) -> TheoremDecision | None:
        candidates = self.candidates(problem, context=context)
        return candidates[0] if candidates else None

    def family(self, family: str) -> tuple[Theorem, ...]:
        return tuple(t for t in self._theorems if t.family == family)


THEOREMS = TheoremRegistry()
