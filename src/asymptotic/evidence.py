"""Common evidence model for symbolic and numerical asymptotic claims."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceStatus(Enum):
    """Strength of the evidence supporting an asymptotic claim."""

    PROVED = "proved"
    CONDITIONALLY_PROVED = "conditionally_proved"
    NUMERICALLY_SUPPORTED = "numerically_supported"
    FORMAL = "formal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Evidence:
    """Auditable evidence for a public asymptotic result or validation check."""

    status: EvidenceStatus
    statement: str
    assumptions: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()
    details: tuple[tuple[str, object], ...] = ()

    @property
    def proved(self) -> bool:
        """Whether the claim is proved without unresolved obligations."""
        return self.status is EvidenceStatus.PROVED and not self.obligations

    @classmethod
    def combine(cls, *items: Evidence, statement: str) -> Evidence:
        """Combine independent evidence without overstating the weakest component."""
        if not items:
            return cls(EvidenceStatus.UNKNOWN, statement, obligations=("no evidence",))
        rank = {
            EvidenceStatus.PROVED: 4,
            EvidenceStatus.CONDITIONALLY_PROVED: 3,
            EvidenceStatus.NUMERICALLY_SUPPORTED: 2,
            EvidenceStatus.FORMAL: 1,
            EvidenceStatus.UNKNOWN: 0,
        }
        status = min((item.status for item in items), key=rank.__getitem__)
        assumptions = tuple(
            dict.fromkeys(x for item in items for x in item.assumptions)
        )
        obligations = tuple(
            dict.fromkeys(x for item in items for x in item.obligations)
        )
        if obligations and status is EvidenceStatus.PROVED:
            status = EvidenceStatus.CONDITIONALLY_PROVED
        return cls(status, statement, assumptions, obligations)
