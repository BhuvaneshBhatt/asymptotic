"""Composable coverage/completeness certificates for local geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoverageStatus(Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CoverageObligation:
    kind: str
    region: object
    discharged: bool
    evidence: str


@dataclass(frozen=True)
class CoverageCertificate:
    status: CoverageStatus
    provider: str
    statement: str
    covered_regimes: tuple[object, ...] = ()
    missing_regimes: tuple[object, ...] = ()
    obligations: tuple[CoverageObligation, ...] = ()

    @property
    def certified(self):
        return (
            self.status is CoverageStatus.COMPLETE
            and not self.missing_regimes
            and all(o.discharged for o in self.obligations)
        )

    @classmethod
    def complete(cls, provider, statement, covered=(), obligations=()):
        return cls(
            CoverageStatus.COMPLETE,
            provider,
            statement,
            tuple(covered),
            (),
            tuple(obligations),
        )

    @classmethod
    def partial(cls, provider, statement, covered=(), missing=(), obligations=()):
        return cls(
            CoverageStatus.PARTIAL,
            provider,
            statement,
            tuple(covered),
            tuple(missing),
            tuple(obligations),
        )

    @classmethod
    def unknown(cls, provider, statement, missing=()):
        return cls(CoverageStatus.UNKNOWN, provider, statement, (), tuple(missing), ())

    def combine(self, *others, provider="combined_coverage", statement=None):
        certs = (self,) + others
        covered = tuple(x for c in certs for x in c.covered_regimes)
        missing = tuple(x for c in certs for x in c.missing_regimes)
        obligations = tuple(x for c in certs for x in c.obligations)
        if all(c.certified for c in certs):
            status = CoverageStatus.COMPLETE
        elif any(c.status is CoverageStatus.PARTIAL for c in certs):
            status = CoverageStatus.PARTIAL
        else:
            status = CoverageStatus.UNKNOWN
        return CoverageCertificate(
            status,
            provider,
            statement or "combined coverage obligations",
            covered,
            missing,
            obligations,
        )


def coverage_certificate(cert):
    if isinstance(cert, CoverageCertificate):
        return cert
    if getattr(cert, "certified", False):
        return CoverageCertificate.complete(
            getattr(cert, "provider", "coverage"),
            getattr(cert, "statement", "complete coverage"),
            getattr(cert, "covered_regimes", ()),
        )
    return CoverageCertificate.partial(
        getattr(cert, "provider", "coverage"),
        getattr(cert, "statement", "incomplete coverage"),
        getattr(cert, "covered_regimes", ()),
        getattr(cert, "missing_regimes", ()),
    )


__all__ = [
    "CoverageCertificate",
    "CoverageObligation",
    "CoverageStatus",
    "coverage_certificate",
]
