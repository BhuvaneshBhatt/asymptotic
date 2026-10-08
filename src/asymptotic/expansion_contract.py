"""Dependency-neutral unified contract for certified asymptotic expansions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import sympy as sp

from .coverage import CoverageCertificate
from .proof_obligations import ProofObligation
from .uniform_remainder import UniformRemainder


@runtime_checkable
class AsymptoticExpansion(Protocol):
    """Minimal common behavioral contract for one expansion result."""

    expression: sp.Expr

    @property
    def certified(self) -> bool: ...


@dataclass(frozen=True)
class ExpansionCertificate:
    """Canonical certification view without erasing representation-specific data."""

    expression: sp.Expr
    representation: sp.Expr | None
    regime: object | None
    domain: sp.Expr
    remainder: UniformRemainder | None
    remainder_scale: sp.Expr | None
    coverage: CoverageCertificate | None
    obligations: tuple[ProofObligation, ...]
    evidence: tuple[str, ...]
    source_kind: str
    source: AsymptoticExpansion
    parameter_domain: object | None = None
    children: tuple[ExpansionCertificate, ...] = ()

    @property
    def certified(self) -> bool:
        return (
            self.source.certified
            and not self.obligations
            and (self.remainder is None or self.remainder.certified)
            and (self.coverage is None or self.coverage.certified)
            and all(child.certified for child in self.children)
        )


def _coverage(result):
    value = getattr(result, "coverage", None)
    if isinstance(value, CoverageCertificate):
        return value
    return None


def _representation(result):
    for name in ("approximation", "representation", "algebraic_part"):
        value = getattr(result, name, None)
        if value is not None:
            return sp.sympify(value)
    truncated = getattr(result, "truncated_expression", None)
    return sp.sympify(truncated()) if callable(truncated) else None


def _remainder(result):
    method = getattr(result, "uniform_remainder", None)
    if callable(method):
        return method()
    return None


def expansion_certificate(result: AsymptoticExpansion) -> ExpansionCertificate:
    """Return the canonical certification view for any supported expansion result."""
    if not isinstance(result, AsymptoticExpansion):
        raise TypeError("result does not satisfy the asymptotic expansion contract")
    expression = getattr(result, "expression", None)
    if expression is None:
        # Stokes-aware results are local representations rather than transformations
        # of a stored source expression; preserve their represented algebraic part.
        expression = getattr(result, "algebraic_part", None)
    if expression is None:
        raise TypeError("expansion result does not expose a represented expression")
    obligations = tuple(getattr(result, "obligations", ()))
    method = getattr(result, "method", None)
    provider = getattr(getattr(result, "remainder_certificate", None), "provider", None)
    evidence = tuple(x for x in (method, provider) if isinstance(x, str) and x)
    return ExpansionCertificate(
        sp.sympify(expression),
        _representation(result),
        getattr(result, "regime", None),
        sp.sympify(getattr(result, "domain", sp.S.true)),
        _remainder(result),
        (
            None
            if getattr(result, "remainder_scale", None) is None
            else sp.sympify(result.remainder_scale)
        ),
        _coverage(result),
        obligations,
        evidence,
        type(result).__name__,
        result,
        getattr(result, "parameter_domains", None),
        tuple(
            expansion_certificate(child)
            for child in getattr(result, "branches", ())
            if isinstance(child, AsymptoticExpansion)
        ),
    )


def expansion_is_certified(result: AsymptoticExpansion) -> bool:
    """Certification predicate with common remainder/coverage/obligation semantics."""
    return expansion_certificate(result).certified
