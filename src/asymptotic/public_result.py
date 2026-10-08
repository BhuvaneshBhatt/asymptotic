"""Uniform, lightweight presentation of heterogeneous public result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ResultExplanation:
    """Stable user-facing summary of a public asymptotic result."""

    status: str | None
    method: str | None
    certified: bool | None
    reason: str | None
    limitations: tuple[str, ...] = ()

    def __str__(self) -> str:
        parts = []
        if self.status:
            parts.append(f"status={self.status}")
        if self.method:
            parts.append(f"method={self.method}")
        if self.certified is not None:
            parts.append(f"certified={self.certified}")
        if self.reason:
            parts.append(self.reason)
        parts.extend(self.limitations)
        return "; ".join(parts) if parts else "No additional result metadata."


@runtime_checkable
class PublicResult(Protocol):
    """Protocol for results that provide their own user-facing explanation."""

    def explain(self) -> ResultExplanation:
        """Return stable presentation metadata without exposing internal machinery."""


def _text(value: object | None) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def explain(result: object) -> ResultExplanation:
    """Explain any public result through one stable presentation boundary.

    Result types may implement ``PublicResult.explain`` directly. For other
    public records, common status, method, certification, and limitation fields
    are summarized without coupling theorem code to presentation logic.
    """
    if isinstance(result, PublicResult):
        return result.explain()

    status = _text(getattr(result, "status", None))
    method = _text(getattr(result, "method", None))
    reason = _text(getattr(result, "reason", None))
    if reason is None:
        reason = _text(getattr(result, "limitation", None))

    certified_value = getattr(result, "certified", None)
    certified = certified_value if isinstance(certified_value, bool) else None
    if certified is None:
        certificate = getattr(result, "certificate", None)
        value = getattr(certificate, "certified", None)
        certified = value if isinstance(value, bool) else None

    limitations = []
    for name in ("limitations", "warnings"):
        value = getattr(result, name, None)
        if isinstance(value, str):
            limitations.append(value)
        elif isinstance(value, (tuple, list)):
            limitations.extend(str(item) for item in value if item)

    return ResultExplanation(status, method, certified, reason, tuple(limitations))
