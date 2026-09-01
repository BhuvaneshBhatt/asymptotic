"""Reference asymptotic problems used as a durable capability corpus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CapabilityStatus(Enum):
    """Expected strength of support for one reference problem."""

    CERTIFIED = "certified"
    FORMAL = "formal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReferenceCase:
    """Named mathematical problem with an executable expected outcome."""

    name: str
    area: str
    status: CapabilityStatus
    check: Callable[[], bool]
