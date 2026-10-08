"""Proof-aware real limits and their result models."""

from ._limit_engine import limit
from .limit_models import (
    LimitEvidence,
    LimitStatus,
    SimultaneousLimitDoesNotExist,
    SimultaneousLimitResult,
)
from .limit_primitives import normalize_limit_target

__all__ = [
    "LimitEvidence",
    "LimitStatus",
    "SimultaneousLimitDoesNotExist",
    "SimultaneousLimitResult",
    "limit",
    "normalize_limit_target",
]
