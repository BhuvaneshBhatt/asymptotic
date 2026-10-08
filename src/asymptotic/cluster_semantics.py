"""Unified logical semantics for certified multivariate cluster sets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp


class ClusterLimitStatus(Enum):
    """Logical limit conclusion implied by a certified cluster set."""

    LIMIT = "limit"
    DOES_NOT_EXIST = "does_not_exist"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClusterLimitSemantics:
    """A limit conclusion derived only from certified cluster-set information."""

    status: ClusterLimitStatus
    value: sp.Expr | None
    cluster_set: sp.Set | None
    provider: str
    statement: str

    @property
    def certified(self) -> bool:
        return self.status is not ClusterLimitStatus.UNKNOWN


def _singleton_value(cluster_set):
    if isinstance(cluster_set, sp.FiniteSet) and len(cluster_set) == 1:
        return next(iter(cluster_set))
    if isinstance(cluster_set, sp.Interval):
        equal = sp.simplify(cluster_set.start - cluster_set.end)
        if equal == 0:
            return cluster_set.start
    return None


def cluster_limit_semantics(cluster_result) -> ClusterLimitSemantics:
    """Derive existence/DNE from a *certified complete* cluster-set result.

    A singleton cluster set certifies the limit. Any certified non-singleton
    cluster set certifies DNE. Partial or uncertified cluster sets remain
    UNKNOWN regardless of how many sampled values they contain.
    """
    certified = bool(getattr(cluster_result, "certified", False))
    cluster_set = getattr(cluster_result, "cluster_set", None)
    provider = getattr(cluster_result, "provider", "cluster_set")
    if not certified or not isinstance(cluster_set, sp.Set):
        return ClusterLimitSemantics(
            ClusterLimitStatus.UNKNOWN,
            None,
            cluster_set,
            provider,
            "cluster set is not certified complete",
        )

    value = _singleton_value(cluster_set)
    if value is not None:
        return ClusterLimitSemantics(
            ClusterLimitStatus.LIMIT,
            sp.simplify(value),
            cluster_set,
            provider,
            "certified complete cluster set is a singleton",
        )

    empty = cluster_set is sp.S.EmptySet
    if empty:
        return ClusterLimitSemantics(
            ClusterLimitStatus.UNKNOWN,
            None,
            cluster_set,
            provider,
            "empty cluster set does not certify a punctured-neighbourhood limit",
        )

    return ClusterLimitSemantics(
        ClusterLimitStatus.DOES_NOT_EXIST,
        None,
        cluster_set,
        provider,
        "certified complete cluster set contains more than one cluster value",
    )


def cluster_semantics_to_limit_result(cluster_result):
    """Convert unified cluster semantics to the core limit result protocol."""
    from .limit_models import LimitEvidence, LimitStatus, SimultaneousLimitResult

    semantics = cluster_limit_semantics(cluster_result)
    status = {
        ClusterLimitStatus.LIMIT: LimitStatus.PROVED,
        ClusterLimitStatus.DOES_NOT_EXIST: LimitStatus.DOES_NOT_EXIST,
        ClusterLimitStatus.UNKNOWN: LimitStatus.UNKNOWN,
    }[semantics.status]
    evidence = (
        LimitEvidence(
            "cluster_set_semantics",
            semantics.statement,
            value=semantics.value,
        ),
    )
    return SimultaneousLimitResult(
        cluster_result.expression,
        cluster_result.variables,
        cluster_result.target,
        status,
        semantics.value,
        evidence,
        cluster_result.variables,
        sp.S.true,
    )


__all__ = [
    "ClusterLimitSemantics",
    "ClusterLimitStatus",
    "cluster_limit_semantics",
    "cluster_semantics_to_limit_result",
]
