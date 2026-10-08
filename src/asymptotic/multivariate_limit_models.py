"""Result models shared by multivariate limit theorem providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .chart_models import CoordinateChart
from .coverage import CoverageCertificate


class AdvancedLimitStatus(Enum):
    CERTIFIED = "certified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LocalGerm:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    limit: sp.Expr | None
    domain: sp.Expr
    status: AdvancedLimitStatus
    provider: str
    statement: str

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class NewtonFanLimitResult:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    value: sp.Expr | None
    status: AdvancedLimitStatus
    weights: tuple[tuple[int, ...], ...]
    provider: str
    statement: str

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class ClusterSetResult:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    cluster_set: sp.Set | None
    liminf: sp.Expr | None
    limsup: sp.Expr | None
    status: AdvancedLimitStatus
    provider: str
    statement: str
    coverage: CoverageCertificate | None = None

    @property
    def certified(self) -> bool:
        return (
            self.status is AdvancedLimitStatus.CERTIFIED
            and self.coverage is not None
            and self.coverage.certified
        )

    @property
    def limit_semantics(self):
        """Return the unified logical limit conclusion of this cluster set."""
        from .cluster_semantics import cluster_limit_semantics

        return cluster_limit_semantics(self)


@dataclass(frozen=True)
class ExtremalBoundResult:
    """A liminf/limsup value together with its originating proof status."""

    value: sp.Expr | None
    status: AdvancedLimitStatus
    provider: str
    statement: str
    cluster_result: ClusterSetResult

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class RelativeLimitResult:
    """A simultaneous limit explicitly relative to a local approach domain."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    domain: sp.Expr
    value: sp.Expr | None
    status: AdvancedLimitStatus
    provider: str
    statement: str

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class LocalImageStratum:
    """One semialgebraic image stratum that accumulates at the germ target."""

    constraint: sp.Expr
    accumulating: bool
    provider: str


@dataclass(frozen=True)
class VectorLocalGerm:
    """Joint finite germ map with certified constraints on its local image."""

    expressions: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    limits: tuple[sp.Expr, ...] | None
    domain: sp.Expr
    status: AdvancedLimitStatus
    provider: str
    statement: str
    image_variables: tuple[sp.Symbol, ...] = ()
    image_constraint: sp.Expr = sp.S.true
    image_provider: str = "none"
    image_strata: tuple[LocalImageStratum, ...] = ()
    local_radius: sp.Expr | None = None

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class VectorLimitResult:
    """First-class certified limit of a finite-dimensional real map."""

    expressions: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    value: tuple[sp.Expr, ...] | None
    domain: sp.Expr
    status: AdvancedLimitStatus
    provider: str
    statement: str
    image_germ: VectorLocalGerm | None = None

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class VectorClusterSetResult:
    """Certified joint cluster set of a finite-dimensional real map."""

    expressions: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    cluster_set: sp.Set | None
    status: AdvancedLimitStatus
    provider: str
    statement: str
    image_germ: VectorLocalGerm | None = None
    coverage: CoverageCertificate | None = None

    @property
    def certified(self) -> bool:
        return (
            self.status is AdvancedLimitStatus.CERTIFIED
            and self.coverage is not None
            and self.coverage.certified
        )


@dataclass(frozen=True)
class ComplexMultivariateLimitResult:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    value: sp.Expr | None
    status: AdvancedLimitStatus
    provider: str
    statement: str

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class ProjectiveClusterAtlasResult:
    """Certified cluster geometry assembled from several projective ends."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    ends: tuple[tuple[sp.Expr, ...], ...]
    cluster_set: sp.Set | None
    status: AdvancedLimitStatus
    provider: str
    statement: str
    charts: tuple[CoordinateChart, ...] = ()
    coverage: CoverageCertificate | None = None

    @property
    def certified(self) -> bool:
        return (
            self.status is AdvancedLimitStatus.CERTIFIED
            and self.coverage is not None
            and self.coverage.certified
        )


@dataclass(frozen=True)
class ComplexMeromorphicGeometryResult:
    """Local zero/pole valuation data for a complex meromorphic germ."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    numerator_order: int | None
    denominator_order: int | None
    classification: str
    value: sp.Expr | None
    status: AdvancedLimitStatus
    provider: str

    @property
    def certified(self) -> bool:
        return self.status is AdvancedLimitStatus.CERTIFIED


@dataclass(frozen=True)
class JointClusterStratum:
    """One certified limiting-image stratum in a joint cluster computation."""

    weight: tuple[int, ...] | None
    constraint: sp.Expr
    cluster_set: sp.Set
    provider: str
    source_domain: sp.Expr = sp.S.true
    structured_geometry: object | None = None
    local_stratum: object | None = None

    @property
    def stratum_kind(self):
        return "cluster_image"

    @property
    def local_constraint(self):
        return self.constraint

    @property
    def intrinsic_dimension(self):
        return getattr(self.structured_geometry, "dimension", None)

    @property
    def valuation_data(self):
        return self.weight

    @property
    def child_strata(self):
        return (self.local_stratum,) if self.local_stratum is not None else ()

    @property
    def stratum_certified(self):
        return True


@dataclass(frozen=True)
class JointClusterGeometryResult:
    """Correlated joint cluster geometry assembled from local limiting strata."""

    expressions: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    cluster_set: sp.Set | None
    strata: tuple[JointClusterStratum, ...]
    status: AdvancedLimitStatus
    provider: str
    statement: str
    newton_fan: object | None = None
    coverage: CoverageCertificate | None = None

    @property
    def certified(self) -> bool:
        return (
            self.status is AdvancedLimitStatus.CERTIFIED
            and self.coverage is not None
            and self.coverage.certified
        )
