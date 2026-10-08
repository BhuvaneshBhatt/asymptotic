"""Certified Stokes geometry and beyond-all-orders exponential corrections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .complex_domain import ComplexBranchMetadata, ComplexSector
from .coverage import CoverageCertificate
from .proof_obligations import ObligationKind, ProofObligation
from .sectorial_transseries import (
    SectorialDominance,
    compare_sectorial_exponentials,
)


class TransitionKind(Enum):
    STOKES = "stokes"
    ANTI_STOKES = "anti_stokes"


@dataclass(frozen=True)
class StokesSurface:
    angle: sp.Expr
    kind: TransitionKind
    phase_difference: sp.Expr
    certified: bool = True


@dataclass(frozen=True)
class DominanceSector:
    sector: ComplexSector
    relation: SectorialDominance
    certified: bool


@dataclass(frozen=True)
class SectorConnection:
    left_index: int
    right_index: int
    boundary: StokesSurface
    multiplier: sp.Expr | None
    statement: str
    certified: bool


@dataclass(frozen=True)
class DominanceSectorAtlas:
    left_scale: sp.Expr
    right_scale: sp.Expr
    variable: sp.Symbol
    sectors: tuple[DominanceSector, ...]
    surfaces: tuple[StokesSurface, ...]
    connections: tuple[SectorConnection, ...]
    coverage: CoverageCertificate
    obligations: tuple[ProofObligation, ...] = ()


@dataclass(frozen=True)
class ExponentialCorrection:
    coefficient: sp.Expr
    scale: sp.Expr
    algebraic_variable: sp.Symbol
    sector: ComplexSector
    beyond_all_orders: bool
    certified: bool
    statement: str


@dataclass(frozen=True)
class StokesAwareExpansion:
    algebraic_part: sp.Expr
    corrections: tuple[ExponentialCorrection, ...]
    sector: ComplexSector
    branch: ComplexBranchMetadata
    certified: bool
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def expression(self) -> sp.Expr:
        return sp.simplify(
            self.algebraic_part
            + sp.Add(*(c.coefficient * c.scale for c in self.corrections))
        )


def _inverse_power(expr: sp.Expr, variable: sp.Symbol):
    expr = sp.expand_power_base(sp.sympify(expr), force=False)
    exponent = expr.as_powers_dict().get(variable, sp.S.Zero)
    if exponent.is_Integer is not True or exponent >= 0:
        return None
    coefficient = sp.simplify(expr / variable**exponent)
    if variable in coefficient.free_symbols or coefficient.is_zero is not False:
        return None
    return coefficient, int(-exponent)


def _principal_arg(expr: sp.Expr) -> sp.Expr | None:
    if expr.free_symbols:
        return None
    return sp.arg(expr)


def derive_stokes_surfaces(
    left_scale: sp.Expr, right_scale: sp.Expr, variable: sp.Symbol
) -> tuple[StokesSurface, ...]:
    """Derive exact rays for exp(phi)/exp(psi) when phi-psi=c/z**n.

    Anti-Stokes rays are equal-magnitude rays Re(phi-psi)=0.  Stokes rays are
    equal-phase rays Im(phi-psi)=0.  Both names are explicit here because the
    literature sometimes interchanges the terminology.
    """

    def exponent_of(scale):
        if scale.func is sp.exp:
            return scale.args[0]
        if scale == 1:
            return sp.S.Zero
        return None

    left_exponent = exponent_of(left_scale)
    right_exponent = exponent_of(right_scale)
    if left_exponent is None or right_exponent is None:
        return ()
    delta = sp.simplify(left_exponent - right_exponent)
    data = _inverse_power(delta, variable)
    if data is None:
        return ()
    coefficient, n = data
    alpha = _principal_arg(coefficient)
    if alpha is None:
        return ()
    surfaces = []
    for k in range(2 * n):
        anti = sp.simplify((alpha - sp.pi / 2 - k * sp.pi) / n)
        surfaces.append(StokesSurface(anti, TransitionKind.ANTI_STOKES, delta))
    for k in range(2 * n):
        stokes = sp.simplify((alpha - k * sp.pi) / n)
        surfaces.append(StokesSurface(stokes, TransitionKind.STOKES, delta))
    # exact angular deduplication within each kind
    unique = {}
    for surface in surfaces:
        key = (surface.kind, sp.simplify(sp.exp(sp.I * surface.angle)))
        unique[key] = surface
    return tuple(unique.values())


def dominance_sector_atlas(
    left_scale: sp.Expr, right_scale: sp.Expr, variable: sp.Symbol
) -> DominanceSectorAtlas:
    surfaces = derive_stokes_surfaces(left_scale, right_scale, variable)
    anti = [s for s in surfaces if s.kind is TransitionKind.ANTI_STOKES]
    if not anti:
        obligation = ProofObligation(
            ObligationKind.PROJECTIVE_COVERAGE,
            "exact Stokes atlas requires phase difference c/z**n",
            provider="dominance_sector_atlas",
            expression=sp.sstr(left_scale / right_scale),
        )
        return DominanceSectorAtlas(
            left_scale,
            right_scale,
            variable,
            (),
            surfaces,
            (),
            CoverageCertificate.unknown("stokes_atlas", obligation.statement),
            (obligation,),
        )
    # Sort numerically only after exact rays have been derived.  Centers/openings
    # remain exact SymPy expressions.
    ordered = sorted(anti, key=lambda s: float(sp.N(sp.arg(sp.exp(sp.I * s.angle)))))
    angles = [sp.arg(sp.exp(sp.I * s.angle)) for s in ordered]
    sectors = []
    for i, a in enumerate(angles):
        b = angles[(i + 1) % len(angles)]
        if i == len(angles) - 1:
            b = sp.simplify(b + 2 * sp.pi)
        opening = sp.simplify(b - a)
        center = sp.simplify(a + opening / 2)
        sector = ComplexSector(
            center, opening, excluded_rays=(a, b), label=f"dominance-{i}"
        )
        comparison = compare_sectorial_exponentials(
            left_scale, right_scale, variable, sector=sector
        )
        sectors.append(
            DominanceSector(sector, comparison.relation, comparison.certified)
        )
    complete = bool(sectors) and all(s.certified for s in sectors)
    coverage = (
        CoverageCertificate.complete(
            "stokes_atlas",
            "anti-Stokes rays partition the punctured angular circle",
            tuple(s.sector for s in sectors),
        )
        if complete
        else CoverageCertificate.unknown(
            "stokes_atlas",
            "one or more dominance sectors were not certified",
            tuple(s.sector for s in sectors),
        )
    )
    connections = []
    for i, boundary in enumerate(ordered):
        connections.append(
            SectorConnection(
                (i - 1) % len(sectors),
                i,
                boundary,
                None,
                "adjacent sector representations meet here; Stokes multiplier is unspecified",
                True,
            )
        )
    return DominanceSectorAtlas(
        left_scale,
        right_scale,
        variable,
        tuple(sectors),
        surfaces,
        tuple(connections),
        coverage,
    )


def certify_beyond_all_orders(
    scale: sp.Expr,
    variable: sp.Symbol,
    *,
    sector: ComplexSector,
    coefficient: sp.Expr = sp.S.One,
) -> ExponentialCorrection:
    """Certify exp(phi)=o(z**N) for every fixed N from uniform exponential decay."""
    comparison = compare_sectorial_exponentials(
        scale, sp.S.One, variable, sector=sector
    )
    certified = (
        comparison.certified and comparison.relation is SectorialDominance.SMALLER
    )
    statement = (
        "uniform exponential decay implies o(z**N) for every fixed algebraic order N"
        if certified
        else "beyond-all-orders decay was not certified on the requested sector"
    )
    return ExponentialCorrection(
        sp.sympify(coefficient),
        sp.sympify(scale),
        variable,
        sector,
        certified,
        certified,
        statement,
    )


def stokes_aware_expansion(
    algebraic_part: sp.Expr,
    corrections: tuple[tuple[sp.Expr, sp.Expr], ...],
    variable: sp.Symbol,
    *,
    sector: ComplexSector,
    branch: ComplexBranchMetadata | None = None,
) -> StokesAwareExpansion:
    branch = branch or ComplexBranchMetadata()
    certified_corrections = tuple(
        certify_beyond_all_orders(
            scale, variable, sector=sector, coefficient=coefficient
        )
        for coefficient, scale in corrections
    )
    obligations = ()
    if not all(c.certified for c in certified_corrections):
        obligations = (
            ProofObligation(
                ObligationKind.ORDER_RELATION,
                "at least one exponential correction is not uniformly beyond all algebraic orders on this sector",
                provider="stokes_aware_expansion",
            ),
        )
    return StokesAwareExpansion(
        sp.sympify(algebraic_part),
        certified_corrections,
        sector,
        branch,
        not obligations,
        obligations,
    )
