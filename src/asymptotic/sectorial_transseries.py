"""Certified finite-height complex sectorial logarithmico-exponential scales."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .complex_domain import ComplexBranchMetadata, ComplexSector
from .coverage import CoverageCertificate
from .proof_obligations import ObligationKind, ProofObligation


class SectorialDominance(Enum):
    """Magnitude relation valid throughout an open complex sector."""

    SMALLER = "smaller"
    SAME_ORDER = "same_order"
    LARGER = "larger"
    OSCILLATORY_COMPARABLE = "oscillatory_comparable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SectorialScaleComparison:
    relation: SectorialDominance
    sector: ComplexSector
    certified: bool
    statement: str
    coverage: CoverageCertificate
    obligations: tuple[ProofObligation, ...] = ()


@dataclass(frozen=True)
class SectorialTransseriesExpansion:
    """Exact/certified LE representation on one branch-safe complex sector."""

    expression: sp.Expr
    representation: sp.Expr
    variable: sp.Symbol
    sector: ComplexSector
    branch: ComplexBranchMetadata
    certified: bool
    method: str
    coverage: CoverageCertificate
    obligations: tuple[ProofObligation, ...] = ()


def _strict_real_sign(value: sp.Expr) -> int | None:
    value = sp.simplify(value)
    if value.is_positive is True:
        return 1
    if value.is_negative is True:
        return -1
    if value.is_zero is True:
        return 0
    return None


def _sector_cos_sign(
    sector: ComplexSector, harmonic: int, phase: sp.Expr
) -> int | None:
    """Certify a constant sign for cos(harmonic*theta + phase) on a sector.

    The proof is elementary: after mapping the sector center into a
    cosine lobe, the whole open angular interval must lie strictly inside that
    lobe.  Touching a zero ray is not certified.
    """
    if not phase.is_real:
        return None
    center = sp.simplify(harmonic * sector.center_angle + phase)
    half_width = sp.simplify(abs(harmonic) * sector.opening / 2)
    if half_width.is_nonnegative is not True:
        return None
    # Reduce exact numeric centers modulo 2*pi.  Symbolic centers decline.
    if center.free_symbols:
        return None
    reduced = sp.arg(sp.exp(sp.I * center))
    margin_positive = sp.simplify(sp.pi / 2 - sp.Abs(reduced) - half_width)
    if margin_positive.is_nonnegative is True:
        return 1
    distance_to_pi = sp.Abs(sp.arg(sp.exp(sp.I * (reduced - sp.pi))))
    margin_negative = sp.simplify(sp.pi / 2 - distance_to_pi - half_width)
    if margin_negative.is_nonnegative is True:
        return -1
    return None


def _inverse_power_phase(expr: sp.Expr, variable: sp.Symbol):
    """Return (coefficient, n) for c/z**n, n>0, when exact."""
    expr = sp.expand_power_base(sp.sympify(expr), force=False)
    powers = expr.as_powers_dict()
    exponent = powers.get(variable, sp.S.Zero)
    if exponent.is_Integer is not True or exponent >= 0:
        return None
    coefficient = sp.simplify(expr / variable**exponent)
    if variable in coefficient.free_symbols:
        return None
    return coefficient, int(-exponent)


def _real_part_sign_on_sector(
    expr: sp.Expr, variable: sp.Symbol, sector: ComplexSector
):
    """Certify the sign of Re(expr) as z->0 inside a sector for c/z**n."""
    data = _inverse_power_phase(expr, variable)
    if data is None:
        return None
    coefficient, n = data
    if coefficient.is_real is True:
        coefficient_sign = _strict_real_sign(coefficient)
        if coefficient_sign is None:
            return None
        angular_sign = _sector_cos_sign(sector, n, sp.S.Zero)
        return None if angular_sign is None else coefficient_sign * angular_sign
    magnitude = sp.Abs(coefficient)
    if magnitude.is_positive is not True:
        return None
    phase = sp.arg(coefficient)
    angular_sign = _sector_cos_sign(sector, n, -phase)
    return angular_sign


def compare_sectorial_exponentials(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    *,
    sector: ComplexSector,
) -> SectorialScaleComparison:
    """Compare ``exp(phi)`` and ``exp(psi)`` uniformly on an open sector.

    The sectorial transseries engine certifies inverse-power exponential phases.  Other finite-height
    LE expressions decline rather than extrapolating a ray calculation.
    """
    left = sp.sympify(left)
    right = sp.sympify(right)

    def exponent_of(scale: sp.Expr):
        if scale.func is sp.exp:
            return scale.args[0]
        if scale == 1:
            return sp.S.Zero
        return None

    left_exponent = exponent_of(left)
    right_exponent = exponent_of(right)
    if left_exponent is None or right_exponent is None:
        obligation = ProofObligation(
            ObligationKind.THEOREM_PREREQUISITE,
            "sectorial dominance supports two exponential scales",
            provider="compare_sectorial_exponentials",
            expression=sp.sstr(left / right),
        )
        return SectorialScaleComparison(
            SectorialDominance.UNKNOWN,
            sector,
            False,
            obligation.statement,
            CoverageCertificate.unknown(
                "sectorial_exp_dominance", obligation.statement, (sector,)
            ),
            (obligation,),
        )
    delta = sp.simplify(left_exponent - right_exponent)
    sign = _real_part_sign_on_sector(delta, variable, sector)
    if sign in (-1, 1):
        relation = SectorialDominance.SMALLER if sign < 0 else SectorialDominance.LARGER
        statement = f"Re({sp.sstr(delta)}) has constant {'negative' if sign < 0 else 'positive'} sign on the sector"
        coverage = CoverageCertificate.complete(
            "sectorial_exp_dominance", statement, (sector,)
        )
        return SectorialScaleComparison(relation, sector, True, statement, coverage)
    obligation = ProofObligation(
        ObligationKind.ORDER_RELATION,
        "the sign of the real exponential phase difference is not uniform on the requested sector",
        provider="compare_sectorial_exponentials",
        expression=sp.sstr(delta),
    )
    return SectorialScaleComparison(
        SectorialDominance.UNKNOWN,
        sector,
        False,
        obligation.statement,
        CoverageCertificate.unknown(
            "sectorial_exp_dominance", obligation.statement, (sector,)
        ),
        (obligation,),
    )


def _branch_safe(sector: ComplexSector, branch: ComplexBranchMetadata) -> bool:
    for ray in branch.branch_cuts:
        if sector.contains_angle(ray) is not False:
            return False
    return True


def _branched_log(variable: sp.Symbol, branch: ComplexBranchMetadata) -> sp.Expr:
    return sp.log(variable) + 2 * sp.pi * sp.I * branch.logarithm_branch


def branch_aware_rewrite(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    sector: ComplexSector,
    branch: ComplexBranchMetadata,
) -> SectorialTransseriesExpansion:
    """Rewrite log/power expressions on an explicitly branch-safe sector.

    This is an exact representation, not a Taylor approximation.  Noninteger
    powers are represented through the selected logarithm branch.
    """
    expr = sp.sympify(expr)
    if not _branch_safe(sector, branch):
        obligation = ProofObligation(
            ObligationKind.COMPLEX_BRANCH,
            "requested sector intersects or cannot exclude a selected branch cut",
            provider="branch_aware_rewrite",
            expression=sp.sstr(expr),
        )
        return SectorialTransseriesExpansion(
            expr,
            expr,
            variable,
            sector,
            branch,
            False,
            "branch_prerequisite_unproved",
            CoverageCertificate.unknown(
                "sectorial_branch", obligation.statement, (sector,)
            ),
            (obligation,),
        )

    selected_log = _branched_log(variable, branch)

    def rewrite(node: sp.Expr) -> sp.Expr:
        if node.func is sp.log and node.args[0] == variable:
            return selected_log
        if node.is_Pow and node.base == variable and node.exp.is_integer is not True:
            return sp.exp(node.exp * selected_log)
        if not node.args:
            return node
        return node.func(*(rewrite(arg) for arg in node.args))

    representation = rewrite(expr)
    statement = (
        "selected log/power branch is analytic throughout the requested open sector"
    )
    return SectorialTransseriesExpansion(
        expr,
        representation,
        variable,
        sector,
        branch,
        True,
        "exact_branch_aware_logexp",
        CoverageCertificate.complete("sectorial_branch", statement, (sector,)),
    )


__all__ = [
    "SectorialDominance",
    "SectorialScaleComparison",
    "SectorialTransseriesExpansion",
    "branch_aware_rewrite",
    "compare_sectorial_exponentials",
]
