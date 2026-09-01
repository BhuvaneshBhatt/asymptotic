"""Sector and branch metadata for complex asymptotic germs.

The objects here are deliberately descriptive rather than analytic proofs: they
record the sector on which an expansion is intended and the branch choices used
to construct it.  Certification code can therefore reject incompatible
continuations instead of silently treating a ray expansion as globally valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ComplexSector:
    """An open angular sector for a complex asymptotic germ.

    ``center_angle`` and ``opening`` are measured in radians.  ``opening`` must
    be positive and at most ``2*pi``.  Boundary rays are excluded by default;
    callers may explicitly record them in ``excluded_rays`` when they coincide
    with branch or Stokes cuts.
    """

    center_angle: sp.Expr = sp.S.Zero
    opening: sp.Expr = sp.pi
    excluded_rays: tuple[sp.Expr, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        center = sp.sympify(self.center_angle)
        opening = sp.sympify(self.opening)
        object.__setattr__(self, "center_angle", center)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "excluded_rays", tuple(sp.sympify(r) for r in self.excluded_rays))
        if opening.is_positive is not True:
            raise ValueError("sector opening must be provably positive")
        if sp.simplify(opening - 2 * sp.pi).is_positive is True:
            raise ValueError("sector opening cannot exceed 2*pi")

    @property
    def lower_angle(self) -> sp.Expr:
        return sp.simplify(self.center_angle - self.opening / 2)

    @property
    def upper_angle(self) -> sp.Expr:
        return sp.simplify(self.center_angle + self.opening / 2)

    def contains_angle(self, angle: sp.Expr) -> bool | None:
        """Decide containment when the angular inequalities are symbolic-decidable."""

        angle = sp.sympify(angle)
        delta = sp.arg(sp.exp(sp.I * (angle - self.center_angle)))
        test = sp.simplify(sp.Abs(delta) - self.opening / 2)
        if test.is_negative is True:
            return True
        if test.is_nonnegative is True:
            return False
        return None


@dataclass(frozen=True)
class ComplexBranchMetadata:
    """Branch choices and continuation provenance for one complex germ."""

    logarithm_branch: int = 0
    power_branch: int = 0
    branch_cuts: tuple[sp.Expr, ...] = ()
    stokes_rays: tuple[sp.Expr, ...] = ()
    continuation: str | None = None
    principal: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.logarithm_branch, int) or not isinstance(self.power_branch, int):
            raise TypeError("branch indices must be integers")
        object.__setattr__(self, "branch_cuts", tuple(sp.sympify(r) for r in self.branch_cuts))
        object.__setattr__(self, "stokes_rays", tuple(sp.sympify(r) for r in self.stokes_rays))


def complex_germ_metadata(
    *,
    sector: ComplexSector | None = None,
    branch: ComplexBranchMetadata | None = None,
) -> dict[str, object]:
    """Return canonical metadata entries for a complex asymptotic germ."""

    out: dict[str, object] = {}
    if sector is not None:
        out["complex_sector"] = sector
    if branch is not None:
        out["complex_branch"] = branch
    return out


def merge_complex_germ_metadata(
    metadata: dict[str, object] | None,
    *,
    sector: ComplexSector | None = None,
    branch: ComplexBranchMetadata | None = None,
) -> dict[str, object]:
    """Merge explicit complex metadata, rejecting incompatible re-annotation."""

    out = dict(metadata or {})
    if sector is not None:
        existing = out.get("complex_sector")
        if existing is not None and existing != sector:
            raise ValueError("conflicting complex-sector metadata")
        out["complex_sector"] = sector
    if branch is not None:
        existing = out.get("complex_branch")
        if existing is not None and existing != branch:
            raise ValueError("conflicting complex-branch metadata")
        out["complex_branch"] = branch
    return out
