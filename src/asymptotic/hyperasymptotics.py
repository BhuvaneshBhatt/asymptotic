"""Generic terminant and bounded-level hyperasymptotic machinery."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ExponentialScale:
    """An adjacent exponential contribution and its saddle geometry."""

    singulant: sp.Expr
    amplitude: sp.Expr
    orientation: int
    stokes_curves: tuple[sp.Expr, ...]
    anti_stokes_curves: tuple[sp.Expr, ...]
    adjacent_saddle: str | None = None


@dataclass(frozen=True)
class TerminantLevel:
    """One exponentially improved level of a hyperasymptotic expansion."""

    level: int
    truncation_index: sp.Expr
    singulant: sp.Expr
    correction: sp.Expr
    multiplier: sp.Expr
    remainder_scale: sp.Expr
    hypotheses_verified: bool


@dataclass(frozen=True)
class HyperasymptoticExpansion:
    """Poincare prefix followed by a bounded sequence of terminant levels."""

    poincare_prefix: sp.Expr
    levels: tuple[TerminantLevel, ...]
    approximation: sp.Expr
    remainder_scale: sp.Expr
    certified: bool


def terminant(p: sp.Expr, z: sp.Expr) -> sp.Expr:
    """Principal terminant used by the generic Stokes engine."""
    p = sp.sympify(p)
    z = sp.sympify(z)
    return sp.gamma(p) * sp.uppergamma(1 - p, z) / (2 * sp.pi)


def rescaled_terminant(p: sp.Expr, z: sp.Expr) -> sp.Expr:
    """Exponentially rescaled principal terminant."""
    return sp.exp(z) * terminant(p, z)


def least_term_index(singulant: sp.Expr, coefficient_step: int = 1) -> sp.Expr:
    """Nearest admissible truncation index to the least term."""
    if coefficient_step < 1:
        raise ValueError("coefficient_step must be positive")
    return coefficient_step * sp.floor(sp.Abs(singulant) / coefficient_step)


def stokes_multiplier(scale: ExponentialScale, truncation_index: sp.Expr) -> sp.Expr:
    """Terminant Stokes multiplier with explicit orientation."""
    return scale.orientation * rescaled_terminant(truncation_index, scale.singulant)


def terminant_reexpand(
    coefficients: Sequence[sp.Expr],
    scale: ExponentialScale,
    *,
    reexpansion_terms: int,
    truncation_index: sp.Expr | None = None,
) -> TerminantLevel:
    """Re-expand one least-term remainder on an adjacent exponential scale."""
    if reexpansion_terms < 1:
        raise ValueError("reexpansion_terms must be positive")
    if not coefficients:
        raise ValueError("coefficients must not be empty")
    n = (
        least_term_index(scale.singulant)
        if truncation_index is None
        else truncation_index
    )
    correction = sp.Add(
        *(
            coefficients[j]
            * rescaled_terminant(n - j, scale.singulant)
            / scale.singulant**j
            for j in range(min(reexpansion_terms, len(coefficients)))
        )
    )
    multiplier = stokes_multiplier(scale, n)
    remainder = sp.Abs(scale.amplitude * sp.exp(-scale.singulant)) * sp.Abs(
        scale.singulant
    ) ** (-reexpansion_terms)
    verified = scale.orientation in (-1, 1)
    return TerminantLevel(
        1, n, scale.singulant, correction, multiplier, remainder, verified
    )


def hyperasymptotic_series(
    poincare_prefix: sp.Expr,
    coefficient_provider: Callable[[int, int], Sequence[sp.Expr]],
    scales: Sequence[ExponentialScale],
    *,
    levels: int = 1,
    reexpansion_terms: int = 2,
    max_levels: int = 3,
) -> HyperasymptoticExpansion:
    """Recursively re-expand exponentially small remainders to bounded depth."""
    if levels < 0 or levels > max_levels:
        raise ValueError(f"levels must lie between 0 and {max_levels}")
    if levels and not scales:
        raise ValueError("at least one exponential scale is required")
    approximation = sp.sympify(poincare_prefix)
    records: list[TerminantLevel] = []
    residual = sp.S.Zero
    for level in range(1, levels + 1):
        scale = scales[(level - 1) % len(scales)]
        coefficients = coefficient_provider(level, reexpansion_terms)
        record = terminant_reexpand(
            coefficients, scale, reexpansion_terms=reexpansion_terms
        )
        record = TerminantLevel(
            level,
            record.truncation_index,
            record.singulant,
            record.correction,
            record.multiplier,
            record.remainder_scale,
            record.hypotheses_verified,
        )
        approximation += scale.amplitude * sp.exp(-scale.singulant) * record.correction
        residual = record.remainder_scale
        records.append(record)
    return HyperasymptoticExpansion(
        sp.sympify(poincare_prefix),
        tuple(records),
        sp.simplify(approximation),
        residual,
        all(record.hypotheses_verified for record in records),
    )


@dataclass(frozen=True)
class StokesTransition:
    """Exact terminant multiplier together with its boundary geometry."""

    scale: ExponentialScale
    truncation_index: sp.Expr
    exact_multiplier: sp.Expr
    smoothing_approximation: sp.Expr | None
    boundary_variable: sp.Expr | None
    hypotheses_verified: bool


def stokes_transition(
    scale: ExponentialScale,
    large_parameter: sp.Symbol,
    *,
    truncation_index: sp.Expr | None = None,
) -> StokesTransition:
    """Construct an oriented terminant transition on a proved Stokes scale.

    The exact multiplier is retained. The erfc expression records the universal
    leading smoothing only when the Stokes analyzer proves O(|chi|^-1/2)
    angular scaling.
    """
    from .stokes_geometry import analyze_stokes_geometry

    geometry = analyze_stokes_geometry(scale, large_parameter)
    n = (
        least_term_index(scale.singulant)
        if truncation_index is None
        else sp.sympify(truncation_index)
    )
    exact = stokes_multiplier(scale, n)
    smoothing = None
    if geometry.hypotheses_verified and geometry.boundary_variable is not None:
        smoothing = sp.Rational(1, 2) * sp.erfc(
            -scale.orientation * geometry.boundary_variable / sp.sqrt(2)
        )
    return StokesTransition(
        scale,
        n,
        exact,
        smoothing,
        geometry.boundary_variable,
        geometry.hypotheses_verified,
    )
