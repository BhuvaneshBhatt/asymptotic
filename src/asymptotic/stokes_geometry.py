"""Automatic Stokes/anti-Stokes geometry for exponential scales."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_limit
from .hyperasymptotics import ExponentialScale
from .regime_selection import RegimeGeometry


@dataclass(frozen=True)
class StokesGeometryCertificate:
    singulant: sp.Expr
    large_parameter: sp.Symbol
    phase: sp.Expr
    boundary_variable: sp.Expr | None
    stokes: bool
    anti_stokes: bool
    geometry: RegimeGeometry
    hypotheses_verified: bool


def analyze_stokes_geometry(
    scale: ExponentialScale,
    large_parameter: sp.Symbol,
) -> StokesGeometryCertificate:
    """Prove Stokes or anti-Stokes boundary scaling from a singulant."""
    chi = sp.sympify(scale.singulant)
    n = sp.sympify(large_parameter)
    phase = sp.arg(chi)
    try:
        phase_limit = bounded_limit(phase, n, sp.oo)
    except (TypeError, ValueError, NotImplementedError, RecursionError, sp.PoleError):
        phase_limit = None
    stokes = phase_limit == 0
    anti = phase_limit in (sp.pi / 2, -sp.pi / 2)
    boundary = None
    verified = False
    if stokes:
        boundary = sp.sqrt(sp.Abs(chi)) * phase
    elif anti:
        boundary = sp.sqrt(sp.Abs(chi)) * (phase - phase_limit)
    if boundary is not None:
        try:
            boundary_limit = bounded_limit(boundary, n, sp.oo)
        except (
            TypeError,
            ValueError,
            NotImplementedError,
            RecursionError,
            sp.PoleError,
        ):
            boundary_limit = None
        verified = (
            boundary_limit is not None
            and not isinstance(boundary_limit, sp.Limit)
            and boundary_limit not in (sp.oo, -sp.oo, sp.zoo, sp.nan)
        )
    geometry = RegimeGeometry(
        stokes_singulants=(chi,) if verified else (),
        stokes_distance_scale=boundary if verified else None,
        hypotheses_verified=verified,
    )
    return StokesGeometryCertificate(
        chi, n, phase, boundary, stokes, anti, geometry, verified
    )


def select_stokes_regime(
    scale: ExponentialScale,
    large_parameter: sp.Symbol,
):
    """Analyze a scale and return the corresponding certified regime."""
    from .regime_selection import select_asymptotic_regime

    certificate = analyze_stokes_geometry(scale, large_parameter)
    return select_asymptotic_regime(
        sp.exp(-large_parameter),
        large_parameter,
        geometry=certificate.geometry,
    )
