"""Periodic resonance and Fredholm solvability conditions.

The routines here are independent of any particular perturbation method.  A
periodic order equation is solvable only when its forcing is orthogonal to the
relevant adjoint null modes.  Lindstedt–Poincaré and multiple-scale methods
can therefore share the same projection machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_definite_integral, bounded_simplify
from .perturbation import SolvabilityCondition


@dataclass(frozen=True)
class PeriodicSolvabilityResult:
    """Orthogonality conditions for one periodic perturbation forcing."""

    forcing: sp.Expr
    variable: sp.Symbol
    period: sp.Expr
    adjoint_modes: tuple[sp.Expr, ...]
    projections: tuple[sp.Expr, ...]
    conditions: tuple[SolvabilityCondition, ...]

    @property
    def satisfied(self) -> bool | None:
        """Return whether every projection is exactly zero when decidable."""
        undecided = False
        for projection in self.projections:
            simplified = bounded_simplify(projection)
            if simplified == 0:
                continue
            if simplified.is_zero is False:
                return False
            undecided = True
        return None if undecided else True


def periodic_solvability_conditions(
    forcing: sp.Expr,
    variable: sp.Symbol,
    adjoint_modes: Sequence[sp.Expr],
    *,
    period: sp.Expr = 2 * sp.pi,
    start: sp.Expr = 0,
    source_order: int = 0,
) -> PeriodicSolvabilityResult:
    """Project periodic forcing onto supplied adjoint null modes.

    For each mode ``phi`` this computes the normalized inner product
    ``Integral(forcing*phi, (variable, start, start + period))/period``.  A
    vanishing projection is the Fredholm solvability condition that prevents
    resonant secular growth.  Uncomputed integrals remain explicit symbolic
    ``Integral`` objects rather than being guessed.
    """
    variable = sp.sympify(variable)
    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a Symbol")
    period = sp.sympify(period)
    if period.is_zero is True:
        raise ValueError("period must be nonzero")
    modes = tuple(map(sp.sympify, adjoint_modes))
    if not modes:
        raise ValueError("at least one adjoint null mode is required")
    forcing = sp.sympify(forcing)

    projections: list[sp.Expr] = []
    conditions: list[SolvabilityCondition] = []
    for mode in modes:
        integrand = sp.expand(forcing * mode)
        value = bounded_definite_integral(
            integrand,
            variable,
            start,
            start + period,
        )
        if value is None:
            value = sp.Integral(integrand, (variable, start, start + period))
        projection = bounded_simplify(value / period)
        projections.append(projection)
        conditions.append(
            SolvabilityCondition(
                expression=projection,
                source_order=source_order,
                reason="periodic forcing must be orthogonal to an adjoint null mode",
            )
        )
    return PeriodicSolvabilityResult(
        forcing=forcing,
        variable=variable,
        period=period,
        adjoint_modes=modes,
        projections=tuple(projections),
        conditions=tuple(conditions),
    )


__all__ = ["PeriodicSolvabilityResult", "periodic_solvability_conditions"]
