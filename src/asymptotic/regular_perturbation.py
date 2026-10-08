"""Regular perturbation solving built on the common hierarchy layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

from ._perturbation_ode import apply_ode_conditions, dsolve_order, independent_variable
from ._symbolic_policy import bounded_solve_system
from .perturbation import (
    EquationLike,
    PerturbationHierarchy,
    _substitute_unknowns,
    perturbation_hierarchy,
)


@dataclass(frozen=True)
class RegularPerturbationBranch:
    """One recursively solved branch of a regular perturbation problem."""

    hierarchy: PerturbationHierarchy
    complete: bool
    unresolved_order: int | None = None
    limitation: str | None = None

    @property
    def approximation(self) -> tuple[sp.Expr, ...]:
        """Return the reconstructed approximation through the solved prefix."""
        through = self.hierarchy.solved_through
        if through is None:
            through = 0
        return self.hierarchy.approximation(through)

    @property
    def residual(self) -> tuple[sp.Expr, ...]:
        """Return original-equation residuals for the reconstructed prefix."""
        through = self.hierarchy.solved_through
        if through is None:
            through = 0
        return self.hierarchy.residual(through)

    @property
    def condition_residual(self) -> tuple[sp.Expr, ...]:
        """Return condition residuals for the reconstructed prefix."""
        through = self.hierarchy.solved_through
        if through is None:
            through = 0
        return self.hierarchy.condition_residual(through)

    @property
    def verified(self) -> bool:
        """Whether all solved hierarchy equations and conditions vanish exactly."""
        through = self.hierarchy.solved_through
        if through is None:
            return False
        substitutions = self.hierarchy.substitutions(through)
        supplied_unknowns = tuple(substitutions)
        supplied_values = tuple(substitutions.values())
        for index in range(through + 1):
            order = self.hierarchy.order(index)
            residuals = order.equations + order.conditions
            for expression in residuals:
                evaluated = _substitute_unknowns(
                    expression, supplied_unknowns, supplied_values
                )
                if sp.simplify(evaluated) != 0:
                    return False
        return True


@dataclass(frozen=True)
class RegularPerturbationResult:
    """Structured result from :func:`regular_perturbation`.

    ``branches`` preserves every branch that the bounded recursive solver can
    certify.  A branch may be partial when a later perturbation order cannot be
    solved symbolically; earlier solved orders remain available for inspection
    and manual continuation through the hierarchy API.
    """

    branches: tuple[RegularPerturbationBranch, ...]
    parameter: sp.Symbol
    unknowns: tuple[sp.Expr, ...]
    gauges: tuple[sp.Expr, ...]
    method: str = "regular"

    @property
    def complete(self) -> bool:
        """Whether every retained branch is solved through the requested order."""
        return bool(self.branches) and all(branch.complete for branch in self.branches)

    @property
    def approximations(self) -> tuple[tuple[sp.Expr, ...], ...]:
        """Return reconstructed approximations for all retained branches."""
        return tuple(branch.approximation for branch in self.branches)


def _substitute_previous(
    order, hierarchy: PerturbationHierarchy
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    substitutions = hierarchy.substitutions(order.index - 1) if order.index else {}
    equations = tuple(
        sp.expand(eq.subs(substitutions).doit()) for eq in order.equations
    )
    conditions = tuple(
        sp.expand(cond.subs(substitutions).doit()) for cond in order.conditions
    )
    return equations, conditions


def _solve_algebraic_order(
    hierarchy: PerturbationHierarchy,
    index: int,
) -> tuple[dict[sp.Expr, sp.Expr], ...] | None:
    order = hierarchy.order(index)
    equations, conditions = _substitute_previous(order, hierarchy)
    system = equations + conditions
    if not system:
        return ({},)
    solved = bounded_solve_system(
        system,
        order.unknowns,
        allow_general=True,
    )
    if solved is None:
        return None
    return tuple(dict(solution) for solution in solved)


def _solve_ode_order(
    hierarchy: PerturbationHierarchy,
    index: int,
) -> tuple[dict[sp.Expr, sp.Expr], ...] | None:
    order = hierarchy.order(index)
    variable = independent_variable(order.unknowns)
    if variable is None:
        return None
    equations, conditions = _substitute_previous(order, hierarchy)
    unknowns = tuple(order.unknowns)
    solutions = dsolve_order(equations, unknowns)
    if solutions is None:
        return None
    known_symbols = set().union(
        *(equation.free_symbols for equation in equations + conditions),
    )
    known_symbols.add(variable)
    conditioned = apply_ode_conditions(
        solutions,
        unknowns,
        conditions,
        variable,
        known_symbols,
    )
    if conditioned is None:
        return None
    return (dict(zip(unknowns, conditioned)),)


def _solve_order(
    hierarchy: PerturbationHierarchy,
    index: int,
) -> tuple[dict[sp.Expr, sp.Expr], ...] | None:
    unknowns = hierarchy.order(index).unknowns
    if all(isinstance(unknown, sp.Symbol) for unknown in unknowns):
        return _solve_algebraic_order(hierarchy, index)
    if all(isinstance(unknown, AppliedUndef) for unknown in unknowns):
        return _solve_ode_order(hierarchy, index)
    return None


def _advance_branch(
    hierarchy: PerturbationHierarchy,
    index: int,
) -> tuple[PerturbationHierarchy, ...] | None:
    solutions = _solve_order(hierarchy, index)
    if solutions is None:
        return None
    return tuple(hierarchy.with_solution(index, solution) for solution in solutions)


def regular_perturbation(
    equations: EquationLike | Sequence[EquationLike],
    unknowns: sp.Expr | Sequence[sp.Expr],
    parameter: sp.Symbol,
    *,
    order: int | None = None,
    gauges: Sequence[sp.Expr] | None = None,
    conditions: EquationLike | Sequence[EquationLike] = (),
    assumptions: sp.Expr = sp.S.true,
) -> RegularPerturbationResult:
    """Solve a regular perturbation problem recursively order by order.

    The function first constructs the common :func:`perturbation_hierarchy`,
    then solves each coefficient problem using bounded algebraic solving or
    ordinary differential-equation solving as appropriate.  Algebraic branch
    multiplicity is preserved.  If a later order cannot be solved, the solved
    prefix is returned as a partial branch instead of being discarded or
    guessed.

    Initial, boundary, and normalization conditions are expanded at the same
    perturbation orders as the governing equations.  ODE integration constants
    are fixed from the conditions whenever they are uniquely determined.
    """
    hierarchy = perturbation_hierarchy(
        equations,
        unknowns,
        parameter,
        order=order,
        gauges=gauges,
        conditions=conditions,
        assumptions=assumptions,
    )
    active = (hierarchy,)
    partial: list[RegularPerturbationBranch] = []

    for index in range(len(hierarchy.orders)):
        next_active: list[PerturbationHierarchy] = []
        for branch in active:
            advanced = _advance_branch(branch, index)
            if advanced is None:
                partial.append(
                    RegularPerturbationBranch(
                        hierarchy=branch,
                        complete=False,
                        unresolved_order=index,
                        limitation=f"could not solve perturbation order {index}",
                    )
                )
                continue
            next_active.extend(advanced)
        active = tuple(next_active)
        if not active:
            break

    complete = tuple(
        RegularPerturbationBranch(hierarchy=branch, complete=True) for branch in active
    )
    branches = tuple(partial) + complete
    return RegularPerturbationResult(
        branches=branches,
        parameter=hierarchy.parameter,
        unknowns=hierarchy.unknowns,
        gauges=hierarchy.gauges,
    )


__all__ = [
    "RegularPerturbationBranch",
    "RegularPerturbationResult",
    "regular_perturbation",
]
