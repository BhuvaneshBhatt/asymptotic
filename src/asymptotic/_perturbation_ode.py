"""Shared bounded ODE solving utilities for perturbation methods."""

from __future__ import annotations

import re
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

from ._symbolic_policy import DEFAULT_SYMBOLIC_POLICY, bounded_solve_system

_CONSTANT_NAME = re.compile(r"C\d+\Z")


def independent_variable(unknowns: tuple[sp.Expr, ...]) -> sp.Symbol | None:
    """Return the common scalar independent variable of function unknowns."""
    if not unknowns or not all(
        isinstance(unknown, AppliedUndef) for unknown in unknowns
    ):
        return None
    variables = []
    for unknown in unknowns:
        if len(unknown.args) != 1 or not isinstance(unknown.args[0], sp.Symbol):
            return None
        variables.append(unknown.args[0])
    first = variables[0]
    return first if all(variable == first for variable in variables) else None


def replace_function_family(
    expression: sp.Expr,
    function: sp.FunctionClass,
    replacement: sp.Expr,
    variable: sp.Symbol,
) -> sp.Expr:
    """Replace all one-variable evaluations of a function by one expression."""

    def matches(node: sp.Basic) -> bool:
        return (
            isinstance(node, AppliedUndef)
            and node.func == function
            and len(node.args) == 1
        )

    def replace(node: AppliedUndef) -> sp.Expr:
        return replacement.subs(variable, node.args[0])

    return sp.expand(expression.replace(matches, replace).doit())


def introduced_constants(
    expressions: tuple[sp.Expr, ...],
    known_symbols: set[sp.Symbol],
) -> tuple[sp.Symbol, ...]:
    """Return integration constants introduced by SymPy's ODE solver."""
    symbols: set[sp.Symbol] = set()
    for expression in expressions:
        symbols.update(expression.free_symbols)
    return tuple(
        sorted(
            (
                symbol
                for symbol in symbols - known_symbols
                if _CONSTANT_NAME.fullmatch(symbol.name)
            ),
            key=lambda symbol: symbol.name,
        )
    )


def apply_ode_conditions(
    solutions: tuple[sp.Expr, ...],
    unknowns: tuple[AppliedUndef, ...],
    conditions: tuple[sp.Expr, ...],
    variable: sp.Symbol,
    known_symbols: set[sp.Symbol],
) -> tuple[sp.Expr, ...] | None:
    """Determine integration constants from one perturbation order's data."""
    if not conditions:
        return solutions
    residuals = list(conditions)
    for unknown, solution in zip(unknowns, solutions):
        residuals = [
            replace_function_family(residual, unknown.func, solution, variable)
            for residual in residuals
        ]
    constants = introduced_constants(tuple(solutions), known_symbols)
    if not constants:
        if all(sp.simplify(residual) == 0 for residual in residuals):
            return solutions
        return None
    solved = bounded_solve_system(residuals, constants, allow_general=True)
    if solved is None or len(solved) != 1:
        return None
    constant_values = solved[0]
    return tuple(sp.simplify(solution.subs(constant_values)) for solution in solutions)


def dsolve_order(
    equations: tuple[sp.Expr, ...],
    unknowns: tuple[AppliedUndef, ...],
) -> tuple[sp.Expr, ...] | None:
    """Solve a small scalar or coupled ODE coefficient problem within a budget."""
    total_ops = sum(int(sp.count_ops(equation, visual=False)) for equation in equations)
    if total_ops > 4 * DEFAULT_SYMBOLIC_POLICY.solve_ops:
        return None
    try:
        if len(unknowns) == 1 and len(equations) == 1:
            solved: Any = sp.dsolve(sp.Eq(equations[0], 0), unknowns[0])
            if not isinstance(solved, sp.Equality) or solved.lhs != unknowns[0]:
                return None
            return (sp.sympify(solved.rhs),)
        solved = sp.dsolve(
            [sp.Eq(equation, 0) for equation in equations],
            list(unknowns),
        )
    except (
        NotImplementedError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        sp.PolynomialError,
    ):
        return None
    if isinstance(solved, sp.Equality):
        solved = [solved]
    if not isinstance(solved, (list, tuple)):
        return None
    by_unknown = {
        equation.lhs: equation.rhs
        for equation in solved
        if isinstance(equation, sp.Equality)
    }
    if any(unknown not in by_unknown for unknown in unknowns):
        return None
    return tuple(sp.sympify(by_unknown[unknown]) for unknown in unknowns)


__all__ = ["apply_ode_conditions", "dsolve_order", "independent_variable"]
