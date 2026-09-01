"""High-level asymptotic solving for scalar recurrences."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_rsolve
from .discrete_scale import (
    DiscreteAsymptoticBranch,
    birkhoff_trjitzinsky_branches,
    inhomogeneous_particular_solution,
    linear_recurrence_data,
)
from .transseries import TransseriesExpansion, transseries_from_expression


@dataclass(frozen=True)
class AsymptoticRSolveResult:
    """Result of asymptotically solving a scalar recurrence.

    Exact recurrence solving is preferred when available. Otherwise polynomial-
    coefficient linear recurrences can be analyzed by discrete Newton polygons
    and factorial-scale Birkhoff--Trjitzinsky lifting.
    """

    expression: sp.Expr
    sequence: sp.Expr
    index: sp.Symbol
    point: sp.Expr
    status: str
    method: str
    series: TransseriesExpansion | None = None
    limitation: str | None = None
    branches: tuple[DiscreteAsymptoticBranch, ...] = ()
    particular_expression: sp.Expr | None = None
    particular_residual: sp.Expr | None = None

    def residual(self, recurrence: sp.Expr | sp.Equality) -> sp.Expr:
        """Return the recurrence residual of the reported finite expression."""

        recurrence = sp.sympify(recurrence)
        if isinstance(recurrence, sp.Equality):
            recurrence = recurrence.lhs - recurrence.rhs
        function = self.sequence.func

        def replace_shift(node: sp.Basic) -> sp.Basic:
            if getattr(node, "func", None) == function and len(node.args) == 1:
                return self.expression.subs(self.index, node.args[0])
            return node

        substituted = recurrence.replace(
            lambda node: getattr(node, "func", None) == function and len(node.args) == 1,
            replace_shift,
        )
        return sp.expand(substituted)


def asymptotic_rsolve(
    recurrence: sp.Expr | sp.Equality,
    sequence: sp.Expr,
    index: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    initial_conditions: dict | None = None,
    method: str = "auto",
) -> AsymptoticRSolveResult:
    """Solve a scalar recurrence and expand the resulting solution asymptotically.

    ``auto`` prefers an exact recurrence solution and otherwise applies native
    discrete Newton analysis. Simple roots use ordinary Birkhoff--Trjitzinsky
    lifting, repeated constant-coefficient roots use exact polynomial Jordan
    chains, and supported repeated variable-coefficient roots use secondary
    Newton phases with ramified inverse-power lattices.
    """

    if not isinstance(index, sp.Symbol):
        raise TypeError("index must be a Symbol")
    if terms < 1:
        raise ValueError("terms must be positive")
    recurrence = sp.sympify(recurrence)
    if isinstance(recurrence, sp.Equality):
        recurrence = sp.expand(recurrence.lhs - recurrence.rhs)
    sequence = sp.sympify(sequence)

    if method not in {"auto", "exact", "native"}:
        raise ValueError("method must be one of: auto, exact, native")

    if point is not sp.oo and method == "native":
        raise NotImplementedError("native discrete asymptotic lifting is defined at +oo")

    exact = None
    exact_error = None
    if method in {"auto", "exact"}:
        exact = bounded_rsolve(
            recurrence, sequence, initial_conditions=initial_conditions, allow_general=True
        )
        if method == "exact" and exact is None:
            raise NotImplementedError(
                "exact recurrence solving did not produce a solution"
            ) from exact_error

    if exact is None and method in {"auto", "native"}:
        if initial_conditions:
            raise NotImplementedError(
                "native asymptotic branches do not determine connection constants from initial data"
            )
        try:
            data = linear_recurrence_data(recurrence, sequence, index)
            branches = birkhoff_trjitzinsky_branches(data, terms=terms)
            particular_data = inhomogeneous_particular_solution(data, terms=terms)
        except (ValueError, NotImplementedError, sp.PolynomialError) as exc:
            if method == "native":
                raise NotImplementedError(
                    "native discrete asymptotic lifting did not resolve the recurrence"
                ) from exc
            branches = ()
            particular_data = None
        if branches or particular_data is not None:
            constants = sp.symbols(f"C1:{len(branches) + 1}")
            homogeneous = sum(c * b.expression for c, b in zip(constants, branches))
            particular = sp.S.Zero if particular_data is None else particular_data[0]
            particular_residual = None if particular_data is None else particular_data[1]
            expression = sp.expand(homogeneous + particular)
            return AsymptoticRSolveResult(
                expression=expression,
                sequence=sequence,
                index=index,
                point=sp.sympify(point),
                status="FORMAL",
                method="discrete-newton-birkhoff-trjitzinsky",
                series=None,
                limitation=(
                    None
                    if len(branches) == data.order
                    else "the native Newton hierarchy did not resolve a complete fundamental set"
                ),
                branches=branches,
                particular_expression=None if particular_data is None else particular,
                particular_residual=particular_residual,
            )

    if exact is None:
        raise NotImplementedError(
            "recurrence was not resolved by exact or native discrete asymptotic solving"
        ) from exact_error
    exact = sp.sympify(exact)

    try:
        series = transseries_from_expression(exact, index, point=point, complete=True)
        expression = series.truncate(terms)
        status = "FORMAL" if index in exact.free_symbols else "EXACT"
    except (TypeError, ValueError, NotImplementedError):
        try:
            expression = sp.series(exact, index, point, terms).removeO()
        except (TypeError, ValueError, NotImplementedError):
            expression = exact
        series = None
        status = "FORMAL" if index in exact.free_symbols else "EXACT"

    return AsymptoticRSolveResult(
        expression=sp.sympify(expression),
        sequence=sequence,
        index=index,
        point=sp.sympify(point),
        status=status,
        method="exact-rsolve-then-asymptotic",
        series=series,
        limitation=None,
        branches=(),
    )
