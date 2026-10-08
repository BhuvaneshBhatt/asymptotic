"""High-level asymptotic expansion dispatcher."""

from __future__ import annotations

from typing import Literal

import sympy as sp

from .multiseries import Multiseries, multiseries
from .nested import NestedExpansion, nested_series
from .puiseux import PuiseuxSeries, puiseux_series
from .transseries import TransseriesExpansion, transseries_from_expression

ExpansionMethod = Literal["auto", "multiseries", "nested", "puiseux", "transseries"]
ExpansionResult = Multiseries | NestedExpansion | PuiseuxSeries | TransseriesExpansion


def _auto_method(expr: sp.Expr, variable: sp.Symbol, point: sp.Expr) -> ExpansionMethod:
    """Choose a representation from cheap structural predicates only."""
    if point not in (sp.oo, -sp.oo):
        # Algebraic finite germs, including fractional powers, belong naturally
        # to Puiseux. Nested exp/log structure needs the nested engine.
        if expr.has(sp.exp, sp.log) and any(
            node.has(sp.exp, sp.log)
            for node in sp.preorder_traversal(expr)
            if node is not expr and isinstance(node, sp.Expr)
        ):
            return "nested"
        return "puiseux"
    # At infinity, nested logarithmic/exponential scales are classified without
    # trying several expansion engines and catching failures.
    depth = 0
    for node in sp.preorder_traversal(expr):
        if getattr(node, "func", None) in (sp.exp, sp.log):
            depth = max(
                depth,
                sum(
                    1
                    for parent in sp.preorder_traversal(node.args[0])
                    if getattr(parent, "func", None) in (sp.exp, sp.log)
                )
                + 1,
            )
    return "nested" if depth >= 2 else "multiseries"


def expand(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    assumptions: sp.Expr = sp.S.true,
    method: ExpansionMethod = "auto",
) -> ExpansionResult:
    """Expand *expr* asymptotically using one task-oriented entry point.

    ``method="auto"`` uses a local Puiseux expansion at finite points and the
    general multiseries engine at infinity. Specialized representations remain
    directly selectable when the user knows the desired asymptotic structure.
    The dispatcher contains no independent asymptotic algorithm.
    """
    expr = sp.sympify(expr)
    variable = sp.sympify(variable)
    point = sp.sympify(point)
    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a SymPy Symbol")
    if isinstance(terms, bool) or not isinstance(terms, int) or terms < 1:
        raise ValueError("terms must be a positive integer")

    valid = {"auto", "multiseries", "nested", "puiseux", "transseries"}
    if method not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(
            f"unknown expansion method {method!r}; choose one of: {choices}"
        )

    selected = method
    if selected == "auto":
        selected = _auto_method(expr, variable, point)

    if selected == "multiseries":
        return multiseries(
            expr, variable, point=point, terms=terms, assumptions=assumptions
        )
    if selected == "nested":
        return nested_series(
            expr, variable, point=point, depth=terms, assumptions=assumptions
        )
    if selected == "puiseux":
        return puiseux_series(
            expr, variable, point=point, terms=terms, assumptions=assumptions
        )

    refined = sp.refine(expr, assumptions)
    return transseries_from_expression(refined, variable, point=point)
