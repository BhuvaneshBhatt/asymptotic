from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp


@dataclass(frozen=True)
class ExpLogExtension:
    generator: sp.Expr
    kind: Literal["log", "exp"]
    argument: sp.Expr


@dataclass(frozen=True)
class ExpLogTower:
    """A dependency-ordered list of exp/log generators in an expression."""

    variable: sp.Symbol
    extensions: tuple[ExpLogExtension, ...]

    @classmethod
    def from_expr(cls, expr: sp.Expr, variable: sp.Symbol) -> ExpLogTower:
        expr = sp.sympify(expr)
        ordered = []
        seen = set()

        def visit(node: sp.Expr) -> None:
            for arg in node.args:
                visit(arg)
            if node in seen:
                return
            if node.func is sp.exp:
                seen.add(node)
                ordered.append(ExpLogExtension(node, "exp", node.args[0]))
            elif node.func is sp.log:
                seen.add(node)
                ordered.append(ExpLogExtension(node, "log", node.args[0]))

        visit(expr)
        return cls(variable, tuple(ordered))
