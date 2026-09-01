"""Nested logarithmico-exponential scale analysis.

This module provides a structural scale descriptor and comparison routine for
finite-height exp/log expressions.  It complements ``AsymptoticMonomial``:
monomials remain the efficient multiplicative representation, while this layer
handles nested generators such as ``exp(exp(x))`` and ``log(log(x))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, GrowthComparison
from .tower import ExpLogTower


@dataclass(frozen=True)
class LogExpScale:
    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    exponential_height: int
    logarithmic_depth: int
    tower: ExpLogTower

    @classmethod
    def from_expr(
        cls, expr: sp.Expr, variable: sp.Symbol, *, point: sp.Expr = sp.oo
    ) -> LogExpScale:
        expr = analytic_powsimp(sp.sympify(expr))
        tower = ExpLogTower.from_expr(expr, variable)

        def exp_height(node: sp.Expr) -> int:
            child = max((exp_height(arg) for arg in node.args), default=0)
            return child + 1 if node.func is sp.exp and node.has(variable) else child

        def log_depth(node: sp.Expr) -> int:
            child = max((log_depth(arg) for arg in node.args), default=0)
            return child + 1 if node.func is sp.log and node.has(variable) else child

        return cls(
            expr,
            variable,
            sp.sympify(point),
            exp_height(expr),
            log_depth(expr),
            tower,
        )


def _ratio_comparison(
    left: sp.Expr, right: sp.Expr, ctx: AsymptoticContext
) -> GrowthComparison | None:
    quotient = analytic_powsimp(sp.simplify(left / right))
    candidates = [quotient]
    if quotient.is_positive is not True:
        candidates.append(sp.Abs(quotient))
    for target in candidates:
        try:
            ratio = ctx.limit(target)
        except SYMBOLIC_ERRORS:
            ratio = None
        if ratio == 0:
            return GrowthComparison.SMALLER
        if ratio in (sp.oo, -sp.oo):
            return GrowthComparison.LARGER
        if getattr(ratio, "is_finite", None) is True and ratio.is_zero is False:
            return GrowthComparison.SAME_ORDER
    return None


def compare_log_exp_scales(
    left: sp.Expr | LogExpScale,
    right: sp.Expr | LogExpScale,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    max_log_reductions: int = 8,
) -> GrowthComparison:
    """Compare finite-height logarithmico-exponential magnitudes.

    Ratio limits are attempted first.  When direct comparison is opaque, the
    algorithm repeatedly takes logarithms of magnitudes; this turns nested
    exponentials into their arguments and exposes their first differing scale.
    """

    l_expr = left.expression if isinstance(left, LogExpScale) else sp.sympify(left)
    r_expr = right.expression if isinstance(right, LogExpScale) else sp.sympify(right)
    ctx = AsymptoticContext(variable, point=point)
    direct = _ratio_comparison(l_expr, r_expr, ctx)
    if direct is not None:
        return direct

    l_cur = sp.Abs(l_expr)
    r_cur = sp.Abs(r_expr)
    for _ in range(max_log_reductions):
        try:
            l_lim = ctx.limit(l_cur)
            r_lim = ctx.limit(r_cur)
        except SYMBOLIC_ERRORS:
            break
        # Taking logs preserves ordering for eventually positive quantities
        # tending to infinity.  If only one diverges, the answer is immediate.
        if l_lim is sp.oo and r_lim is not sp.oo:
            return GrowthComparison.LARGER
        if r_lim is sp.oo and l_lim is not sp.oo:
            return GrowthComparison.SMALLER
        if l_lim is not sp.oo or r_lim is not sp.oo:
            break
        l_cur = sp.log(l_cur)
        r_cur = sp.log(r_cur)
        direct = _ratio_comparison(l_cur, r_cur, ctx)
        if direct in (GrowthComparison.LARGER, GrowthComparison.SMALLER):
            return direct
        # SAME_ORDER after logarithms does not imply same order before logs;
        # inspect the difference of logarithms, i.e. log(left/right).
        try:
            delta = ctx.limit(sp.simplify(l_cur - r_cur))
        except SYMBOLIC_ERRORS:
            delta = None
        if delta is sp.oo:
            return GrowthComparison.LARGER
        if delta is -sp.oo:
            return GrowthComparison.SMALLER
        if getattr(delta, "is_finite", None) is True:
            return GrowthComparison.SAME_ORDER
    return GrowthComparison.UNKNOWN
