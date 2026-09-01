from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .context import AsymptoticContext, GrowthComparison, context_for
from .decomposition import StructuralDecomposition, decompose_expression
from .tower import ExpLogTower


@dataclass(frozen=True)
class MRVClass:
    """A most-rapidly-varying comparability class."""

    representative: sp.Expr
    members: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class MRVDecomposition:
    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    structural: StructuralDecomposition
    classes: tuple[MRVClass, ...]
    most_rapid: MRVClass | None

    @property
    def representative(self) -> sp.Expr | None:
        return None if self.most_rapid is None else self.most_rapid.representative


def _stirling_log_scale(z: sp.Expr, ctx: AsymptoticContext) -> sp.Expr | None:
    """Return the positive-real leading log-Gamma scale when its germ is proved."""

    z = sp.sympify(z)
    if ctx.eventual_sign(z) == 1 and ctx.limit(z) is sp.oo:
        return (z - sp.Rational(1, 2)) * sp.log(z) - z
    return None


def _factorial_family_log(expr: sp.Expr, ctx: AsymptoticContext) -> tuple[sp.Expr, bool] | None:
    """Normalize factorial-family products to one additive logarithmic scale.

    Gamma, factorial, binomial and rising-factorial/Pochhammer factors are
    rewritten to signed log-Gamma contributions.  Multiplication and constant
    powers stay additive here, so cancellations in Gamma ratios are visible to
    Hardy/MRV comparison instead of comparing the individual Gamma factors.
    The normalization is used only when every required Gamma argument is
    proved positive and unbounded on the configured germ.
    """

    expr = sp.sympify(expr)
    if expr.func is sp.gamma:
        scale = _stirling_log_scale(expr.args[0], ctx)
        return None if scale is None else (scale, True)
    if expr.func is sp.factorial:
        scale = _stirling_log_scale(expr.args[0] + 1, ctx)
        return None if scale is None else (scale, True)
    if expr.func is sp.binomial:
        upper, lower = expr.args
        if (
            ctx.variable not in lower.free_symbols
            and ctx.eventual_sign(upper) == 1
            and ctx.limit(upper) is sp.oo
        ):
            return (lower * sp.log(upper), True)
        pieces = (
            _stirling_log_scale(upper + 1, ctx),
            _stirling_log_scale(lower + 1, ctx),
            _stirling_log_scale(upper - lower + 1, ctx),
        )
        if any(piece is None for piece in pieces):
            return None
        return (pieces[0] - pieces[1] - pieces[2], True)
    if expr.func is sp.RisingFactorial:
        start, length = expr.args
        if (
            ctx.variable not in length.free_symbols
            and ctx.eventual_sign(start) == 1
            and ctx.limit(start) is sp.oo
        ):
            return (length * sp.log(start), True)
        upper = _stirling_log_scale(start + length, ctx)
        lower = _stirling_log_scale(start, ctx)
        if upper is None or lower is None:
            return None
        return (upper - lower, True)
    if expr.is_Pow and ctx.variable not in expr.exp.free_symbols:
        normalized = _factorial_family_log(expr.base, ctx)
        if normalized is None:
            return None
        value, found = normalized
        return (expr.exp * value, found)
    if expr.is_Mul:
        total = sp.S.Zero
        found = False
        for factor in expr.args:
            normalized = _factorial_family_log(factor, ctx)
            if normalized is not None:
                value, contains_family = normalized
                total += value
                found = found or contains_family
            else:
                # Non-factorial factors still contribute their ordinary
                # logarithmic magnitude to the product's net variation.
                total += sp.log(sp.Abs(factor))
        return (sp.expand(total), found) if found else None
    return None


def _variation_measure(expr: sp.Expr, ctx: AsymptoticContext) -> sp.Expr:
    """Return a logarithmic magnitude whose growth measures variation rate."""

    expr = sp.sympify(expr)
    if expr.func is sp.exp:
        return sp.Abs(expr.args[0])
    factorial_measure = _factorial_family_log(expr, ctx)
    if factorial_measure is not None:
        value, found = factorial_measure
        if found:
            return sp.Abs(value)
    return sp.log(1 + sp.Abs(expr))


def mrv_decomposition(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = sp.oo,
    *,
    context: AsymptoticContext | None = None,
    structural: StructuralDecomposition | None = None,
) -> MRVDecomposition:
    """Compute explicit MRV comparability classes for an expression.

    Candidates come from the dependency-ordered exp/log tower plus the
    independent variable.  They are grouped by logarithmic growth and the
    class with the largest variation measure is selected.  Unknown comparisons
    remain separate rather than being silently equated.
    """

    expr = sp.sympify(expr)
    ctx = context_for(variable, point, context)
    structural = structural or decompose_expression(expr, variable)
    tower = ExpLogTower.from_expr(structural.canonical, variable)
    candidates: list[sp.Expr] = [variable]
    candidates.extend(ext.generator for ext in tower.extensions if ext.generator.has(variable))
    # Include maximal composition inners because a non-exp/log analytic wrapper
    # can hide the actual rapidly varying argument.
    candidates.extend(layer.inner for layer in structural.composition if layer.inner.has(variable))
    # Gamma/factorial factors are genuine Hardy-scale generators even though
    # they do not belong to the exp/log tower syntactically.
    factorial_funcs = {sp.gamma, sp.factorial, sp.binomial, sp.RisingFactorial}
    family_nodes = [
        node
        for node in sp.preorder_traversal(structural.canonical)
        if node.has(variable)
        and (
            getattr(node, "func", None) in factorial_funcs
            or ((node.is_Mul or node.is_Pow) and node.has(*factorial_funcs))
        )
    ]
    # Keep maximal factorial-family products/ratios.  Their constituent Gamma
    # factors may cancel and must not independently re-enter the MRV race.
    maximal_family = [
        node
        for node in family_nodes
        if not any(node != other and other.has(node) for other in family_nodes)
    ]
    candidates.extend(maximal_family)

    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)

    groups = []
    for candidate in unique:
        placed = False
        for group in groups:
            rel, _ = ctx.compare_growth(
                _variation_measure(candidate, ctx), _variation_measure(group[0], ctx)
            )
            if rel is GrowthComparison.SAME_ORDER:
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    classes = tuple(MRVClass(group[0], tuple(group)) for group in groups)
    most: MRVClass | None = classes[0] if len(classes) == 1 else None
    if len(classes) > 1:
        proven_maxima: list[MRVClass] = []
        for candidate in classes:
            dominates_all = True
            for other in classes:
                if candidate is other:
                    continue
                rel, _ = ctx.compare_growth(
                    _variation_measure(candidate.representative, ctx),
                    _variation_measure(other.representative, ctx),
                )
                if rel is not GrowthComparison.LARGER:
                    dominates_all = False
                    break
            if dominates_all:
                proven_maxima.append(candidate)
        if len(proven_maxima) == 1:
            most = proven_maxima[0]
    return MRVDecomposition(expr, variable, point, structural, classes, most)
