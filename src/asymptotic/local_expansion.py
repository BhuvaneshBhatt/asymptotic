"""Certified finite local expansions used by limit proofs.

The module keeps primitive expansion knowledge separate from limit dispatch.
Providers return exact finite prefixes with an explicit SymPy order remainder;
consumers may use a prefix only when the remainder is asymptotically negligible
for the conclusion being drawn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import sympy as sp
from sympy.core.function import PoleError

from ._symbolic_policy import bounded_limit


@dataclass(frozen=True)
class SectorCertificate:
    """Open or closed angular sector on the principal argument branch."""

    lower_argument: sp.Expr
    upper_argument: sp.Expr
    boundary_margin_required: bool = False


@dataclass(frozen=True)
class ExpansionCertificate:
    """Hypotheses and remainder information supporting a local expansion."""

    sector: SectorCertificate | None = None
    parameter_conditions: tuple[sp.Expr, ...] = ()
    branch: str = "principal"
    remainder_bound: sp.Expr | None = None
    closed_subsector_uniform: bool = False
    hypotheses_verified: bool = True


@dataclass(frozen=True)
class LocalExpansion:
    """Finite local prefix together with its asymptotic remainder order."""

    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    prefix: sp.Expr
    order: sp.Expr | None
    depth: int
    provider: str
    certificate: ExpansionCertificate | None = None


ExpansionProvider = Callable[[sp.Expr, sp.Symbol, sp.Expr, int], LocalExpansion | None]
_PROVIDERS: dict[object, ExpansionProvider] = {}


def register_local_expansion(function: object, provider: ExpansionProvider) -> None:
    """Register a primitive local-expansion provider."""
    _PROVIDERS[function] = provider


@lru_cache(maxsize=4096)
def _series_expansion(
    expr: sp.Expr, variable: sp.Symbol, point: sp.Expr, depth: int
) -> LocalExpansion | None:
    """Build an exact SymPy series prefix and retain its order term."""
    try:
        series = sp.series(expr, variable, point, depth)
    except (TypeError, ValueError, NotImplementedError, PoleError, RecursionError):
        return None
    if not hasattr(series, "removeO"):
        return None
    prefix = sp.expand(series.removeO())
    order_terms = tuple(series.atoms(sp.Order))
    order = order_terms[0].expr if len(order_terms) == 1 else None
    return LocalExpansion(
        expr, variable, sp.sympify(point), prefix, order, depth, "series"
    )


def local_series(
    expr: sp.Expr, variable: sp.Symbol, point: sp.Expr = 0, *, depth: int = 4
) -> LocalExpansion | None:
    """Expand a scalar local germ after exact algebraic cancellation.

    Primitive providers are consulted when the expression itself is a registered
    function. Composite expressions use the common series backend so products
    and sums are expanded as a whole and equal singular monomials cancel before
    valuation.
    """
    expr = sp.sympify(expr)
    if point == sp.oo:
        from .uniform_integration import uniform_composite_expansion

        uniform = uniform_composite_expansion(expr, variable, sp.oo)
        if uniform is not None:
            return LocalExpansion(
                expr,
                variable,
                sp.oo,
                uniform.prefix,
                uniform.remainder_scale,
                depth,
                "uniform-asymptotic-regime",
                ExpansionCertificate(
                    branch="theorem-selected uniform branch",
                    remainder_bound=uniform.remainder_scale,
                    hypotheses_verified=uniform.certified,
                ),
            )
    provider = _PROVIDERS.get(expr.func) or _builtin_providers().get(expr.func)
    if provider is not None:
        result = provider(expr, variable, sp.sympify(point), depth)
        if result is not None:
            return result
    return _series_expansion(expr, variable, sp.sympify(point), depth)


def adaptive_path_limit(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = 0,
    *,
    depths: tuple[int, ...] = (2, 4, 6, 8),
) -> tuple[sp.Expr, LocalExpansion] | None:
    """Certify a scalar path limit from a finite prefix and vanishing remainder.

    A prefix is accepted only when its recorded order term tends to zero on the
    one-sided path germ. This makes the result suitable as a negative witness
    for a multivariate limit; agreement of path expansions is never used to
    prove existence.
    """
    for depth in depths:
        expansion = local_series(expr, variable, point, depth=depth)
        if expansion is None or expansion.order is None:
            continue
        if (
            expansion.certificate is not None
            and not expansion.certificate.hypotheses_verified
        ):
            continue
        try:
            remainder_limit = bounded_limit(
                sp.Abs(expansion.order), variable, point, direction="+"
            )
            value = bounded_limit(expansion.prefix, variable, point, direction="+")
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            continue
        if remainder_limit != 0:
            continue
        if (
            value is None
            or isinstance(value, (sp.Limit, sp.Set, sp.AccumBounds))
            or value.has(variable)
        ):
            continue
        if value in (sp.nan, sp.zoo):
            continue
        return value, expansion
    return None


@lru_cache(maxsize=1)
def _builtin_providers():
    # Provider implementations depend on these result models. Lazy loading
    # avoids an import cycle; custom registrations never overwrite builtins.
    from .special_expansions import _BUILTIN_PROVIDERS

    return _BUILTIN_PROVIDERS
