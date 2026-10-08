"""Common local-stratum protocol and dimension-stratified frontier complexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import sympy as sp


@runtime_checkable
class LocalStratumProtocol(Protocol):
    """Structural protocol shared by local valuation and cluster strata."""

    @property
    def stratum_kind(self) -> str: ...

    @property
    def local_constraint(self) -> sp.Expr: ...

    @property
    def intrinsic_dimension(self) -> int | None: ...

    @property
    def valuation_data(self) -> Any: ...

    @property
    def child_strata(self) -> tuple[Any, ...]: ...

    @property
    def stratum_certified(self) -> bool: ...


@dataclass(frozen=True)
class LocalStratum:
    """Subsystem-neutral local stratum used by recursive cluster geometry."""

    stratum_kind: str
    local_constraint: sp.Expr = sp.S.true
    intrinsic_dimension: int | None = None
    valuation_data: Any = None
    child_strata: tuple[LocalStratum, ...] = ()
    stratum_certified: bool = True
    payload: Any = None
    label: str = ""


@dataclass(frozen=True)
class FrontierStratum:
    """One pure-dimensional piece of a relative-frontier incidence complex."""

    index: int
    dimension: int
    formula: sp.Expr
    closure: sp.Expr
    relative_interior: sp.Expr
    frontier: sp.Expr
    incident_to: tuple[int, ...] = ()


@dataclass(frozen=True)
class FrontierIncidenceComplex:
    """Dimension-stratified semialgebraic frontier with certified incidence."""

    variables: tuple[sp.Symbol, ...]
    formula: sp.Expr
    dimension: int
    strata: tuple[FrontierStratum, ...]
    incidence: tuple[tuple[int, int], ...]
    certified: bool


def as_local_stratum(value: Any) -> LocalStratum:
    """Adapt a subsystem object implementing the local-stratum protocol."""
    if isinstance(value, LocalStratum):
        return value
    if isinstance(value, LocalStratumProtocol):
        return LocalStratum(
            value.stratum_kind,
            sp.sympify(value.local_constraint),
            value.intrinsic_dimension,
            value.valuation_data,
            tuple(as_local_stratum(c) for c in value.child_strata),
            bool(value.stratum_certified),
            value,
            type(value).__name__,
        )
    raise TypeError(f"{type(value).__name__} does not implement LocalStratumProtocol")


def combine_local_strata(
    *values: Any, kind: str = "product", constraint=True
) -> LocalStratum:
    """Form a recursive product/fiber stratum without subsystem-specific logic."""
    children = tuple(as_local_stratum(v) for v in values)
    dims = [c.intrinsic_dimension for c in children]
    dimension = sum(dims) if all(d is not None for d in dims) else None
    return LocalStratum(
        kind,
        sp.And(sp.sympify(constraint), *(c.local_constraint for c in children)),
        dimension,
        None,
        children,
        all(c.stratum_certified for c in children),
        None,
        kind,
    )


def frontier_incidence_complex(region, variables) -> FrontierIncidenceComplex | None:
    """Build a relative-frontier complex grouped by intrinsic dimension.

    Unlike ambient boundary, this recursively removes the relative interior of
    each component.  A closed segment therefore has one 1D interior stratum and
    two 0D endpoint strata rather than being its own boundary.
    """
    from semialg import (
        connected_components,
        is_satisfiable,
        region_closure,
        region_dimension,
        relative_interior,
    )

    variables = tuple(variables)
    formula = sp.sympify(region)
    try:
        top_dimension = int(region_dimension(formula, variables))
    except (ArithmeticError, ValueError, NotImplementedError, sp.PolynomialError):
        return None

    raw: list[tuple[int, sp.Expr, sp.Expr, sp.Expr, sp.Expr]] = []
    seen: set[str] = set()
    pending = [formula]
    certified = True
    while pending:
        piece = pending.pop()
        key = sp.srepr(piece)
        if key in seen or piece is sp.S.false:
            continue
        seen.add(key)
        try:
            components = tuple(connected_components(piece, variables)) or (piece,)
        except (ArithmeticError, ValueError, NotImplementedError, sp.PolynomialError):
            components = (piece,)
            certified = False
        for component in components:
            try:
                dim = int(region_dimension(component, variables))
                closure = sp.simplify(region_closure(component, variables))
                if dim == 0:
                    interior = component
                    frontier = sp.S.false
                else:
                    interior = sp.simplify(relative_interior(component, variables))
                    frontier = sp.And(closure, sp.Not(interior))
            except (
                ArithmeticError,
                ValueError,
                NotImplementedError,
                sp.PolynomialError,
            ):
                certified = False
                continue
            body = sp.And(component, interior)
            if body is not sp.S.false:
                raw.append((dim, body, closure, interior, frontier))
            if frontier is not sp.S.false:
                try:
                    fd = int(region_dimension(frontier, variables))
                except (
                    ArithmeticError,
                    ValueError,
                    NotImplementedError,
                    sp.PolynomialError,
                ):
                    certified = False
                    continue
                if fd < dim:
                    pending.append(frontier)

    # Deduplicate formulas and order high dimension first.
    unique = {}
    for item in raw:
        unique[(item[0], sp.srepr(item[1]))] = item
    ordered = sorted(unique.values(), key=lambda x: (-x[0], sp.default_sort_key(x[1])))
    incidence: list[tuple[int, int]] = []
    incident = {i: set() for i in range(len(ordered))}
    for i, (di, _, cli, _, _) in enumerate(ordered):
        for j, (dj, fj, _, _, _) in enumerate(ordered):
            if di <= dj:
                continue
            try:
                if is_satisfiable(sp.And(cli, fj), variables):
                    incidence.append((i, j))
                    incident[i].add(j)
                    incident[j].add(i)
            except (
                ArithmeticError,
                ValueError,
                NotImplementedError,
                sp.PolynomialError,
            ):
                certified = False
    strata = tuple(
        FrontierStratum(i, d, f, cl, ri, fr, tuple(sorted(incident[i])))
        for i, (d, f, cl, ri, fr) in enumerate(ordered)
    )
    return FrontierIncidenceComplex(
        variables, formula, top_dimension, strata, tuple(incidence), certified
    )


__all__ = [
    "FrontierIncidenceComplex",
    "FrontierStratum",
    "LocalStratum",
    "LocalStratumProtocol",
    "as_local_stratum",
    "combine_local_strata",
    "frontier_incidence_complex",
]
