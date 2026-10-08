"""Mapped local strata, certified fiber products, and induced frontier maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .local_strata import FrontierIncidenceComplex, as_local_stratum


@dataclass(frozen=True)
class StratumMap:
    """Explicit map from local base/fiber coordinates to cluster-value space."""

    source_variables: tuple[sp.Symbol, ...]
    expressions: tuple[sp.Expr, ...]
    target_variables: tuple[sp.Symbol, ...]
    source_constraint: sp.Expr = sp.S.true
    certified: bool = True
    provider: str = "explicit"


@dataclass(frozen=True)
class FiberProductSpec:
    """Dependency semantics for combining local base and fiber strata.

    ``product`` means independent coordinates; ``fiber_product`` and ``pullback``
    impose the supplied dependency relation; ``union`` represents alternatives.
    """

    mode: str
    relation: sp.Expr = sp.S.true
    certified: bool = True

    def __post_init__(self):
        if self.mode not in {"product", "fiber_product", "pullback", "union"}:
            raise ValueError(
                "fiber-product mode must be product, fiber_product, pullback, or union"
            )


@dataclass(frozen=True)
class MappedLocalStratum:
    """A local stratum equipped with fibers, dependency relation, and image map."""

    base: Any
    fibers: tuple[Any, ...]
    mapping: StratumMap
    fiber_product: FiberProductSpec = FiberProductSpec("product")
    label: str = "mapped_stratum"

    @property
    def stratum_kind(self):
        return "mapped_stratum"

    @property
    def local_constraint(self):
        pieces = [as_local_stratum(self.base).local_constraint]
        pieces.extend(as_local_stratum(f).local_constraint for f in self.fibers)
        pieces.extend((self.mapping.source_constraint, self.fiber_product.relation))
        return sp.And(*pieces)

    @property
    def intrinsic_dimension(self):
        dims = [as_local_stratum(self.base).intrinsic_dimension]
        dims.extend(as_local_stratum(f).intrinsic_dimension for f in self.fibers)
        if self.fiber_product.mode == "product" and all(d is not None for d in dims):
            return sum(dims)
        return None

    @property
    def valuation_data(self):
        return as_local_stratum(self.base).valuation_data

    @property
    def child_strata(self):
        return (self.base, *self.fibers)

    @property
    def stratum_certified(self):
        return (
            self.mapping.certified
            and self.fiber_product.certified
            and all(as_local_stratum(v).stratum_certified for v in self.child_strata)
        )


@dataclass(frozen=True)
class InducedFrontierStratumMap:
    """Image of one source frontier stratum under a certified stratum map."""

    source_index: int
    source_dimension: int
    image_formula: sp.Expr | None
    image_dimension: int | None
    collapsed: bool
    certified: bool


@dataclass(frozen=True)
class InducedFrontierMap:
    """Functorial image of a frontier/incidence complex under a local map."""

    source: FrontierIncidenceComplex
    strata: tuple[InducedFrontierStratumMap, ...]
    image_incidence: tuple[tuple[int, int], ...]
    coincident_images: tuple[tuple[int, int], ...]
    certified: bool


def mapped_local_stratum(
    base: Any,
    mapping: StratumMap,
    *fibers: Any,
    relation=True,
    mode: str = "product",
    label: str = "mapped_stratum",
) -> MappedLocalStratum:
    """Construct a mapped base/fiber stratum with explicit dependency semantics."""
    return MappedLocalStratum(
        base,
        tuple(fibers),
        mapping,
        FiberProductSpec(mode, sp.sympify(relation), True),
        label,
    )


def _image_formula(mapping: StratumMap, source_formula: sp.Expr):
    equations = [
        sp.Eq(y, expression)
        for y, expression in zip(
            mapping.target_variables, mapping.expressions, strict=True
        )
    ]
    formula = sp.And(source_formula, mapping.source_constraint, *equations)
    eliminated = tuple(v for v in mapping.source_variables if v in formula.free_symbols)
    if not eliminated:
        return sp.simplify(formula)
    try:
        from semialg import quantifier_eliminate

        return sp.simplify(
            quantifier_eliminate(
                formula,
                quantifiers=[*(("exists", v) for v in eliminated)],
                variables=(*mapping.target_variables, *eliminated),
            )
        )
    except (
        ImportError,
        ModuleNotFoundError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
        sp.PolynomialError,
    ):
        return None


def mapped_image_set(mapped: MappedLocalStratum) -> sp.Set | None:
    """Return the exact parametric image of a mapped base/fiber stratum.

    The source is retained as a ConditionSet instead of forcing QE.  This is
    exact, preserves base/fiber dependencies, and lets downstream geometry
    choose elimination only when an implicit image is actually required.
    """
    mapping = mapped.mapping
    if not mapping.source_variables:
        return sp.FiniteSet(*(mapping.expressions or ()))
    params = (
        mapping.source_variables[0]
        if len(mapping.source_variables) == 1
        else sp.Tuple(*mapping.source_variables)
    )
    body = (
        mapping.expressions[0]
        if len(mapping.expressions) == 1
        else sp.Tuple(*mapping.expressions)
    )
    universe = (
        sp.S.Reals
        if len(mapping.source_variables) == 1
        else sp.ProductSet(*([sp.S.Reals] * len(mapping.source_variables)))
    )
    source = sp.ConditionSet(params, mapped.local_constraint, universe)
    return sp.ImageSet(sp.Lambda((params,), body), source)


def induced_frontier_map(
    frontier: FrontierIncidenceComplex,
    mapping: StratumMap,
) -> InducedFrontierMap:
    """Map a frontier complex, detecting dimension collapse and coincident images."""
    mapped = []
    certified = frontier.certified and mapping.certified
    for stratum in frontier.strata:
        if not any(
            expr.free_symbols & set(mapping.source_variables)
            for expr in mapping.expressions
        ):
            formula = sp.And(
                *(
                    sp.Eq(y, expr)
                    for y, expr in zip(
                        mapping.target_variables, mapping.expressions, strict=True
                    )
                )
            )
            dim = 0
            ok = True
        else:
            formula = _image_formula(mapping, stratum.formula)
            dim = None
            ok = formula is not None
        if ok and dim is None:
            try:
                from semialg import region_dimension

                dim = int(region_dimension(formula, mapping.target_variables))
            except (
                ImportError,
                ModuleNotFoundError,
                ArithmeticError,
                TypeError,
                ValueError,
                NotImplementedError,
                sp.PolynomialError,
            ):
                ok = False
        certified = certified and ok
        mapped.append(
            InducedFrontierStratumMap(
                stratum.index,
                stratum.dimension,
                formula,
                dim,
                dim is not None and dim < stratum.dimension,
                ok,
            )
        )
    image_incidence = []
    for a, b in frontier.incidence:
        if mapped[a].image_formula is not None and mapped[b].image_formula is not None:
            image_incidence.append((a, b))
    coincident = []
    for i in range(len(mapped)):
        fi = mapped[i].image_formula
        if fi is None:
            continue
        for j in range(i + 1, len(mapped)):
            fj = mapped[j].image_formula
            if fj is not None and sp.simplify(sp.Equivalent(fi, fj)) is sp.S.true:
                coincident.append((i, j))
    return InducedFrontierMap(
        frontier,
        tuple(mapped),
        tuple(image_incidence),
        tuple(coincident),
        certified,
    )


__all__ = [
    "FiberProductSpec",
    "InducedFrontierMap",
    "InducedFrontierStratumMap",
    "MappedLocalStratum",
    "StratumMap",
    "induced_frontier_map",
    "mapped_image_set",
    "mapped_local_stratum",
]
