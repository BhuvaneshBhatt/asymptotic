"""Well-founded progress measures for recursive geometric resolution."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, order=True)
class ResolutionMeasure:
    dimension: int
    valuation_complexity: int
    singular_complexity: int


@dataclass(frozen=True)
class ResolutionProgressCertificate:
    parent: ResolutionMeasure
    child: ResolutionMeasure
    certified: bool
    reason: str


def valuation_complexity(expr, variables):
    try:
        num, den = sp.fraction(sp.cancel(expr))
        return sum(len(sp.Poly(e, *variables).terms()) for e in (num, den))
    except (sp.PolynomialError, ValueError, TypeError):
        return len(tuple(sp.preorder_traversal(expr)))


def progress_certificate(
    parent_dimension,
    child_dimension,
    expr,
    variables,
    parent_singular=1,
    child_singular=0,
):
    parent = ResolutionMeasure(
        int(parent_dimension),
        valuation_complexity(expr, variables),
        int(parent_singular),
    )
    child = ResolutionMeasure(
        int(child_dimension), valuation_complexity(expr, variables), int(child_singular)
    )
    # Dimension is the primary well-founded measure. Equal-dimensional steps
    # require a strict decrease in valuation or singular complexity.
    certified = child.dimension < parent.dimension or (
        child.dimension == parent.dimension
        and (child.valuation_complexity, child.singular_complexity)
        < (parent.valuation_complexity, parent.singular_complexity)
    )
    return ResolutionProgressCertificate(
        parent,
        child,
        certified,
        "strict lexicographic decrease in (dimension, valuation complexity, singular complexity)"
        if certified
        else "no certified well-founded decrease",
    )


__all__ = [
    "ResolutionMeasure",
    "ResolutionProgressCertificate",
    "progress_certificate",
    "valuation_complexity",
]
