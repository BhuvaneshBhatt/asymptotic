"""Recursive evaluation of subsystem-neutral local strata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .local_strata import FrontierIncidenceComplex, LocalStratum, as_local_stratum


@dataclass(frozen=True)
class RecursiveFrontierNode:
    """One frontier stratum embedded in a recursive local-stratum tree."""

    path: tuple[int, ...]
    dimension: int | None
    formula: sp.Expr
    source_kind: str


@dataclass(frozen=True)
class RecursiveFrontierComplex:
    """Frontier/incidence data propagated through a recursive stratum tree."""

    nodes: tuple[RecursiveFrontierNode, ...]
    incidence: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    certified: bool


@dataclass(frozen=True)
class StratumEvaluationResult:
    """Limiting image and recursive fibers of one common local stratum."""

    stratum: LocalStratum
    limiting_image: sp.Set | None
    children: tuple[StratumEvaluationResult, ...]
    frontier_complex: RecursiveFrontierComplex | None
    induced_frontier_map: Any | None
    certified: bool
    provider: str
    statement: str
    stratified_mapping: Any | None = None
    completeness_certificate: Any | None = None


def _condition_set(variables, formula):
    variables = tuple(variables)
    if not variables:
        return sp.S.UniversalSet if formula is not sp.S.false else sp.S.EmptySet
    value = variables[0] if len(variables) == 1 else sp.Tuple(*variables)
    return sp.ConditionSet(
        value,
        formula,
        sp.S.Reals
        if len(variables) == 1
        else sp.ProductSet(*([sp.S.Reals] * len(variables))),
    )


def _payload_image(local: LocalStratum):
    payload = local.payload
    if payload is None:
        return None, None, "structural"
    # Cluster-image strata already own their exact limiting image.
    cluster_set = getattr(payload, "cluster_set", None)
    if isinstance(cluster_set, sp.Set):
        geometry = getattr(payload, "structured_geometry", None)
        return cluster_set, getattr(geometry, "frontier_complex", None), "cluster_image"
    # Structured semialgebraic strata expose ambient variables and a formula.
    variables = getattr(payload, "variables", None)
    formula = getattr(payload, "formula", None)
    if variables is not None and formula is not None:
        return _condition_set(variables, formula), None, "semialgebraic_image"
    # A phase torus is itself an exact compact fiber.
    phase_variables = getattr(payload, "phase_variables", None)
    constraint = getattr(payload, "constraint", None)
    if phase_variables is not None and constraint is not None:
        flat = tuple(v for pair in phase_variables for v in pair)
        return _condition_set(flat, constraint), None, "phase_fiber"
    # Positive complex divisor order has singleton finite limiting image 0.
    order = getattr(payload, "order", None)
    if local.stratum_kind == "complex_divisor" and order is not None and order > 0:
        return sp.FiniteSet(sp.S.Zero), None, "complex_divisor_zero"
    return None, None, "structural"


def _lift_frontier(frontier: FrontierIncidenceComplex | None, path, kind):
    if frontier is None:
        return (), (), True
    nodes = tuple(
        RecursiveFrontierNode(path + (s.index,), s.dimension, s.formula, kind)
        for s in frontier.strata
    )
    incidence = tuple((path + (a,), path + (b,)) for a, b in frontier.incidence)
    return nodes, incidence, frontier.certified


def _merge_frontiers(local_frontier, children, path, kind):
    nodes, incidence, certified = _lift_frontier(local_frontier, path, kind)
    nodes = list(nodes)
    incidence = list(incidence)
    for child in children:
        fc = child.frontier_complex
        if fc is not None:
            nodes.extend(fc.nodes)
            incidence.extend(fc.incidence)
            certified = certified and fc.certified
    if not nodes:
        return None
    return RecursiveFrontierComplex(
        tuple(nodes), tuple(dict.fromkeys(incidence)), certified
    )


def evaluate_local_stratum(
    value: Any, *, _path: tuple[int, ...] = ()
) -> StratumEvaluationResult:
    """Recursively evaluate a local stratum without subsystem-specific dispatch.

    Structural strata (Newton cones and projective charts) contribute valuation
    context; image-bearing children contribute the limiting image.  Fiber strata
    such as phase tori remain attached recursively instead of being flattened
    into an unsound Cartesian product.
    """
    local = as_local_stratum(value)
    children = tuple(
        evaluate_local_stratum(child, _path=_path + (i,))
        for i, child in enumerate(local.child_strata)
    )
    own_image, own_frontier, provider = _payload_image(local)
    induced = None
    if local.stratum_kind == "mapped_stratum" and local.payload is not None:
        from .mapped_strata import induced_frontier_map, mapped_image_set
        from .stratified_mapping import refine_mapped_stratum

        # Rank refinement is automatic for low-dimensional maps; higher-dimensional
        # phase/fiber bundles retain the exact parametric image and defer CAD.
        # Automatic refinement stays eager through three source dimensions.
        # Higher-dimensional callers can invoke refine_mapped_stratum directly;
        # its scheduler constructs exact rank formulas while deferring costly QE.
        refinement = (
            refine_mapped_stratum(local.payload)
            if len(local.payload.mapping.source_variables) <= 3
            else None
        )
        image = mapped_image_set(local.payload)
        provider = (
            "mapped_stratum_image" if image is not None else "mapped_stratum_unresolved"
        )
        # The base is child zero.  Map its already-certified frontier complex
        # when the payload exposes one directly; otherwise preserve recursive
        # frontier data without inventing a functorial image.
        base_payload = as_local_stratum(local.payload.base).payload
        geometry = getattr(base_payload, "structured_geometry", None)
        base_frontier = getattr(geometry, "frontier_complex", None)
        if base_frontier is None:
            base_frontier = getattr(base_payload, "frontier_complex", None)
        if base_frontier is not None:
            induced = induced_frontier_map(base_frontier, local.payload.mapping)
    else:
        refinement = None
        image = own_image
    if image is None:
        child_images = [
            c.limiting_image for c in children if c.limiting_image is not None
        ]
        # A structural node may transparently forward one image.  Multiple
        # children are a union only for explicit union/atlas nodes; otherwise
        # retain them as fibers because a Cartesian product would lose relations.
        if len(child_images) == 1:
            image = child_images[0]
            provider = "recursive_forward"
        elif child_images and local.stratum_kind in {
            "union",
            "atlas",
            "projective_atlas",
        }:
            image = sp.Union(*child_images)
            provider = "recursive_union"
    frontier = _merge_frontiers(own_frontier, children, _path, local.stratum_kind)
    certified = local.stratum_certified and all(c.certified for c in children)
    if image is None and local.stratum_kind in {
        "cluster_image",
        "semialgebraic_stratum",
    }:
        certified = False
    from .stratified_mapping import completeness_certificate

    certificate = completeness_certificate(value)
    certified = certified and certificate.certified
    return StratumEvaluationResult(
        local,
        image,
        children,
        frontier,
        induced,
        certified,
        provider,
        f"evaluated {local.stratum_kind} with {len(children)} recursive child strata",
        refinement,
        certificate,
    )


__all__ = [
    "RecursiveFrontierComplex",
    "RecursiveFrontierNode",
    "StratumEvaluationResult",
    "evaluate_local_stratum",
]
