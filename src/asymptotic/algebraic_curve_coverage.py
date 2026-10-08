"""Certified local decomposition and valuation atlases for bivariate curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import sympy as sp

from .blowup_geometry import BlowUpAtlas, weighted_spherical_atlas
from .coverage import CoverageCertificate
from .puiseux import NormalizedAlgebraicApproach, normalize_algebraic_approaches


@dataclass(frozen=True)
class CurveComponent:
    polynomial: sp.Expr
    multiplicity: int
    approaches: tuple[NormalizedAlgebraicApproach, ...]
    coverage: CoverageCertificate


@dataclass(frozen=True)
class BivariateCurveDecomposition:
    polynomial: sp.Expr
    components: tuple[CurveComponent, ...]
    coverage: CoverageCertificate

    @property
    def certified(self) -> bool:
        return self.coverage.certified


def _squarefree_factors(polynomial, x, y):
    poly = sp.Poly(sp.expand(polynomial), y, x)
    _, factors = sp.factor_list(poly)
    return tuple((f.as_expr(), int(m)) for f, m in factors)


def certified_bivariate_curve_decomposition(
    polynomial,
    dependent,
    variable,
    *,
    point=0,
    dependent_point=0,
    domain=sp.S.true,
    terms=8,
):
    """Decompose a square-free bivariate algebraic germ into covered components.

    Completeness is asserted only when exact factorization succeeds and each
    factor's local dependent-variable roots are isolated exactly by the Puiseux
    normalization layer. Repeated factors are recorded but do not create new
    geometric branches.
    """
    polynomial = sp.sympify(polynomial)
    try:
        factors = _squarefree_factors(polynomial, variable, dependent)
    except (sp.PolynomialError, TypeError, ValueError):
        return BivariateCurveDecomposition(
            polynomial,
            (),
            CoverageCertificate.unknown(
                "bivariate_curve_factorization",
                "polynomial factorization unavailable",
                ("factorization",),
            ),
        )
    components = []
    for factor, multiplicity in factors:
        try:
            approaches = normalize_algebraic_approaches(
                factor,
                dependent,
                variable,
                point=point,
                dependent_point=dependent_point,
                domain=domain,
                terms=terms,
            )
        except (NotImplementedError, sp.PolynomialError, TypeError, ValueError):
            approaches = ()
        cert = (
            approaches[0].coverage
            if approaches
            else CoverageCertificate.partial(
                "exact_algebraic_branch_cover",
                "no exact local branch isolation",
                (),
                ("unisolated_component",),
            )
        )
        components.append(CurveComponent(factor, multiplicity, approaches, cert))
    coverage = (
        CoverageCertificate.complete(
            "squarefree_bivariate_curve_cover",
            "exact polynomial factorization and every local factor has complete exact branch coverage",
            tuple(range(len(components))),
        )
        if components and all(c.coverage.certified for c in components)
        else CoverageCertificate.partial(
            "squarefree_bivariate_curve_cover",
            "some algebraic factor lacks complete exact local branch coverage",
            tuple(i for i, c in enumerate(components) if c.coverage.certified),
            tuple(i for i, c in enumerate(components) if not c.coverage.certified),
        )
    )
    return BivariateCurveDecomposition(polynomial, tuple(components), coverage)


def valuation_from_approach(
    approach: NormalizedAlgebraicApproach, target=None
) -> tuple[int, int] | None:
    """Extract primitive local-coordinate valuation from a certified ramified approach."""
    if not approach.exact:
        return None
    t = approach.parameter
    subs = dict(approach.substitutions)
    vals = []
    target_values = (
        tuple(sp.S.Zero for _ in subs)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    if len(target_values) != len(subs):
        return None
    for expr, center in zip(subs.values(), target_values, strict=True):
        try:
            p = sp.Poly(sp.expand(expr - center), t)
            powers = [mon[0] for mon, c in p.terms() if c != 0]
        except sp.PolynomialError:
            return None
        if not powers:
            return None
        vals.append(min(powers))
    if len(vals) != 2 or any(v <= 0 for v in vals):
        return None
    d = gcd(int(vals[0]), int(vals[1]))
    return (int(vals[0]) // d, int(vals[1]) // d)


def weighted_atlases_for_decomposition(
    decomposition, variables, target
) -> tuple[BlowUpAtlas, ...]:
    """Build covered weighted spherical atlases for certified branch valuations."""
    if not decomposition.certified:
        return ()
    weights = []
    for comp in decomposition.components:
        for approach in comp.approaches:
            w = valuation_from_approach(approach, target)
            if w and w not in weights:
                weights.append(w)
    return tuple(weighted_spherical_atlas(variables, target, w) for w in weights)
