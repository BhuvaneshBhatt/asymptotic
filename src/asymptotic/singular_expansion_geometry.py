"""Certified orchestration for singular algebraic expansion geometry."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.core.function import PoleError

from .algebraic_curve_coverage import (
    BivariateCurveDecomposition,
    certified_bivariate_curve_decomposition,
    valuation_from_approach,
    weighted_atlases_for_decomposition,
)
from .blowup_geometry import BlowUpAtlas, ProjectiveBlowUpAtlas, projective_blowup_atlas
from .coverage import CoverageCertificate
from .multivariate_expansion import (
    MultivariateAsymptoticExpansion,
    multivariate_expansion,
)
from .proof_obligations import ObligationKind, ProofObligation


@dataclass(frozen=True)
class SingularBranchExpansion:
    component: int
    branch: int
    parameter: sp.Symbol
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...]
    transformed_expression: sp.Expr
    series: sp.Expr
    valuation: tuple[int, int] | None
    exact_branch: bool


@dataclass(frozen=True)
class SingularChartExpansion:
    weights: tuple[int, ...]
    atlas: BlowUpAtlas
    expansion: MultivariateAsymptoticExpansion

    @property
    def certified(self) -> bool:
        return (
            self.atlas.certified
            and self.expansion.certified
            and self.expansion.uniform_remainder().certified
        )


@dataclass(frozen=True)
class SingularExpansionAtlas:
    expression: sp.Expr
    curve: sp.Expr
    variables: tuple[sp.Symbol, sp.Symbol]
    target: tuple[sp.Expr, sp.Expr]
    decomposition: BivariateCurveDecomposition
    branches: tuple[SingularBranchExpansion, ...]
    charts: tuple[SingularChartExpansion, ...]
    coverage: CoverageCertificate
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def certified(self) -> bool:
        return (
            self.coverage.certified
            and bool(self.branches)
            and all(c.certified for c in self.charts)
        )


@dataclass(frozen=True)
class ProjectiveInfinityExpansion:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    reciprocal_variables: tuple[sp.Symbol, ...]
    transformed_expression: sp.Expr
    projective_atlas: ProjectiveBlowUpAtlas
    expansion: MultivariateAsymptoticExpansion
    coverage: CoverageCertificate
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def certified(self) -> bool:
        return (
            self.coverage.certified
            and self.projective_atlas.certified
            and self.expansion.certified
        )


def _branch_series(expr, approach, order):
    substitutions = dict(approach.substitutions)
    transformed = sp.simplify(sp.sympify(expr).subs(substitutions, simultaneous=True))
    try:
        series = sp.series(transformed, approach.parameter, 0, int(order) + 1)
    except (TypeError, ValueError, NotImplementedError, PoleError):
        series = transformed
    return transformed, series


def singular_expansion_atlas(
    expr,
    curve,
    variables,
    target=(0, 0),
    *,
    order=4,
    domain=sp.S.true,
    radius=None,
):
    """Construct a certified branch/valuation/chart expansion atlas.

    Completeness is asserted only when exact local algebraic branches cover the
    curve germ and every valuation family used for chart expansion has a
    certified uniform remainder.
    """
    x, y = tuple(variables)
    x0, y0 = map(sp.sympify, target)
    expr, curve = sp.sympify(expr), sp.sympify(curve)
    decomposition = certified_bivariate_curve_decomposition(
        curve,
        y,
        x,
        point=x0,
        dependent_point=y0,
        domain=domain,
        terms=max(8, order + 2),
    )
    branches = []
    for ci, component in enumerate(decomposition.components):
        for bi, approach in enumerate(component.approaches):
            transformed, series = _branch_series(expr, approach, order)
            branches.append(
                SingularBranchExpansion(
                    ci,
                    bi,
                    approach.parameter,
                    approach.substitutions,
                    transformed,
                    series,
                    valuation_from_approach(approach, (x0, y0)),
                    approach.exact,
                )
            )

    atlases = weighted_atlases_for_decomposition(decomposition, (x, y), (x0, y0))
    charts = []
    for atlas in atlases:
        weights = atlas.charts[0].valuation.weights
        expansion = multivariate_expansion(
            expr,
            (x, y),
            (x0, y0),
            order=order,
            weights=weights,
            domain=domain,
            radius=radius,
        )
        charts.append(SingularChartExpansion(tuple(weights), atlas, expansion))

    obligations = []
    if not decomposition.certified:
        obligations.append(
            ProofObligation(
                ObligationKind.BRANCH_COVERAGE,
                "exact local algebraic branches do not completely cover the singular germ",
                "singular_expansion_atlas",
                str(curve),
            )
        )
    incomplete = tuple(c.weights for c in charts if not c.certified)
    if incomplete:
        obligations.append(
            ProofObligation(
                ObligationKind.THEOREM_PREREQUISITE,
                "one or more valuation charts lack a certified uniform remainder",
                "singular_expansion_atlas",
                str(incomplete),
            )
        )
    if decomposition.certified and branches and charts and not incomplete:
        coverage = CoverageCertificate.complete(
            "singular_algebraic_branch_and_weighted_chart_cover",
            "exact algebraic branches cover the curve germ and their valuation atlases have certified uniform expansions",
            tuple(range(len(branches))),
        )
    else:
        coverage = CoverageCertificate.partial(
            "singular_algebraic_branch_and_weighted_chart_cover",
            "singular expansion coverage is incomplete",
            tuple(i for i, b in enumerate(branches) if b.exact_branch),
            tuple(o.statement for o in obligations) or ("no_certified_chart_family",),
        )
    return SingularExpansionAtlas(
        expr,
        curve,
        (x, y),
        (x0, y0),
        decomposition,
        tuple(branches),
        tuple(charts),
        coverage,
        tuple(obligations),
    )


def projective_infinity_expansion(expr, variables, *, order=4, radius=None):
    """Expand a real multivariate expression at simultaneous infinity.

    The reciprocal coordinates ``u_i=1/x_i`` compactify the end to the origin;
    the standard projective atlas certifies directional coverage there.
    """
    variables = tuple(variables)
    reciprocal = tuple(sp.Symbol(f"_inv_{v.name}", real=True) for v in variables)
    substitutions = {v: 1 / u for v, u in zip(variables, reciprocal, strict=True)}
    transformed = sp.cancel(
        sp.together(sp.sympify(expr).subs(substitutions, simultaneous=True))
    )
    projective = projective_blowup_atlas(reciprocal, (0,) * len(reciprocal))
    expansion = multivariate_expansion(
        transformed, reciprocal, (0,) * len(reciprocal), order=order, radius=radius
    )
    obligations = ()
    if expansion.certified:
        coverage = CoverageCertificate.complete(
            "reciprocal_projective_infinity_cover",
            "reciprocal compactification and standard projective charts cover all real directions at simultaneous infinity",
            tuple(range(len(projective.charts))),
        )
    else:
        obligations = (
            ProofObligation(
                ObligationKind.PROJECTIVE_COVERAGE,
                "reciprocal expression lacks a certified uniform expansion on the compactified end",
                "projective_infinity_expansion",
                str(transformed),
            ),
        )
        coverage = CoverageCertificate.partial(
            "reciprocal_projective_infinity_cover",
            "projective geometry is complete but the expansion is not certified",
            (),
            ("uniform_remainder",),
        )
    return ProjectiveInfinityExpansion(
        sp.sympify(expr),
        variables,
        reciprocal,
        transformed,
        projective,
        expansion,
        coverage,
        obligations,
    )


__all__ = [
    "ProjectiveInfinityExpansion",
    "SingularBranchExpansion",
    "SingularChartExpansion",
    "SingularExpansionAtlas",
    "projective_infinity_expansion",
    "singular_expansion_atlas",
]
