"""Parameter-cell decomposition for local blow-up/projective geometry."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .blowup_geometry import newton_valuation_rays, projective_blowup_atlas
from .coverage import CoverageCertificate
from .parameter_cells import parameter_truth_cells


@dataclass(frozen=True)
class ParameterizedGeometryCell:
    condition: sp.Expr
    signature: tuple[bool, ...]
    projective_atlas: object
    valuation_rays: tuple[tuple[int, ...], ...]
    coverage: CoverageCertificate
    topology_predicates: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class ParameterizedGeometry:
    parameters: tuple[sp.Symbol, ...]
    cells: tuple[ParameterizedGeometryCell, ...]
    coverage: CoverageCertificate

    @property
    def certified(self):
        return self.coverage.certified and all(c.coverage.certified for c in self.cells)


def _homogeneous_polynomial(expr, variables):
    try:
        return sp.Poly(sp.expand(expr), *variables)
    except sp.PolynomialError:
        return None


def _projective_topology_predicates(expressions, variables, parameters):
    """Discriminants/resultants where projective pole/critical topology can change."""
    predicates = []
    if len(variables) == 2:
        x, y = variables
        t = sp.Dummy("_projective_cell_t", real=True)
        for expr in expressions:
            num, den = sp.fraction(sp.cancel(sp.together(expr)))
            p = sp.expand(num.subs({x: 1, y: t}))
            q = sp.expand(den.subs({x: 1, y: t}))
            try:
                qp = sp.Poly(q, t)
                if qp.degree() > 0:
                    disc = sp.factor(sp.discriminant(q, t))
                    if disc != 0 and disc.free_symbols & set(parameters):
                        predicates.append(sp.Eq(disc, 0))
                derivative_num = sp.fraction(sp.cancel(sp.diff(p / q, t)))[0]
                if sp.Poly(derivative_num, t).degree() > 0:
                    disc = sp.factor(sp.discriminant(derivative_num, t))
                    if disc != 0 and disc.free_symbols & set(parameters):
                        predicates.append(sp.Eq(disc, 0))
                resultant = sp.factor(sp.resultant(p, q, t))
                if resultant != 0 and resultant.free_symbols & set(parameters):
                    predicates.append(sp.Eq(resultant, 0))
            except (sp.PolynomialError, TypeError, ValueError):
                continue
    # Coefficients crossing zero change polynomial/Newton support in all dimensions.
    for expr in expressions:
        num, den = sp.fraction(sp.cancel(sp.together(expr)))
        for part in (num, den):
            poly = _homogeneous_polynomial(part, variables)
            if poly is None:
                continue
            for coeff in poly.coeffs():
                if coeff.free_symbols & set(parameters):
                    predicates.append(sp.Eq(sp.factor(coeff), 0))
    # Stable deterministic unique list.
    return tuple(dict.fromkeys(map(sp.sympify, predicates)))


def parameterized_geometry_cells(
    expressions, variables, target=None, *, condition=sp.S.true, parameters=None
):
    """Partition parameter space wherever local geometric topology may change."""
    expressions = tuple(map(sp.sympify, expressions))
    variables = tuple(variables)
    target = (
        tuple(sp.S.Zero for _ in variables)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    if parameters is None:
        parameters = tuple(
            sorted(
                set().union(*(e.free_symbols for e in expressions)) - set(variables),
                key=sp.default_sort_key,
            )
        )
    else:
        parameters = tuple(parameters)
    predicates = _projective_topology_predicates(expressions, variables, parameters)
    cells = parameter_truth_cells(predicates, parameters, condition)
    if not cells:
        return ParameterizedGeometry(
            parameters,
            (),
            CoverageCertificate.unknown(
                "parameterized_geometry", "parameter topology cells were not certified"
            ),
        )
    projective = projective_blowup_atlas(variables, target)
    out = []
    for cell in cells:
        # Specialize equalities that directly fix parameters when possible.
        specialized = tuple(sp.simplify(e) for e in expressions)
        rays, ray_coverage, _fan = newton_valuation_rays(specialized, variables, target)
        coverage = projective.coverage.combine(
            ray_coverage,
            provider="parameter_geometry_cell",
            statement="projective directions and Newton valuation regimes are covered on this exact parameter cell",
        )
        out.append(
            ParameterizedGeometryCell(
                cell.condition, cell.signature, projective, rays, coverage, predicates
            )
        )
    all_coverage = CoverageCertificate.complete(
        "parameter_truth_cell_cover",
        "Boolean truth cells exhaust the supplied semialgebraic parameter condition",
        tuple(c.condition for c in cells),
    )
    return ParameterizedGeometry(parameters, tuple(out), all_coverage)


__all__ = [
    "ParameterizedGeometry",
    "ParameterizedGeometryCell",
    "parameterized_geometry_cells",
]
