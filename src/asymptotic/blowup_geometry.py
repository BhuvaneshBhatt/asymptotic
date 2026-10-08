"""Unified weighted blow-up and valuation geometry for local asymptotics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from math import gcd

import sympy as sp

from .chart_models import (
    CoordinateChart,
    chart_exceptional_domain,
    chart_pullback_domain,
    chart_transform,
)
from .coverage import CoverageCertificate


def primitive_weight(weights):
    """Canonical positive integral representative of a valuation ray."""
    values = tuple(int(w) for w in weights)
    if not values or any(w <= 0 for w in values):
        raise ValueError("blow-up weights must be positive integers")
    divisor = reduce(gcd, values)
    return tuple(w // divisor for w in values)


@dataclass(frozen=True)
class Valuation:
    """Monomial valuation induced by one positive weight vector."""

    variables: tuple[sp.Symbol, ...]
    weights: tuple[int, ...]

    def __post_init__(self):
        if len(self.variables) != len(self.weights):
            raise ValueError("one valuation weight is required per variable")
        object.__setattr__(self, "weights", primitive_weight(self.weights))

    def monomial(self, exponents):
        return sum(w * e for w, e in zip(self.weights, exponents, strict=True))

    def polynomial(self, expression):
        try:
            poly = sp.Poly(sp.expand(expression), *self.variables)
        except sp.PolynomialError:
            return None
        values = [
            self.monomial(mon) for mon, coefficient in poly.terms() if coefficient != 0
        ]
        return min(values) if values else sp.oo

    def coordinate_orders(self, expression):
        """Largest coordinate monomial dividing a polynomial germ."""
        try:
            poly = sp.Poly(sp.expand(expression), *self.variables)
        except sp.PolynomialError:
            return None
        terms = [(mon, coeff) for mon, coeff in poly.terms() if coeff != 0]
        if not terms:
            return None
        return tuple(
            min(mon[i] for mon, _ in terms) for i in range(len(self.variables))
        )

    def initial_form(self, expression):
        value = self.polynomial(expression)
        if value is None or value is sp.oo:
            return None
        poly = sp.Poly(sp.expand(expression), *self.variables)
        return sp.Add(
            *(
                coefficient
                * sp.Mul(*(v**e for v, e in zip(self.variables, mon, strict=True)))
                for mon, coefficient in poly.terms()
                if coefficient != 0 and self.monomial(mon) == value
            )
        )


@dataclass(frozen=True)
class BlowUpChart:
    """A local weighted blow-up chart ``x_i=a_i+r**w_i*u_i``."""

    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    valuation: Valuation
    radial_variable: sp.Symbol
    angular_variables: tuple[sp.Symbol, ...]
    angular_constraint: sp.Expr
    sector: sp.Expr = sp.S.true
    dominant_coordinate: int | None = None
    provider: str = "weighted_blowup"

    @property
    def weights(self):
        return self.valuation.weights

    @property
    def substitution(self):
        r = self.radial_variable
        return {
            variable: target + r**weight * angular
            for variable, target, weight, angular in zip(
                self.variables,
                self.target,
                self.weights,
                self.angular_variables,
                strict=True,
            )
        }

    @property
    def exceptional_divisor(self):
        return sp.Eq(self.radial_variable, 0)

    def transform(self, expression):
        return sp.cancel(
            sp.together(
                sp.sympify(expression).subs(self.substitution, simultaneous=True)
            )
        )

    def pullback_domain(self, domain):
        return sp.simplify(
            sp.sympify(domain).subs(self.substitution, simultaneous=True)
        )

    @property
    def exceptional_domain(self):
        return sp.And(self.angular_constraint, self.sector)


@dataclass(frozen=True)
class BlowUpAtlas:
    """Finite chart family together with an explicit approach-cover proof."""

    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    charts: tuple[BlowUpChart, ...]
    coverage: CoverageCertificate
    provider: str
    statement: str

    @property
    def coverage_certified(self):
        return self.coverage.certified

    @property
    def certified(self):
        return self.coverage.certified and bool(self.charts)


def weighted_spherical_atlas(variables, target, weights):
    """Construct the canonical max-coordinate atlas for one valuation ray."""
    variables = tuple(variables)
    target = tuple(map(sp.sympify, target))
    valuation = Valuation(variables, tuple(weights))
    r = sp.Symbol("r", positive=True)
    angular = tuple(
        sp.Symbol(f"_blowup_u{i}", real=True) for i in range(len(variables))
    )
    sphere = sp.Eq(sp.Add(*(u**2 for u in angular)), 1)
    charts = []
    for i, ui in enumerate(angular):
        sector = sp.And(
            *(sp.Ge(ui**2, uj**2) for j, uj in enumerate(angular) if j != i)
        )
        charts.append(
            BlowUpChart(variables, target, valuation, r, angular, sphere, sector, i)
        )
    return BlowUpAtlas(
        variables,
        target,
        tuple(charts),
        CoverageCertificate.complete(
            "weighted_spherical_max_coordinate_cover",
            "max-coordinate sectors exhaust the weighted spherical exceptional divisor",
            tuple(range(len(charts))),
        ),
        "weighted_blowup_geometry",
        "max-coordinate sectors cover the weighted spherical exceptional divisor",
    )


@dataclass(frozen=True)
class ProjectiveTransition:
    source_chart: int
    target_chart: int
    condition: sp.Expr
    substitutions: tuple[tuple[sp.Expr, sp.Expr], ...]


@dataclass(frozen=True)
class ProjectiveBlowUpAtlas:
    atlas: BlowUpAtlas
    transitions: tuple[ProjectiveTransition, ...]

    @property
    def charts(self):
        return self.atlas.charts

    @property
    def coverage(self):
        return self.atlas.coverage

    @property
    def certified(self):
        return self.atlas.certified


def projective_chart(variables, target, chart_coordinate=0):
    """Return one standard affine chart of real projective direction space."""
    variables = tuple(variables)
    target = tuple(map(sp.sympify, target))
    n = len(variables)
    if n < 2:
        raise ValueError("projective blow-up requires at least two variables")
    if not 0 <= chart_coordinate < n:
        raise ValueError("chart_coordinate is outside the variable range")
    valuation = Valuation(variables, (1,) * n)
    r = sp.Symbol("r", positive=True)
    angular = tuple(
        sp.S.One
        if j == chart_coordinate
        else sp.Symbol(f"_projective_u{chart_coordinate}_{j}", real=True)
        for j in range(n)
    )
    return BlowUpChart(
        variables,
        target,
        valuation,
        r,
        angular,
        sp.S.true,
        sp.S.true,
        chart_coordinate,
        "projective_blowup",
    )


def projective_blowup_atlas(variables, target=None):
    """Canonical affine atlas of RP^(n-1), with exact overlap transitions."""
    variables = tuple(variables)
    target = (
        tuple(sp.S.Zero for _ in variables)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    charts = tuple(
        projective_chart(variables, target, i) for i in range(len(variables))
    )
    transitions = []
    for i, source in enumerate(charts):
        for j, target_chart in enumerate(charts):
            if i == j:
                continue
            uj = source.angular_variables[j]
            substitutions = []
            # Target chart normalizes direction by source coordinate j.
            for k, target_u in enumerate(target_chart.angular_variables):
                if k == j:
                    continue
                value = sp.simplify(source.angular_variables[k] / uj)
                substitutions.append((target_u, value))
            transitions.append(
                ProjectiveTransition(i, j, sp.Ne(uj, 0), tuple(substitutions))
            )
    atlas = BlowUpAtlas(
        variables,
        target,
        charts,
        CoverageCertificate.complete(
            "real_projective_affine_cover",
            f"{len(charts)} standard affine charts exhaust RP^{len(variables) - 1}",
            tuple(range(len(charts))),
        ),
        "projective_blowup_geometry",
        "standard nonzero-coordinate charts cover every real projective direction",
    )
    return ProjectiveBlowUpAtlas(atlas, tuple(transitions))


def newton_weighted_atlases(expressions, variables, target=None):
    """Lift Newton fan representative valuations into the common chart model."""
    from .newton_geometry import newton_polyhedral_fan

    variables = tuple(variables)
    target = (
        tuple(sp.S.Zero for _ in variables)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    fan = newton_polyhedral_fan(expressions, variables, target)
    if fan is None or not fan.certified:
        return ()
    atlases = []
    seen = set()
    for cone in fan.cones:
        weight = tuple(cone.representative)
        if not weight or any(w <= 0 for w in weight):
            continue
        weight = primitive_weight(weight)
        if weight in seen:
            continue
        seen.add(weight)
        atlases.append(weighted_spherical_atlas(variables, target, weight))
    return tuple(atlases)


def _polynomial_support(poly_expr: sp.Expr, variables: tuple[sp.Symbol, ...]):
    """Return exponent vectors of a polynomial, or an empty tuple if unsupported."""
    try:
        poly = sp.Poly(sp.expand(poly_expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    return tuple(tuple(int(e) for e in monom) for monom in poly.monoms())


def newton_candidate_rays(expr, variables, target) -> tuple[tuple[int, ...], ...]:
    """Generate primitive positive integer rays normal to Newton support faces.

    In two variables these are the familiar Newton-edge normals.  In ``n``
    variables, a simplicial candidate face is determined by ``n`` affinely
    independent support points; the one-dimensional nullspace of their
    exponent differences gives its normal ray.  Only strictly positive rays
    are useful for finite-point blow-ups.
    """
    from functools import reduce
    from itertools import combinations
    from math import lcm

    shifted = sp.cancel(
        expr.subs(
            {v: v + a for v, a in zip(variables, target, strict=True)},
            simultaneous=True,
        )
    )
    num, den = sp.fraction(sp.together(shifted))
    supports = (
        _polynomial_support(num, variables),
        _polynomial_support(den, variables),
    )
    n = len(variables)
    weights: set[tuple[int, ...]] = set()

    def primitive_positive(vector):
        entries = [sp.Rational(v) for v in vector]
        if not entries or any(v == 0 for v in entries):
            return None
        if all(v < 0 for v in entries):
            entries = [-v for v in entries]
        if not all(v > 0 for v in entries):
            return None
        scale = reduce(lcm, (int(v.q) for v in entries), 1)
        ints = [int(v * scale) for v in entries]
        g = reduce(gcd, (abs(v) for v in ints))
        return tuple(v // g for v in ints)

    for support in supports:
        if n == 1:
            continue
        for face in combinations(support, n):
            base = sp.Matrix(face[0])
            differences = sp.Matrix.hstack(*(sp.Matrix(p) - base for p in face[1:])).T
            nullspace = differences.nullspace()
            if len(nullspace) != 1:
                continue
            weight = primitive_positive(nullspace[0])
            if weight is None or weight == (1,) * n:
                continue
            # It must support the selected points on a common exposed face:
            # all support points lie on one side of their common weighted degree.
            level = sum(a * w for a, w in zip(face[0], weight, strict=True))
            degrees = [
                sum(a * w for a, w in zip(point, weight, strict=True))
                for point in support
            ]
            if all(d >= level for d in degrees) or all(d <= level for d in degrees):
                weights.add(weight)
    return tuple(sorted(weights, key=lambda w: (sum(w), w)))


def newton_valuation_rays(expressions, variables, target=None):
    """Return representative positive valuation rays and their common coverage proof."""
    variables = tuple(variables)
    target = (
        tuple(sp.S.Zero for _ in variables)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    return _newton_valuation_rays_cached(
        tuple(map(sp.sympify, expressions)), variables, target
    )


@lru_cache(maxsize=256)
def _newton_valuation_rays_cached(expressions, variables, target):
    from .newton_geometry import newton_polyhedral_fan

    fan = newton_polyhedral_fan(expressions, variables, target)
    rays = tuple(
        sorted(
            {
                primitive_weight(c.representative_weight)
                for c in fan.cones
                if c.representative_weight is not None
                and all(w > 0 for w in c.representative_weight)
            },
            key=lambda w: (sum(w), w),
        )
    )
    coverage = (
        CoverageCertificate.complete(
            "newton_polyhedral_fan",
            "representative valuation rays come from a certified common positive Newton fan",
            rays,
        )
        if fan.coverage_certified and rays
        else CoverageCertificate.partial(
            "newton_polyhedral_fan",
            "Newton fan coverage was not certified",
            rays,
            ("positive_weight_space",),
        )
    )
    return rays, coverage, fan


__all__ = [
    "BlowUpAtlas",
    "BlowUpChart",
    "CoordinateChart",
    "CoverageCertificate",
    "Valuation",
    "chart_exceptional_domain",
    "chart_pullback_domain",
    "chart_transform",
    "newton_candidate_rays",
    "newton_valuation_rays",
    "newton_weighted_atlases",
    "primitive_weight",
    "projective_chart",
    "weighted_spherical_atlas",
]
