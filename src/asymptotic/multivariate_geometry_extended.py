"""Phase, compactified valuation, and intrinsic complex Newton geometry."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .coverage import CoverageCertificate
from .multivariate_limit_models import AdvancedLimitStatus


@dataclass(frozen=True)
class PhaseRelation:
    """Certified asymptotic relation between two oscillatory phases."""

    left: int
    right: int
    offset: sp.Expr
    provider: str


@dataclass(frozen=True)
class PhaseGeometryResult:
    """Finite-dimensional torus model for dependent and independent phases."""

    phases: tuple[sp.Expr, ...]
    independent_phases: tuple[int, ...]
    relations: tuple[PhaseRelation, ...]
    phase_variables: tuple[tuple[sp.Symbol, sp.Symbol], ...]
    constraint: sp.Expr
    status: AdvancedLimitStatus
    provider: str

    @property
    def certified(self):
        return self.status is AdvancedLimitStatus.CERTIFIED

    @property
    def stratum_kind(self):
        return "phase_torus"

    @property
    def local_constraint(self):
        return self.constraint

    @property
    def intrinsic_dimension(self):
        return len(self.independent_phases)

    @property
    def valuation_data(self):
        return self.relations

    @property
    def child_strata(self):
        return ()

    @property
    def stratum_certified(self):
        return self.certified


@dataclass(frozen=True)
class CompactifiedValuationChart:
    """One finite/projective chart together with its local Newton fan."""

    target: tuple[sp.Expr, ...]
    expression: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]
    finite_target: tuple[sp.Expr, ...]
    domain: sp.Expr
    fan: object
    end_signature: tuple[int, ...]

    @property
    def stratum_kind(self):
        return "projective_chart"

    @property
    def local_constraint(self):
        return self.domain

    @property
    def intrinsic_dimension(self):
        return len(self.variables)

    @property
    def valuation_data(self):
        return self.end_signature

    @property
    def child_strata(self):
        return tuple(self.fan.cones) if self.fan is not None else ()

    @property
    def stratum_certified(self):
        return bool(self.fan and self.fan.coverage_certified)


@dataclass(frozen=True)
class CompactifiedValuationAtlas:
    """A common atlas joining projective ends to local Newton valuation fans."""

    variables: tuple[sp.Symbol, ...]
    charts: tuple[CompactifiedValuationChart, ...]
    coverage: CoverageCertificate
    adjacency: tuple[tuple[int, int], ...]

    @property
    def coverage_certified(self):
        return self.coverage.certified


@dataclass(frozen=True)
class ComplexDivisorValuation:
    """Orders of a meromorphic germ along one coordinate or Newton divisor."""

    weight: tuple[int, ...]
    numerator_order: int
    denominator_order: int
    order: int
    initial_numerator: sp.Expr
    initial_denominator: sp.Expr

    @property
    def stratum_kind(self):
        return "complex_divisor"

    @property
    def local_constraint(self):
        return sp.S.true

    @property
    def intrinsic_dimension(self):
        return None

    @property
    def valuation_data(self):
        return self.weight, self.order

    @property
    def child_strata(self):
        return ()

    @property
    def stratum_certified(self):
        return True


@dataclass(frozen=True)
class ComplexNewtonGeometryResult:
    """Intrinsic complex Newton/divisor data for a rational meromorphic germ."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    valuations: tuple[ComplexDivisorValuation, ...]
    classification: str
    value: sp.Expr | None
    status: AdvancedLimitStatus
    fan: object | None
    provider: str

    @property
    def certified(self):
        return self.status is AdvancedLimitStatus.CERTIFIED


def _finite_phase_difference(a, b, variables, target, domain):
    from .multivariate_limits_advanced import certify_local_germ

    germ = certify_local_germ(sp.simplify(a - b), variables, target, domain=domain)
    if germ.certified and sp.sympify(germ.limit).is_finite is True:
        return sp.simplify(germ.limit)
    return None


def phase_geometry(expressions, variables, target, *, domain=True):
    """Build a certified torus model for several asymptotically related phases.

    Phases whose difference has a certified finite limit share one circle
    coordinate with the appropriate rotation.  Unrelated recognized divergent
    phases receive independent circle coordinates, yielding products of circles.
    """
    from .limit_primitives import _normalize_variables, normalize_limit_target
    from .multivariate_limits_advanced import _positive_radial_phase_denominator

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expressions = tuple(map(sp.sympify, expressions))
    atoms = sorted(
        set().union(*(e.atoms(sp.sin, sp.cos) for e in expressions)),
        key=sp.default_sort_key,
    )
    phases = []
    for atom in atoms:
        p = sp.simplify(atom.args[0])
        if p not in phases:
            phases.append(p)
    if not phases:
        return PhaseGeometryResult(
            (), (), (), (), sp.S.true, AdvancedLimitStatus.CERTIFIED, "no_phases"
        )
    recognized = []
    for p in phases:
        num, den = sp.fraction(sp.cancel(p))
        recognized.append(
            sp.simplify(num) in (1, -1)
            and _positive_radial_phase_denominator(den, variables, target)
        )
    # A phase may also be recognized through finite offset from a recognized one.
    parent = list(range(len(phases)))
    offsets = [sp.S.Zero] * len(phases)
    relations = []
    for j in range(len(phases)):
        for i in range(j):
            d = _finite_phase_difference(
                phases[j], phases[i], variables, target, domain
            )
            if d is not None and (recognized[i] or recognized[j]):
                parent[j] = parent[i]
                offsets[j] = sp.simplify(d + offsets[i])
                recognized[j] = True
                relations.append(PhaseRelation(j, i, d, "finite_phase_difference"))
                break
    if not all(recognized):
        return PhaseGeometryResult(
            tuple(phases),
            (),
            tuple(relations),
            (),
            sp.S.true,
            AdvancedLimitStatus.UNKNOWN,
            "unresolved_phase",
        )
    roots = []
    for i in range(len(phases)):
        r = parent[i]
        if r not in roots:
            roots.append(r)
    pairs = tuple(
        (sp.Dummy(f"_phase_c{k}", real=True), sp.Dummy(f"_phase_s{k}", real=True))
        for k in range(len(roots))
    )
    constraint = sp.And(*(sp.Eq(c * c + s * s, 1) for c, s in pairs))
    return PhaseGeometryResult(
        tuple(phases),
        tuple(roots),
        tuple(relations),
        pairs,
        constraint,
        AdvancedLimitStatus.CERTIFIED,
        "phase_relation_torus",
    )


def periodic_phase_cluster_geometry(expressions, variables, target, *, domain=True):
    """Project periodic expressions and convergent amplitudes over a phase torus.

    Expressions are treated as polynomials in their sine/cosine atoms.  Every
    coefficient is certified as a finite local germ, so amplitude × phase
    bundles are handled without destroying correlations between phase factors.
    """
    from .multivariate_limits_advanced import (
        _condition_set,
        _planned_image_projection,
        certify_local_germ,
    )

    variables = (
        tuple(variables) if isinstance(variables, (tuple, list)) else (variables,)
    )
    target = (
        tuple(target)
        if isinstance(target, (tuple, list))
        else (target,) * len(variables)
    )
    expressions = tuple(map(sp.sympify, expressions))
    pg = phase_geometry(expressions, variables, target, domain=domain)
    if not pg.certified:
        return None
    roots = list(pg.independent_phases)
    phase_to_pair = {}
    for p in pg.phases:
        for k, r in enumerate(roots):
            d = _finite_phase_difference(p, pg.phases[r], variables, target, domain)
            if d is not None:
                phase_to_pair[p] = (pg.phase_variables[k], d)
                break
    atoms = sorted(
        set().union(*(e.atoms(sp.sin, sp.cos) for e in expressions)),
        key=sp.default_sort_key,
    )
    placeholders = {
        a: sp.Dummy(f"_osc_atom{i}", real=True) for i, a in enumerate(atoms)
    }
    atom_values = {}
    for atom, h in placeholders.items():
        pair = phase_to_pair.get(sp.simplify(atom.args[0]))
        if pair is None:
            return None
        (c, s), d = pair
        atom_values[h] = (
            sp.simplify(sp.sin(d) * c + sp.cos(d) * s)
            if atom.func is sp.sin
            else sp.simplify(sp.cos(d) * c - sp.sin(d) * s)
        )
    lifted = []
    for e in expressions:
        raw = sp.expand(e.xreplace(placeholders))
        try:
            poly = sp.Poly(raw, *placeholders.values())
        except sp.PolynomialError:
            return None
        rebuilt = sp.S.Zero
        for monom, coeff in poly.terms():
            germ = certify_local_germ(coeff, variables, target, domain=domain)
            if not germ.certified or sp.sympify(germ.limit).is_finite is not True:
                return None
            term = sp.sympify(germ.limit)
            for h, k in zip(placeholders.values(), monom, strict=True):
                term *= atom_values[h] ** k
            rebuilt += term
        lifted.append(sp.simplify(rebuilt))
    source = tuple(v for pair in pg.phase_variables for v in pair)
    out = tuple(sp.Dummy(f"_phase_out{i}", real=True) for i in range(len(lifted)))
    graph = sp.And(
        *(sp.Eq(y, e) for y, e in zip(out, lifted, strict=True)), pg.constraint
    )
    planned = _planned_image_projection(graph, out, source)
    if planned is None:
        return None
    cond, provider = planned
    return (
        _condition_set(out, sp.simplify(cond)),
        pg,
        f"phase_torus_projection:{provider}",
    )


def compactified_valuation_atlas(expressions, variables, targets, *, domain=True):
    """Join finite/projective charts and Newton fans into one valuation atlas."""
    from .limit_primitives import _normalize_variables
    from .multivariate_limits_advanced import (
        _normalize_extended_target,
        projective_limit_chart,
    )
    from .newton_geometry import newton_polyhedral_fan

    variables = _normalize_variables(variables)
    expressions = tuple(map(sp.sympify, expressions))
    if targets and not isinstance(targets[0], (tuple, list)) and len(variables) == 1:
        targets = (targets,)
    charts = []
    for raw in targets:
        tgt = _normalize_extended_target(variables, raw)
        # transform all expressions with the same projective coordinate chart
        first = projective_limit_chart(expressions[0], variables, tgt, domain=domain)
        transformed = [first.expression]
        for e in expressions[1:]:
            transformed.append(
                projective_limit_chart(e, variables, tgt, domain=domain).expression
            )
        fan = newton_polyhedral_fan(tuple(transformed), first.variables, first.target)
        sig = tuple(1 if t is sp.oo else -1 if t is -sp.oo else 0 for t in tgt)
        charts.append(
            CompactifiedValuationChart(
                tgt,
                tuple(transformed),
                first.variables,
                first.target,
                first.domain,
                fan,
                sig,
            )
        )
    # projective charts with different signed ends meet only on directional boundary;
    # record combinatorial adjacency when they differ in one end sign/status.
    edges = []
    for i, a in enumerate(charts):
        for j, b in enumerate(charts[i + 1 :], i + 1):
            if (
                sum(
                    x != y
                    for x, y in zip(a.end_signature, b.end_signature, strict=True)
                )
                == 1
            ):
                edges.append((i, j))
    return CompactifiedValuationAtlas(
        variables,
        tuple(charts),
        (
            CoverageCertificate.complete(
                "compactified_valuation_atlas",
                "all requested projective charts have complete local Newton-fan coverage",
                tuple(c.end_signature for c in charts),
            )
            if charts and all(c.fan.coverage_certified for c in charts)
            else CoverageCertificate.partial(
                "compactified_valuation_atlas",
                "one or more projective charts lack complete local Newton-fan coverage",
                tuple(c.end_signature for c in charts if c.fan.coverage_certified),
                tuple(c.end_signature for c in charts if not c.fan.coverage_certified),
            )
        ),
        tuple(edges),
    )


def _weighted_initial(poly, variables, weight):
    p = sp.Poly(sp.expand(poly), *variables)
    terms = p.terms()
    order = min(sum(a * w for a, w in zip(m, weight, strict=True)) for m, _ in terms)
    init = sp.Add(
        *(
            c * sp.Mul(*(v**a for v, a in zip(variables, m, strict=True)))
            for m, c in terms
            if sum(a * w for a, w in zip(m, weight, strict=True)) == order
        )
    )
    return int(order), sp.expand(init)


def complex_newton_geometry(expr, variables, target=None):
    """Compute intrinsic multivariate complex divisor/Newton valuations.

    No real/imaginary splitting is used.  Every Newton-fan cone representative
    defines a monomial divisor valuation; numerator and denominator initial
    forms determine zero, pole, removable, or direction-dependent behavior.
    """
    from .newton_geometry import newton_polyhedral_fan

    variables = (
        tuple(variables) if isinstance(variables, (tuple, list)) else (variables,)
    )
    if target is None:
        target = (sp.S.Zero,) * len(variables)
    elif not isinstance(target, (tuple, list)):
        target = (target,) * len(variables)
    target = tuple(map(sp.sympify, target))
    shift = {v: v + a for v, a in zip(variables, target, strict=True)}
    e = sp.cancel(sp.together(sp.sympify(expr).subs(shift, simultaneous=True)))
    num, den = map(sp.expand, sp.fraction(e))
    try:
        sp.Poly(num, *variables)
        sp.Poly(den, *variables)
    except sp.PolynomialError:
        return ComplexNewtonGeometryResult(
            sp.sympify(expr),
            variables,
            target,
            (),
            "unknown",
            None,
            AdvancedLimitStatus.UNKNOWN,
            None,
            "nonrational_complex_germ",
        )
    fan = newton_polyhedral_fan((e,), variables, (0,) * len(variables))
    vals = []
    for cone in fan.cones:
        w = cone.representative_weight
        if w is None:
            continue
        no, ni = _weighted_initial(num, variables, w)
        do, di = _weighted_initial(den, variables, w)
        vals.append(ComplexDivisorValuation(w, no, do, no - do, ni, di))
    orders = {v.order for v in vals}
    if vals and all(o > 0 for o in orders):
        cls, value, status = "newton_zero", sp.S.Zero, AdvancedLimitStatus.CERTIFIED
    elif vals and all(o < 0 for o in orders):
        cls, value, status = "newton_pole", sp.zoo, AdvancedLimitStatus.CERTIFIED
    elif vals and orders == {0}:
        ratios = {sp.cancel(v.initial_numerator / v.initial_denominator) for v in vals}
        constants = {r for r in ratios if not r.has(*variables)}
        if len(constants) == 1 and len(ratios) == 1:
            cls, value, status = (
                "newton_removable",
                next(iter(constants)),
                AdvancedLimitStatus.CERTIFIED,
            )
        else:
            cls, value, status = (
                "newton_direction_dependent",
                None,
                AdvancedLimitStatus.CERTIFIED,
            )
    else:
        cls, value, status = (
            "mixed_divisor_orders",
            None,
            AdvancedLimitStatus.CERTIFIED
            if vals and fan.coverage_certified
            else AdvancedLimitStatus.UNKNOWN,
        )
    return ComplexNewtonGeometryResult(
        sp.sympify(expr),
        variables,
        target,
        tuple(vals),
        cls,
        value,
        status,
        fan,
        "complex_newton_divisor_valuation",
    )


__all__ = [
    "CompactifiedValuationAtlas",
    "CompactifiedValuationChart",
    "ComplexDivisorValuation",
    "ComplexNewtonGeometryResult",
    "PhaseGeometryResult",
    "PhaseRelation",
    "compactified_valuation_atlas",
    "complex_newton_geometry",
    "periodic_phase_cluster_geometry",
    "phase_geometry",
]
