"""Certified stratified mappings, completeness accounting, and valuation unification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import sympy as sp

from .coverage import CoverageCertificate
from .local_strata import as_local_stratum, frontier_incidence_complex
from .mapped_strata import MappedLocalStratum, StratumMap, mapped_local_stratum


@dataclass(frozen=True)
class ProjectionAttempt:
    """One scheduled exact image-projection attempt and its outcome."""

    method: str
    estimated_cost: int
    attempted: bool
    succeeded: bool
    reason: str


@dataclass(frozen=True)
class ProjectionSchedule:
    """Lazy exact-projection schedule ordered from substitution to full QE."""

    attempts: tuple[ProjectionAttempt, ...]
    budget: int
    exhausted: bool


@dataclass(frozen=True)
class RegularityRefinement:
    """Rank/frontier refinement with certified smooth-locus bookkeeping.

    This is Whitney-style rather than a claim of a full Whitney (a)/(b)
    stratification: constant-rank pieces and lower-dimensional singular/frontier
    loci are certified exactly, while unresolved tangent-limit conditions remain
    explicit in ``whitney_certified``.
    """

    smooth_rank_pieces: tuple[sp.Expr, ...]
    singular_locus: sp.Expr
    frontier_compatible: bool
    whitney_certified: bool
    certified: bool


@dataclass(frozen=True)
class ProofCertificateNode:
    """One composable proof node with explicit dependencies."""

    claim: str
    provider: str
    certified: bool
    dependencies: tuple[ProofCertificateNode, ...] = ()


@dataclass(frozen=True)
class RankRefinedImageStratum:
    """One source rank stratum together with its exact mapped image."""

    rank: int
    source_formula: sp.Expr
    image_formula: sp.Expr | None
    source_dimension: int | None
    image_dimension: int | None
    certified: bool
    projection_schedule: ProjectionSchedule | None = None


@dataclass(frozen=True)
class StratifiedMappingResult:
    """Jacobian-rank refinement, merged image strata, and rebuilt incidence."""

    mapped: MappedLocalStratum
    rank_strata: tuple[RankRefinedImageStratum, ...]
    image_variables: tuple[sp.Symbol, ...]
    image_formula: sp.Expr | None
    merged_image_strata: tuple[sp.Expr, ...]
    frontier_complex: Any | None
    certified: bool
    provider: str
    regularity: RegularityRefinement | None = None
    proof_certificate: ProofCertificateNode | None = None


@dataclass(frozen=True)
class CompletenessStep:
    """One inspectable exhaustion claim in a recursive cluster proof."""

    path: tuple[int, ...]
    kind: str
    certified: bool
    reason: str


@dataclass(frozen=True)
class ClusterCompletenessCertificate:
    """Inspectable accounting that every represented approach stratum is exhausted."""

    steps: tuple[CompletenessStep, ...]
    exhausted_kinds: tuple[str, ...]
    frontier_pieces: int
    coverage: CoverageCertificate
    statement: str
    proof_certificate: ProofCertificateNode | None = None

    @property
    def certified(self):
        return self.coverage.certified


@dataclass(frozen=True)
class ComplexClusterGeometryResult:
    """Intrinsic complex cluster geometry from Newton divisor initial quotients."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    cluster_set: sp.Set | None
    projective_cluster: str
    divisor_images: tuple[sp.Expr, ...]
    certified: bool
    provider: str


@dataclass(frozen=True)
class UnifiedValuationNode:
    """One node in the common finite/projective/complex valuation compactification."""

    index: int
    kind: str
    payload: Any
    faces: tuple[int, ...] = ()
    adjacent: tuple[int, ...] = ()
    certified: bool = True


@dataclass(frozen=True)
class UnifiedValuationCompactification:
    """Common valuation-space container for Newton, projective, and complex nodes."""

    nodes: tuple[UnifiedValuationNode, ...]
    adjacency: tuple[tuple[int, int], ...]
    coverage: CoverageCertificate

    @property
    def coverage_certified(self):
        return self.coverage.certified


def _minors(matrix: sp.Matrix, size: int):
    if size == 0:
        return (sp.S.One,)
    return tuple(
        sp.expand(matrix.extract(rows, cols).det())
        for rows in combinations(range(matrix.rows), size)
        for cols in combinations(range(matrix.cols), size)
    )


def _rank_formula(jacobian: sp.Matrix, rank: int):
    """Exact polynomial formula saying a Jacobian has rank exactly ``rank``."""
    max_rank = min(jacobian.rows, jacobian.cols)
    if rank < 0 or rank > max_rank:
        return sp.S.false
    upper = (
        sp.And(*[sp.Eq(m, 0) for m in _minors(jacobian, rank + 1)])
        if rank < max_rank
        else sp.S.true
    )
    lower = (
        sp.Or(*[sp.Ne(m, 0) for m in _minors(jacobian, rank)]) if rank else sp.S.true
    )
    return sp.And(upper, lower)


def _formula_cost(formula: sp.Expr, variables: tuple[sp.Symbol, ...]) -> int:
    """Cheap deterministic proxy for CAD/QE cost."""
    ops = int(sp.count_ops(formula, visual=False))
    atoms = len(formula.atoms(sp.Relational)) if hasattr(sp, "Relational") else 0
    return max(1, (ops + atoms + 1) * max(1, len(variables)) ** 2)


def _qe_image(mapping: StratumMap, source_formula: sp.Expr, *, budget: int = 5000):
    """Project an image lazily; return ``(formula, schedule)``.

    Cheap substitution/planned elimination is always attempted first.  Full QE
    is attempted only when the deterministic complexity estimate fits ``budget``.
    """
    equations = [
        sp.Eq(y, e)
        for y, e in zip(mapping.target_variables, mapping.expressions, strict=True)
    ]
    formula = sp.And(source_formula, mapping.source_constraint, *equations)
    # Eliminate source coordinates fixed by exact affine graph equations.
    # Apply each substitution to the whole formula so fiber relations retain
    # their dependence on the target coordinates.
    for variable in mapping.source_variables:
        for relation in sp.And.make_args(formula):
            if not isinstance(relation, sp.Equality):
                continue
            difference = sp.expand(relation.lhs - relation.rhs)
            coefficient = difference.coeff(variable)
            remainder = difference - coefficient * variable
            if coefficient in (1, -1) and not remainder.has(variable):
                formula = formula.subs(variable, -remainder / coefficient)
                break
    eliminated = tuple(v for v in mapping.source_variables if v in formula.free_symbols)
    attempts: list[ProjectionAttempt] = []
    if not eliminated:
        attempts.append(
            ProjectionAttempt(
                "no_elimination", 0, True, True, "no source variables remain"
            )
        )
        return sp.simplify(formula), ProjectionSchedule(tuple(attempts), budget, False)
    try:
        from .multivariate_limits_advanced import _planned_image_projection

        cost = _formula_cost(formula, eliminated)
        if cost > budget // 2:
            attempts.append(
                ProjectionAttempt(
                    "substitution_branch_qe",
                    cost,
                    False,
                    False,
                    "planner deferred by budget",
                )
            )
            planned = None
        else:
            planned = _planned_image_projection(
                formula, mapping.target_variables, eliminated
            )
        ok = planned is not None
        attempts.append(
            ProjectionAttempt(
                "substitution_branch_qe",
                cost,
                True,
                ok,
                "planner reduced image" if ok else "planner left residual projection",
            )
        )
        if ok:
            return sp.simplify(planned[0]), ProjectionSchedule(
                tuple(attempts), budget, False
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
    ) as exc:
        attempts.append(
            ProjectionAttempt(
                "substitution_branch_qe", 0, True, False, type(exc).__name__
            )
        )
    cost = _formula_cost(formula, (*mapping.target_variables, *eliminated))
    if cost > budget:
        attempts.append(
            ProjectionAttempt(
                "full_qe",
                cost,
                False,
                False,
                f"estimated cost {cost} exceeds budget {budget}",
            )
        )
        return None, ProjectionSchedule(tuple(attempts), budget, True)
    try:
        from semialg import quantifier_eliminate

        result = sp.simplify(
            quantifier_eliminate(
                formula,
                quantifiers=[*(("exists", v) for v in eliminated)],
                variables=(*mapping.target_variables, *eliminated),
            )
        )
        attempts.append(
            ProjectionAttempt("full_qe", cost, True, True, "exact QE completed")
        )
        return result, ProjectionSchedule(tuple(attempts), budget, False)
    except (
        ImportError,
        ModuleNotFoundError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
        sp.PolynomialError,
    ) as exc:
        attempts.append(
            ProjectionAttempt("full_qe", cost, True, False, type(exc).__name__)
        )
        return None, ProjectionSchedule(tuple(attempts), budget, True)


def refine_mapped_stratum(mapped: MappedLocalStratum) -> StratifiedMappingResult:
    """Split a mapped stratum where Jacobian rank changes and rebuild image topology."""
    mapping = mapped.mapping
    jac = sp.Matrix(mapping.expressions).jacobian(mapping.source_variables)
    max_rank = min(jac.rows, jac.cols)
    pieces = []
    certified = mapped.stratum_certified
    for rank in range(max_rank, -1, -1):
        source = sp.And(mapped.local_constraint, _rank_formula(jac, rank))
        try:
            from semialg import is_satisfiable, region_dimension

            if len(mapping.source_variables) > 3:
                # Keep the exact rank formula but defer expensive CAD topology.
                sdim = None
            else:
                if not is_satisfiable(source, mapping.source_variables):
                    continue
                sdim = int(region_dimension(source, mapping.source_variables))
        except (
            ImportError,
            ModuleNotFoundError,
            ArithmeticError,
            TypeError,
            ValueError,
            NotImplementedError,
            sp.PolynomialError,
        ):
            sdim = None
            certified = False
        if len(mapping.source_variables) > 3:
            schedule = ProjectionSchedule(
                (
                    ProjectionAttempt(
                        "full_qe",
                        _formula_cost(source, mapping.source_variables),
                        False,
                        False,
                        "higher-dimensional image retained parametrically",
                    ),
                ),
                5000,
                True,
            )
            image = None
        else:
            image, schedule = _cached_qe_image(mapping, source)
        idim = None
        ok = image is not None
        if ok:
            try:
                from semialg import region_dimension

                idim = int(region_dimension(image, mapping.target_variables))
            except (
                ImportError,
                ModuleNotFoundError,
                ArithmeticError,
                TypeError,
                ValueError,
                NotImplementedError,
                sp.PolynomialError,
                sp.polys.polyerrors.CoercionFailed,
            ):
                ok = False
        certified = certified and ok
        pieces.append(
            RankRefinedImageStratum(rank, source, image, sdim, idim, ok, schedule)
        )

    # Merge coincident image strata using exact logical equivalence when possible.
    merged: list[sp.Expr] = []
    for p in pieces:
        if p.image_formula is None:
            continue
        duplicate = False
        for q in merged:
            if sp.simplify(sp.Equivalent(p.image_formula, q)) is sp.S.true:
                duplicate = True
                break
        if not duplicate:
            merged.append(p.image_formula)
    image_formula = sp.Or(*merged) if merged else None
    frontier = None
    if image_formula is not None:
        try:
            frontier = frontier_incidence_complex(
                image_formula, mapping.target_variables
            )
        except (
            ImportError,
            ModuleNotFoundError,
            ArithmeticError,
            TypeError,
            ValueError,
            NotImplementedError,
            sp.PolynomialError,
        ):
            frontier = None
        certified = certified and frontier is not None and frontier.certified
    singular = (
        sp.Or(*(p.source_formula for p in pieces if p.rank < max_rank))
        if pieces
        else sp.S.false
    )
    smooth = tuple(
        p.source_formula for p in pieces if p.rank == max_rank and p.certified
    )
    frontier_ok = (
        bool(frontier is not None and frontier.certified)
        if image_formula is not None
        else False
    )
    regularity = RegularityRefinement(
        smooth, singular, frontier_ok, False, certified and frontier_ok
    )
    deps = tuple(
        ProofCertificateNode(
            f"rank-{p.rank} image projected", "lazy_projection_schedule", p.certified
        )
        for p in pieces
    )
    proof = ProofCertificateNode(
        "mapped image is exhausted by rank strata",
        "jacobian_rank_refinement",
        certified,
        deps,
    )
    return StratifiedMappingResult(
        mapped,
        tuple(pieces),
        mapping.target_variables,
        image_formula,
        tuple(merged),
        frontier,
        certified,
        "jacobian_rank_stratified_image",
        regularity,
        proof,
    )


def completeness_certificate(value: Any) -> ClusterCompletenessCertificate:
    """Build a recursive, inspectable exhaustion certificate for a local-stratum tree."""
    steps: list[CompletenessStep] = []
    frontier_count = 0

    def visit(obj: Any, path: tuple[int, ...]):
        nonlocal frontier_count
        local = as_local_stratum(obj)
        payload = local.payload
        reason = "local stratum is certified"
        ok = local.stratum_certified
        if local.stratum_kind == "newton_cone":
            reason = "Newton cone has an exact representative valuation"
        elif local.stratum_kind == "projective_chart":
            fan = getattr(payload, "fan", None)
            ok = ok and bool(fan and fan.coverage_certified)
            reason = (
                "projective chart is normalized to a Newton fan with certified coverage"
            )
        elif local.stratum_kind == "phase_torus":
            reason = "all certified independent phase circles and finite phase relations are represented"
        elif local.stratum_kind == "complex_divisor":
            reason = "complex divisor order and initial forms are exact"
        elif local.stratum_kind in {"semialgebraic_stratum", "cluster_image"}:
            reason = "semialgebraic limiting image stratum is certified"
        geometry = getattr(payload, "structured_geometry", None)
        fc = getattr(geometry, "frontier_complex", None) or getattr(
            payload, "frontier_complex", None
        )
        if fc is not None:
            frontier_count += len(fc.strata)
            ok = ok and fc.certified
        steps.append(CompletenessStep(path, local.stratum_kind, ok, reason))
        for i, child in enumerate(local.child_strata):
            visit(child, path + (i,))

    visit(value, ())
    kinds = tuple(sorted({s.kind for s in steps}))
    certified = all(s.certified for s in steps)
    leaves = tuple(
        ProofCertificateNode(
            f"exhaust stratum {step.path or ('root',)} ({step.kind})",
            "recursive_completeness",
            step.certified,
        )
        for step in steps
    )
    proof = ProofCertificateNode(
        "all represented approach strata and frontier pieces are exhausted",
        "recursive_completeness_composition",
        certified,
        leaves,
    )
    return ClusterCompletenessCertificate(
        tuple(steps),
        kinds,
        frontier_count,
        (
            CoverageCertificate.complete(
                "recursive_stratum_exhaustion",
                "all represented approach strata and frontier pieces are exhausted",
                tuple(step.path for step in steps),
            )
            if certified
            else CoverageCertificate.partial(
                "recursive_stratum_exhaustion",
                "one or more represented approach strata or frontier pieces are not exhausted",
                tuple(step.path for step in steps if step.certified),
                tuple(step.path for step in steps if not step.certified),
            )
        ),
        f"{'certified' if certified else 'incomplete'} exhaustion of {len(steps)} recursive strata and {frontier_count} frontier pieces",
        proof,
    )


def complex_cluster_geometry(
    expr, variables, target=None
) -> ComplexClusterGeometryResult:
    """Return intrinsic finite/projective complex cluster geometry from Newton divisors."""
    from .multivariate_geometry_extended import complex_newton_geometry

    g = complex_newton_geometry(expr, variables, target)
    images = []
    finite: sp.Set | None = None
    projective = "unknown"
    if not g.certified:
        return ComplexClusterGeometryResult(
            sp.sympify(expr),
            tuple(variables),
            None,
            projective,
            (),
            False,
            "complex_newton_unresolved",
        )
    if g.value is not None and g.value is not sp.zoo:
        finite = sp.FiniteSet(g.value)
        projective = "point"
    elif g.classification == "newton_pole":
        finite = sp.S.EmptySet
        projective = "infinity"
    else:
        for v in g.valuations:
            if v.order == 0:
                images.append(sp.cancel(v.initial_numerator / v.initial_denominator))
        # A nonconstant rational initial quotient on a complex projective divisor
        # has projective image CP^1; its finite cluster values are all C.
        if any(r.has(*g.variables) for r in images):
            finite = sp.S.Complexes
            projective = "riemann_sphere"
        elif images:
            vals = {sp.simplify(r) for r in images}
            finite = sp.FiniteSet(*vals)
            projective = "finite_union"
    return ComplexClusterGeometryResult(
        sp.sympify(expr),
        g.variables,
        finite,
        projective,
        tuple(images),
        finite is not None and g.fan is not None and g.fan.coverage_certified,
        "complex_newton_projective_cluster",
    )


def phase_bundle_over_stratum(
    base: Any, expressions, variables, target, *, domain=True
):
    """Attach the certified phase torus as a dependent fiber over a valuation stratum."""
    from .multivariate_geometry_extended import phase_geometry

    phase = phase_geometry(expressions, variables, target, domain=domain)
    if not phase.certified:
        return None
    flat = tuple(v for pair in phase.phase_variables for v in pair)
    targets = tuple(sp.Dummy(f"_phase_bundle_{i}", real=True) for i in range(len(flat)))
    mapping = StratumMap(
        flat, flat, targets, phase.constraint, True, "phase_bundle_identity"
    )
    return mapped_local_stratum(
        base,
        mapping,
        phase,
        relation=phase.constraint,
        mode="fiber_product",
        label="phase_bundle",
    )


def unified_valuation_compactification(
    *, fan=None, projective_atlas=None, complex_geometry=None
):
    """Combine finite Newton cones, projective charts, and complex divisors in one graph."""
    nodes: list[UnifiedValuationNode] = []
    edges: list[tuple[int, int]] = []
    certified = True

    def add(kind, payload, faces=(), adjacent=(), ok=True):
        idx = len(nodes)
        nodes.append(
            UnifiedValuationNode(idx, kind, payload, tuple(faces), tuple(adjacent), ok)
        )
        return idx

    if fan is not None:
        certified = certified and fan.coverage_certified and fan.adjacency_certified
        offset = len(nodes)
        for cone in fan.cones:
            add(
                "newton_cone",
                cone,
                (),
                tuple(offset + j for j in cone.adjacent),
                cone.stratum_certified,
            )
        for cone in fan.cones:
            for j in cone.adjacent:
                if cone.index < j:
                    edges.append((offset + cone.index, offset + j))
    if projective_atlas is not None:
        chart_ids = []
        for chart in projective_atlas.charts:
            cid = add("projective_chart", chart, ok=chart.stratum_certified)
            chart_ids.append(cid)
            for cone in chart.child_strata:
                nid = add("projective_newton_cone", cone, ok=cone.stratum_certified)
                edges.append((cid, nid))
        certified = certified and projective_atlas.coverage_certified
        for a, b in projective_atlas.adjacency:
            if a < len(chart_ids) and b < len(chart_ids):
                edges.append((chart_ids[a], chart_ids[b]))
    if complex_geometry is not None:
        for div in complex_geometry.valuations:
            add("complex_divisor", div, ok=div.stratum_certified)
        certified = certified and complex_geometry.certified
    return UnifiedValuationCompactification(
        tuple(nodes),
        tuple(dict.fromkeys(edges)),
        (
            CoverageCertificate.complete(
                "unified_valuation_compactification",
                "finite, projective, and complex valuation nodes cover every requested compactification component",
                tuple(node.index for node in nodes),
            )
            if certified
            else CoverageCertificate.partial(
                "unified_valuation_compactification",
                "one or more valuation compactification components lack certified coverage",
                tuple(node.index for node in nodes if node.certified),
                tuple(node.index for node in nodes if not node.certified),
            )
        ),
    )


@dataclass(frozen=True)
class QEBudgetCacheStats:
    """Statistics for adaptive exact QE/projection reuse."""

    hits: int
    misses: int
    escalations: int
    entries: int


@dataclass(frozen=True)
class TangentSpaceLimit:
    """Certified limiting tangent direction for a one-parameter mapped branch."""

    parameter: sp.Symbol
    target_parameter: sp.Expr
    direction: tuple[sp.Expr, ...] | None
    certified: bool
    provider: str
    plucker_coordinates: tuple[sp.Expr, ...] | None = None
    plane_dimension: int | None = None


@dataclass(frozen=True)
class WhitneyPairResult:
    """Whitney (a)/(b) result for one incident smooth-stratum pair."""

    higher_formula: sp.Expr
    lower_formula: sp.Expr
    condition_a: bool | None
    condition_b: bool | None
    tangent_limit: TangentSpaceLimit | None
    certified: bool
    provider: str
    statement: str


@dataclass(frozen=True)
class WhitneyRefinementResult:
    """Whitney checks attached to a Jacobian-rank stratification."""

    mapping_result: StratifiedMappingResult
    pairs: tuple[WhitneyPairResult, ...]
    certified: bool
    provider: str


_QE_RESULT_CACHE: dict[tuple[str, tuple[str, ...]], tuple[int, sp.Expr | None]] = {}
_QE_CACHE_HITS = 0
_QE_CACHE_MISSES = 0
_QE_CACHE_ESCALATIONS = 0


def clear_qe_projection_cache() -> None:
    """Clear the process-local exact projection cache and its counters."""
    global _QE_CACHE_HITS, _QE_CACHE_MISSES, _QE_CACHE_ESCALATIONS
    _QE_RESULT_CACHE.clear()
    _QE_CACHE_HITS = _QE_CACHE_MISSES = _QE_CACHE_ESCALATIONS = 0


def qe_projection_cache_stats() -> QEBudgetCacheStats:
    """Return process-local adaptive QE/projection cache statistics."""
    return QEBudgetCacheStats(
        _QE_CACHE_HITS, _QE_CACHE_MISSES, _QE_CACHE_ESCALATIONS, len(_QE_RESULT_CACHE)
    )


def _cached_qe_image(
    mapping: StratumMap, source_formula: sp.Expr, *, budget: int = 5000
):
    """Budget-aware memoized wrapper around the exact image scheduler."""
    global _QE_CACHE_HITS, _QE_CACHE_MISSES, _QE_CACHE_ESCALATIONS
    key = (
        sp.srepr(
            sp.And(
                source_formula,
                mapping.source_constraint,
                *(
                    sp.Eq(y, e)
                    for y, e in zip(
                        mapping.target_variables, mapping.expressions, strict=True
                    )
                ),
            )
        ),
        tuple(map(str, (*mapping.source_variables, *mapping.target_variables))),
    )
    previous = _QE_RESULT_CACHE.get(key)
    if previous is not None and (previous[1] is not None or previous[0] >= budget):
        _QE_CACHE_HITS += 1
        cached_budget, value = previous
        attempt = ProjectionAttempt(
            "cache", 0, False, value is not None, f"reused budget {cached_budget}"
        )
        return value, ProjectionSchedule((attempt,), budget, value is None)
    _QE_CACHE_MISSES += 1
    if previous is not None and budget > previous[0]:
        _QE_CACHE_ESCALATIONS += 1
    value, schedule = _qe_image(mapping, source_formula, budget=budget)
    _QE_RESULT_CACHE[key] = (budget, value)
    return value, schedule


def _leading_direction(expressions, parameter, target=0):
    shifted = [
        sp.cancel(sp.sympify(e).subs(parameter, parameter + target))
        for e in expressions
    ]
    derivs = [sp.diff(e, parameter) for e in shifted]
    orders = []
    leads = []
    for d in derivs:
        if d == 0:
            orders.append(sp.oo)
            leads.append(sp.S.Zero)
            continue
        try:
            p = sp.Poly(d, parameter)
            terms = [(m[0], c) for m, c in p.terms() if c != 0]
            order = min(k for k, _ in terms)
            coeff = p.coeff_monomial(parameter**order)
        except (sp.PolynomialError, TypeError, ValueError):
            return None
        orders.append(order)
        leads.append(coeff)
    finite = [o for o in orders if o is not sp.oo]
    if not finite:
        return None
    m = min(finite)
    vec = tuple(
        sp.simplify(c if o == m else 0) for o, c in zip(orders, leads, strict=True)
    )
    if all(v == 0 for v in vec):
        return None
    return vec


def _plucker_limit(mapping: StratumMap, target):
    """Exact limiting Grassmann point when Plücker ratios have a common lead."""
    xs = mapping.source_variables
    if not xs:
        return None
    target = tuple(sp.sympify(v) for v in target)
    shifted = tuple(sp.Dummy(f"_w{i}", real=True) for i in range(len(xs)))
    subs = {x: a + u for x, a, u in zip(xs, target, shifted, strict=True)}
    jac = sp.Matrix(mapping.expressions).jacobian(xs).subs(subs)
    k = len(xs)
    if jac.rows < k:
        return None
    minors = []
    for rows in combinations(range(jac.rows), k):
        minors.append(sp.expand(jac.extract(rows, range(k)).det()))
    lead_data = []
    for m in minors:
        if m == 0:
            lead_data.append((sp.oo, sp.S.Zero))
            continue
        try:
            p = sp.Poly(m, *shifted)
        except (sp.PolynomialError, TypeError, ValueError):
            return None
        d = min(sum(mon) for mon, c in p.terms() if c != 0)
        h = sp.Add(
            *(
                c * sp.prod(x**e for x, e in zip(shifted, mon, strict=True))
                for mon, c in p.terms()
                if sum(mon) == d
            )
        )
        lead_data.append((d, sp.expand(h)))
    finite = [d for d, _ in lead_data if d != sp.oo]
    if not finite:
        return None
    d0 = min(finite)
    leads = tuple(h if d == d0 else sp.S.Zero for d, h in lead_data)
    nonzero = [h for h in leads if h != 0]
    pivot = nonzero[0]
    # A Grassmann limit is direction-independent when all leading Plücker
    # coordinates are constant multiples of one common homogeneous factor.
    ratios = []
    for h in leads:
        if h == 0:
            ratios.append(sp.S.Zero)
            continue
        q = sp.cancel(h / pivot)
        if q.has(*shifted):
            return None
        ratios.append(q)
    return tuple(ratios), k


def tangent_space_limit(
    mapping: StratumMap, *, parameter_target=0
) -> TangentSpaceLimit:
    """Certify limiting tangent planes using exact Grassmann/Plücker data."""
    if len(mapping.source_variables) == 1:
        t = mapping.source_variables[0]
        direction = _leading_direction(
            mapping.expressions, t, sp.sympify(parameter_target)
        )
        return TangentSpaceLimit(
            t,
            sp.sympify(parameter_target),
            direction,
            direction is not None,
            "exact_leading_tangent",
            tuple(direction) if direction is not None else None,
            1,
        )
    if isinstance(parameter_target, (tuple, list, sp.Tuple)):
        target = tuple(parameter_target)
    else:
        target = (sp.sympify(parameter_target),) * len(mapping.source_variables)
    pl = _plucker_limit(mapping, target)
    if pl is None:
        return TangentSpaceLimit(
            mapping.source_variables[0],
            sp.Tuple(*target),
            None,
            False,
            "grassmann_limit_unresolved",
            None,
            len(mapping.source_variables),
        )
    coords, k = pl
    return TangentSpaceLimit(
        mapping.source_variables[0],
        sp.Tuple(*target),
        None,
        True,
        "exact_plucker_grassmann_limit",
        coords,
        k,
    )


def check_whitney_conditions(
    mapped: MappedLocalStratum, *, parameter_target=0
) -> WhitneyRefinementResult:
    """Check Whitney (a)/(b) where exact tangent/secant limits are available.

    One-parameter smooth branches incident to a zero-dimensional target are
    certified by exact leading tangent and secant directions.  Unsupported
    higher-dimensional tangent-Grassmann limits remain unresolved rather than
    being inferred numerically.
    """
    result = refine_mapped_stratum(mapped)
    mapping = mapped.mapping
    pairs: list[WhitneyPairResult] = []
    if len(mapping.source_variables) == 1:
        t = mapping.source_variables[0]
        tl = tangent_space_limit(mapping, parameter_target=parameter_target)
        point = tuple(
            sp.simplify(e.subs(t, parameter_target)) for e in mapping.expressions
        )
        secant_expr = tuple(
            sp.cancel(e - p) for e, p in zip(mapping.expressions, point, strict=True)
        )
        sec = _leading_direction(secant_expr, t, sp.sympify(parameter_target))
        # For a point lower stratum Whitney (a) is automatic. Whitney (b)
        # requires the limiting secant line to lie in the limiting tangent line.
        b = None
        if tl.direction is not None and sec is not None:
            M = sp.Matrix([tl.direction, sec])
            b = M.rank() <= 1
        higher = sp.Or(*(p.source_formula for p in result.rank_strata if p.rank > 0))
        lower = sp.And(mapped.local_constraint, sp.Eq(t, parameter_target))
        cert = tl.certified and b is not None
        pairs.append(
            WhitneyPairResult(
                higher,
                lower,
                True,
                b,
                tl,
                cert,
                "exact_curve_tangent_secant",
                "Whitney (a) is automatic over a point; (b) is decided by exact limiting tangent/secant lines.",
            )
        )
    else:
        xs = mapping.source_variables
        target = (
            tuple(sp.sympify(parameter_target) for _ in xs)
            if not isinstance(parameter_target, (tuple, list, sp.Tuple))
            else tuple(map(sp.sympify, parameter_target))
        )
        tl = tangent_space_limit(mapping, parameter_target=target)
        point = tuple(
            sp.simplify(e.subs(dict(zip(xs, target, strict=True))))
            for e in mapping.expressions
        )
        J = sp.Matrix(mapping.expressions).jacobian(xs)
        sec = sp.Matrix(
            [e - p for e, p in zip(mapping.expressions, point, strict=True)]
        )
        aug = J.row_join(sec)
        k = len(xs)
        # Strong exact Whitney-(b) certificate: every (k+1)-minor vanishes
        # identically, so each secant lies in the current tangent k-plane.
        wedge_zero = True
        if aug.rows >= k + 1:
            for rows in combinations(range(aug.rows), k + 1):
                d = sp.cancel(aug.extract(rows, range(k + 1)).det())
                if d != 0 and d.is_zero is not True:
                    wedge_zero = False
                    break
        cert = tl.certified and wedge_zero
        pairs.append(
            WhitneyPairResult(
                mapped.local_constraint,
                sp.And(*(sp.Eq(x, a) for x, a in zip(xs, target, strict=True))),
                True if tl.certified else None,
                True if cert else None,
                tl,
                cert,
                "plucker_grassmann_secant",
                "Whitney (a) over the incident point and (b) are certified by an exact limiting Grassmann point and vanishing secant/tangent wedge.",
            )
        )
    certified = bool(pairs) and all(
        p.certified and p.condition_a is True and p.condition_b is True for p in pairs
    )
    return WhitneyRefinementResult(
        result, tuple(pairs), certified, "whitney_tangent_limit_refinement"
    )


@dataclass(frozen=True)
class GrassmannClusterSet:
    """Exact semialgebraic cluster set of tangent k-planes in Plücker coordinates."""

    coordinates: tuple[sp.Symbol, ...]
    formula: sp.Expr | None
    plane_dimension: int
    certified: bool
    provider: str = "none"


@dataclass(frozen=True)
class GeneralWhitneyPairResult:
    """Whitney (a)/(b) decision for an arbitrary incident smooth-stratum pair."""

    tangent_cluster: GrassmannClusterSet
    condition_a: bool | None
    condition_b: bool | None
    certified: bool
    provider: str
    counterexample_formula: sp.Expr | None = None


def grassmann_tangent_cluster_set(
    mapping: StratumMap, *, source_target=None, source_constraint=sp.S.true
):
    """Compute the tangent-plane cluster set by semialgebraic graph closure.

    The Gauss map is represented projectively by normalized Plücker coordinates.
    Taking the closure of its graph before specializing the source target keeps
    every limiting tangent plane, including direction-dependent limits.
    """
    xs = mapping.source_variables
    k = len(xs)
    if source_target is None:
        source_target = (sp.S.Zero,) * k
    source_target = tuple(map(sp.sympify, source_target))
    J = sp.Matrix(mapping.expressions).jacobian(xs)
    if J.rows < k:
        return GrassmannClusterSet((), None, k, False, "rank_too_small")
    minors = tuple(
        sp.expand(J.extract(rows, range(k)).det())
        for rows in combinations(range(J.rows), k)
    )
    if not minors:
        return GrassmannClusterSet((), None, k, False, "no_plucker_coordinates")
    ps = sp.symbols(f"_pl0:{len(minors)}", real=True)
    norm_m = sp.Add(*(m * m for m in minors))
    relations = [sp.Eq(sp.Add(*(p * p for p in ps)), 1), sp.Gt(norm_m, 0)]
    pivot_rel = []
    for i in range(len(minors)):
        for j in range(i + 1, len(minors)):
            pivot_rel.append(sp.Eq(sp.expand(ps[i] * minors[j] - ps[j] * minors[i]), 0))
    graph = sp.And(source_constraint, *relations, *pivot_rel)
    try:
        from semialg import region_closure

        closed = region_closure(graph, (*xs, *ps))
        formula = sp.simplify(closed.subs(dict(zip(xs, source_target, strict=True))))
    except (
        ImportError,
        ModuleNotFoundError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return GrassmannClusterSet(ps, None, k, False, "semialg_closure_failed")
    return GrassmannClusterSet(ps, formula, k, True, "semialg_gauss_graph_closure")


def _plucker_contains_vector_equations(plucker, vector, ambient_dimension, k):
    """Equations for v lying in a k-plane represented by Plücker coordinates."""
    subsets = tuple(combinations(range(ambient_dimension), k))
    pmap = dict(zip(subsets, plucker, strict=True))
    eqs = []
    # v wedge P = 0: every (k+1)-coordinate vanishes.
    for image in combinations(range(ambient_dimension), k + 1):
        term = sp.S.Zero
        for j, idx in enumerate(image):
            sub = image[:j] + image[j + 1 :]
            term += (-1) ** j * vector[idx] * pmap.get(tuple(sub), sp.S.Zero)
        eqs.append(sp.Eq(sp.expand(term), 0))
    return tuple(eqs)


def check_general_whitney_pair(
    higher: MappedLocalStratum,
    lower: MappedLocalStratum,
    *,
    higher_target=None,
    lower_target=None,
):
    """Certify Whitney (a)/(b) for an incident semialgebraic smooth pair.

    Whitney (a) is a universal containment statement over the exact Grassmann
    cluster set. Whitney (b) uses the joint closure of tangent Plücker and
    normalized secant coordinates, so varying tangent planes are handled rather
    than collapsed to one guessed limit.
    """
    hm, lm = higher.mapping, lower.mapping
    if len(hm.expressions) != len(lm.expressions):
        raise ValueError("higher and lower strata must map to the same ambient space")
    ht = tuple(map(sp.sympify, higher_target or (0,) * len(hm.source_variables)))
    lt = tuple(map(sp.sympify, lower_target or (0,) * len(lm.source_variables)))
    cluster = grassmann_tangent_cluster_set(
        hm, source_target=ht, source_constraint=higher.local_constraint
    )
    if not cluster.certified or cluster.formula is None:
        return GeneralWhitneyPairResult(
            cluster, None, None, False, "grassmann_cluster_unresolved"
        )
    ambient = len(hm.expressions)
    k = len(hm.source_variables)
    lower_J = (
        sp.Matrix(lm.expressions)
        .jacobian(lm.source_variables)
        .subs(dict(zip(lm.source_variables, lt, strict=True)))
    )
    a_eqs = []
    for col in range(lower_J.cols):
        vec = tuple(lower_J[row, col] for row in range(ambient))
        a_eqs.extend(
            _plucker_contains_vector_equations(cluster.coordinates, vec, ambient, k)
        )
    a_good = sp.And(*a_eqs) if a_eqs else sp.S.true
    try:
        from semialg import quantifier_eliminate, region_closure

        bad_a = sp.And(cluster.formula, sp.Not(a_good))
        qa = quantifier_eliminate(
            bad_a,
            quantifiers=[*(("exists", p) for p in cluster.coordinates)],
            variables=cluster.coordinates,
        )
        cond_a = qa is sp.S.false or sp.simplify(qa) is sp.S.false

        # Joint tangent/secant graph. The lower incident image point anchors the secant.
        lower_point = tuple(
            sp.simplify(e.subs(dict(zip(lm.source_variables, lt, strict=True))))
            for e in lm.expressions
        )
        xs = hm.source_variables
        J = sp.Matrix(hm.expressions).jacobian(xs)
        minors = tuple(
            sp.expand(J.extract(rows, range(k)).det())
            for rows in combinations(range(ambient), k)
        )
        ps = cluster.coordinates
        sec = tuple(
            sp.expand(e - p0) for e, p0 in zip(hm.expressions, lower_point, strict=True)
        )
        ss = sp.symbols(f"_sec0:{ambient}", real=True)
        secnorm = sp.Add(*(v * v for v in sec))
        rel = [
            sp.Eq(sp.Add(*(s * s for s in ss)), 1),
            sp.Gt(secnorm, 0),
            sp.Gt(sp.Add(*(m * m for m in minors)), 0),
        ]
        rel.extend(
            sp.Eq(sp.expand(ps[i] * minors[j] - ps[j] * minors[i]), 0)
            for i in range(len(minors))
            for j in range(i + 1, len(minors))
        )
        rel.extend(
            sp.Eq(sp.expand(ss[i] * sec[j] - ss[j] * sec[i]), 0)
            for i in range(ambient)
            for j in range(i + 1, ambient)
        )
        joint = sp.And(higher.local_constraint, *rel)
        joint_closed = region_closure(joint, (*xs, *ps, *ss)).subs(
            dict(zip(xs, ht, strict=True))
        )
        b_good = sp.And(*_plucker_contains_vector_equations(ps, ss, ambient, k))
        bad_b = sp.And(joint_closed, sp.Not(b_good))
        qb = quantifier_eliminate(
            bad_b,
            quantifiers=[*(("exists", p) for p in ps), *(("exists", s) for s in ss)],
            variables=(*ps, *ss),
        )
        cond_b = qb is sp.S.false or sp.simplify(qb) is sp.S.false
    except (
        ImportError,
        ModuleNotFoundError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
    ):
        return GeneralWhitneyPairResult(cluster, None, None, False, "whitney_qe_failed")
    return GeneralWhitneyPairResult(
        cluster,
        cond_a,
        cond_b,
        bool(cond_a and cond_b),
        "semialg_grassmann_secant_closure",
        None if cond_a and cond_b else sp.Or(bad_a, bad_b),
    )


__all__ = [
    "ClusterCompletenessCertificate",
    "CompletenessStep",
    "ComplexClusterGeometryResult",
    "GeneralWhitneyPairResult",
    "GrassmannClusterSet",
    "ProjectionAttempt",
    "ProjectionSchedule",
    "ProofCertificateNode",
    "QEBudgetCacheStats",
    "RankRefinedImageStratum",
    "RegularityRefinement",
    "StratifiedMappingResult",
    "TangentSpaceLimit",
    "UnifiedValuationCompactification",
    "UnifiedValuationNode",
    "WhitneyPairResult",
    "WhitneyRefinementResult",
    "check_general_whitney_pair",
    "check_whitney_conditions",
    "clear_qe_projection_cache",
    "completeness_certificate",
    "complex_cluster_geometry",
    "grassmann_tangent_cluster_set",
    "phase_bundle_over_stratum",
    "qe_projection_cache_stats",
    "refine_mapped_stratum",
    "tangent_space_limit",
    "unified_valuation_compactification",
]
