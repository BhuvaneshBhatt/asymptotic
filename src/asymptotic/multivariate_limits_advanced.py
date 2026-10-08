"""Advanced certified real multivariate-limit structures.

This module builds on the weighted expansion/atlas layer.  It returns UNKNOWN when a finite chart cover, a uniform angular statement, or a
composition theorem cannot be certified.
"""

from __future__ import annotations

import functools

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_policy import bounded_ask, bounded_limit, bounded_solve_one
from .coverage import CoverageCertificate, coverage_certificate
from .multivariate_limit_models import (
    AdvancedLimitStatus,
    ClusterSetResult,
    ComplexMeromorphicGeometryResult,
    ComplexMultivariateLimitResult,
    CoordinateChart,
    ExtremalBoundResult,
    JointClusterGeometryResult,
    JointClusterStratum,
    LocalGerm,
    LocalImageStratum,
    NewtonFanLimitResult,
    ProjectiveClusterAtlasResult,
    RelativeLimitResult,
    VectorClusterSetResult,
    VectorLimitResult,
    VectorLocalGerm,
)


def _norm_vars(variables):
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    return variables, normalize_limit_target


def certify_local_germ(expr, variables, target, *, domain=True) -> LocalGerm:
    """Certify the finite local value of a scalar germ.

    This is the reusable composition boundary: downstream composition code can
    consume a certified inner germ without knowing whether its proof came from
    CAD, a Newton chart, a blow-up, or a transcendental reduction.
    """
    from .limit_models import LimitStatus
    from .limits import limit

    variables, normalize = _norm_vars(variables)
    target = normalize(variables, target)
    expr, domain = sp.sympify(expr), sp.sympify(domain)
    result = limit(expr, variables, target, domain=domain, return_result=True)
    if result.status is LimitStatus.PROVED and result.value is not None:
        return LocalGerm(
            expr,
            variables,
            target,
            result.value,
            domain,
            AdvancedLimitStatus.CERTIFIED,
            "limit",
            "inner scalar germ has a certified finite local value",
        )
    return LocalGerm(
        expr,
        variables,
        target,
        None,
        domain,
        AdvancedLimitStatus.UNKNOWN,
        "none",
        "finite local value was not certified",
    )


def compose_local_germ(outer, inner, variables, target, *, domain=True) -> LocalGerm:
    """Compose a certified scalar germ through a one-variable outer expression."""
    from ._limit_composition import _two_sided_limit

    inner_germ = certify_local_germ(inner, variables, target, domain=domain)
    z = sp.Dummy("_germ_z", real=True)
    outer_expr = sp.sympify(outer(z) if callable(outer) else outer)
    if not inner_germ.certified:
        return LocalGerm(
            outer_expr.subs(z, inner),
            inner_germ.variables,
            inner_germ.target,
            None,
            inner_germ.domain,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "inner germ is not certified",
        )
    value = _two_sided_limit(outer_expr, z, inner_germ.limit)
    if value is None or value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return LocalGerm(
            outer_expr.subs(z, inner),
            inner_germ.variables,
            inner_germ.target,
            None,
            inner_germ.domain,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "outer two-sided local germ was not finite/certified",
        )
    return LocalGerm(
        outer_expr.subs(z, inner),
        inner_germ.variables,
        inner_germ.target,
        value,
        inner_germ.domain,
        AdvancedLimitStatus.CERTIFIED,
        "local_germ_composition",
        "certified inner germ composed with a certified finite outer germ",
    )


def newton_fan_limit(
    expr, variables, target, *, domain=True, order=2
) -> NewtonFanLimitResult:
    """Certify a limit from a finite certified Newton/weighted blow-up atlas.

    Every retained chart must give the same constant radial leading value or a
    strictly positive radial order.  Atlas coverage then upgrades the chartwise
    statement to the full relative-domain limit.
    """
    from .limit_primitives import _normalize_variables, normalize_limit_target
    from .multivariate_expansion import multivariate_atlas

    expr, domain = sp.sympify(expr), sp.sympify(domain)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    atlas = multivariate_atlas(
        expr, variables, target, order=order, domain=domain, first_certified_family=True
    )
    if not atlas.certified:
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            atlas.weights,
            "none",
            "no certified covering Newton atlas",
        )
    values = []
    for chart in atlas.charts:
        lead = chart.expansion.leading_term
        if lead is None:
            values.append(sp.S.Zero)
            continue
        if bounded_ask(sp.Q.positive(lead.order)) is True:
            values.append(sp.S.Zero)
            continue
        if lead.order == 0 and not (
            lead.coefficient.free_symbols & set(chart.expansion.angular_variables)
        ):
            values.append(sp.simplify(lead.coefficient))
            continue
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            atlas.weights,
            "newton_blowup_atlas",
            "a covering chart has an angular/nondecaying leading coefficient",
        )
    if not values:
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            atlas.weights,
            "none",
            "certified atlas has no usable chart values",
        )
    first = values[0]
    if any(sp.simplify(v - first) != 0 for v in values[1:]):
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            atlas.weights,
            "newton_blowup_atlas",
            "covering Newton charts do not certify a common value",
        )
    # Before promoting finite Newton-chart coverage, probe exact higher-order
    # curve refinements that finite weight fans can miss near cancellation
    # divisors (cusps/tacnodes and y=+-x+o(x) approaches).  A disagreement
    # invalidates certification; agreement remains only a soundness guard, not
    # proof by sampling.
    if len(variables) == 2 and all(a.is_finite is not False for a in target):
        tt = sp.Dummy("_newton_guard_t", positive=True)
        x, y = variables
        x0, y0 = target
        guards = []
        for k in (2, 3, 4, 5, 6):
            guards.extend(
                (
                    {x: x0 + tt, y: y0 + tt**k},
                    {x: x0 + tt, y: y0 - tt + tt**k},
                    {x: x0 + tt, y: y0 + tt + tt**k},
                )
            )
        for sub in guards:
            try:
                gv = bounded_limit(
                    expr.subs(sub), tt, 0, direction="+", allow_general=True
                )
            except (TypeError, ValueError, NotImplementedError, sp.PoleError):
                continue
            if gv in (sp.zoo, sp.nan) or isinstance(gv, (sp.Set, sp.AccumBounds)):
                continue
            if sp.simplify(gv - first) != 0:
                return NewtonFanLimitResult(
                    expr,
                    variables,
                    target,
                    None,
                    AdvancedLimitStatus.UNKNOWN,
                    atlas.weights,
                    "newton_curve_guard",
                    "higher-order algebraic curve refinement disagrees with the finite Newton atlas",
                )
    return NewtonFanLimitResult(
        expr,
        variables,
        target,
        first,
        AdvancedLimitStatus.CERTIFIED,
        atlas.weights,
        "newton_blowup_atlas",
        "all charts in a certified finite weighted atlas have the same limit",
    )


def _homogeneous_angular(expr, variables, target, weights):
    """Return the zero-order exceptional-divisor map from a common blow-up chart."""
    from .blowup_geometry import weighted_spherical_atlas

    atlas = weighted_spherical_atlas(variables, target, weights)
    if not atlas.certified:
        return None
    chart = atlas.charts[0]
    r = chart.radial_variable
    transformed = chart.transform(expr)
    num, den = sp.fraction(transformed)
    try:
        pn, pd = sp.Poly(num, r), sp.Poly(den, r)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    nv = min(m[0] for m, _ in pn.terms())
    dv = min(m[0] for m, _ in pd.terms())
    if nv != dv:
        return None
    an = sp.expand(pn.coeff_monomial(r**nv))
    ad = sp.expand(pd.coeff_monomial(r**dv))
    return sp.cancel(an / ad), chart.angular_variables


def multivariate_cluster_set(
    expr, variables, target, *, domain=True
) -> ClusterSetResult:
    """Certify a cluster interval for continuous weighted-homogeneous rational germs."""
    from .angular_extrema import _angular_squared_extremum, _certified_extremum
    from .blowup_geometry import newton_candidate_rays
    from .limit_primitives import _normalize_variables, normalize_limit_target

    def _newton_weight_vectors(expr, variables, target):
        return newton_candidate_rays(expr, tuple(variables), tuple(target))

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr, domain = sp.sympify(expr), sp.sympify(domain)

    # Relative semialgebraic domains are pulled onto each weighted exceptional divisor.
    weights = {(1,) * len(variables)} | set(
        _newton_weight_vectors(expr, variables, target)
    )
    for w in sorted(weights, key=lambda x: (sum(x), x)):
        data = _homogeneous_angular(expr, variables, target, w)
        if data is None:
            continue
        angular, u = data
        _num, den = sp.fraction(angular)
        dmin = _angular_squared_extremum(sp.expand(den**2), u, kind="min")
        if dmin is None or bounded_ask(sp.Q.positive(dmin[0])) is not True:
            continue
        sphere_sum = sp.Add(*(x**2 for x in u))
        sphere = sp.Eq(sphere_sum, 1)
        anum, aden = sp.fraction(sp.cancel(angular))
        if sp.simplify(aden - sphere_sum) == 0:
            angular = sp.expand(anum)
        angular_domain = sphere
        if domain not in (sp.S.true, True):
            from .blowup_geometry import weighted_spherical_atlas
            from .domain_cluster_geometry import transform_domain_to_chart

            geometry = weighted_spherical_atlas(variables, target, w)
            chart = geometry.charts[0]
            transformed = transform_domain_to_chart(domain, chart)
            replacements = dict(zip(chart.angular_variables, u, strict=True))
            angular_domain = sp.And(sphere, transformed.xreplace(replacements))
        lo = _certified_extremum(angular, angular_domain, u, kind="min")
        hi = _certified_extremum(angular, angular_domain, u, kind="max")
        if (lo is None or hi is None) and len(u) == 2:
            from .multivariate_expansion import _reduce_sphere_invariants

            reduced = _reduce_sphere_invariants(angular, u)
            z = sp.Dummy("_cluster_z", real=True)
            # Even angular functions descend exactly to z=u0**2, u1**2=1-z.
            if all(sp.simplify(reduced.xreplace({ui: -ui}) - reduced) == 0 for ui in u):
                descended = sp.cancel(reduced.subs(u[0] ** 2, z).subs(u[1] ** 2, 1 - z))
                if not (descended.free_symbols & set(u)):
                    try:
                        from sympy.calculus.util import maximum, minimum

                        interval = sp.Interval(0, 1)
                        lo = (
                            minimum(descended, z, interval),
                            "sphere_simplex_univariate",
                        )
                        hi = (
                            maximum(descended, z, interval),
                            "sphere_simplex_univariate",
                        )
                    except (
                        ArithmeticError,
                        TypeError,
                        ValueError,
                        NotImplementedError,
                    ):
                        pass
        if lo is None or hi is None:
            from .semialgebraic_angular import semialgebraic_angular_image

            general = semialgebraic_angular_image(angular, u, angular_domain)
            if general.certified:
                lo = (general.minimum, general.provider)
                hi = (general.maximum, general.provider)
        if lo is None or hi is None:
            continue
        a, b = sp.simplify(lo[0]), sp.simplify(hi[0])
        return ClusterSetResult(
            expr,
            variables,
            target,
            sp.Interval(a, b),
            a,
            b,
            AdvancedLimitStatus.CERTIFIED,
            "weighted_angular_extrema",
            f"continuous angular image on the connected sphere/domain for weights {w}",
            coverage=CoverageCertificate.complete(
                "weighted_angular_domain_cover",
                "the weighted exceptional divisor intersected with the approach domain exhausts this homogeneous germ",
                (w,),
            ),
        )
    return ClusterSetResult(
        expr,
        variables,
        target,
        None,
        None,
        None,
        AdvancedLimitStatus.UNKNOWN,
        "none",
        "no continuous weighted-homogeneous angular model was certified",
    )


def multivariate_liminf(expr, variables, target, *, domain=True, return_result=False):
    """Return multivariate liminf; optionally retain certification provenance."""
    cluster = multivariate_cluster_set(expr, variables, target, domain=domain)
    result = ExtremalBoundResult(
        cluster.liminf, cluster.status, cluster.provider, cluster.statement, cluster
    )
    return result if return_result else result.value


def multivariate_limsup(expr, variables, target, *, domain=True, return_result=False):
    """Return multivariate limsup; optionally retain certification provenance."""
    cluster = multivariate_cluster_set(expr, variables, target, domain=domain)
    result = ExtremalBoundResult(
        cluster.limsup, cluster.status, cluster.provider, cluster.statement, cluster
    )
    return result if return_result else result.value


def _bounded_unit_factor(expr: sp.Expr) -> bool:
    """Return whether ``expr`` has a global absolute bound of one on the reals."""
    return expr.func in (sp.sin, sp.cos, sp.sign) or (
        expr.func is sp.Abs and expr.args[0].func in (sp.sin, sp.cos, sp.sign)
    )


def _term_envelope(term: sp.Expr) -> tuple[sp.Expr, bool]:
    """Remove only multiplicative, independently bounded factors from one term.

    This works term-by-term.  Replacing oscillatory atoms globally
    is unsound because cancellation between additive terms can make the replaced
    expression vanish even when the original expression does not.
    """
    coeff, factors = sp.sympify(term).as_coeff_mul()
    kept = [sp.Abs(coeff)]
    removed = False
    for factor in factors:
        if _bounded_unit_factor(factor):
            removed = True
            continue
        if (
            isinstance(factor, sp.Pow)
            and factor.exp.is_integer is True
            and factor.exp.is_nonnegative is True
            and _bounded_unit_factor(factor.base)
        ):
            removed = True
            continue
        kept.append(sp.Abs(factor))
    return sp.Mul(*kept), removed


def bounded_factor_envelope_limit(
    expr, variables, target, *, domain=True
) -> NewtonFanLimitResult:
    """Certify a zero limit from a finite sum of vanishing term envelopes.

    Each removed factor has a proved global bound ``|factor| <= 1``.  Every
    additive term is handled independently, so no cancellation of artificial
    replacements can be mistaken for a proof.
    """
    from .limit_primitives import _normalize_variables
    from .limits import LimitStatus, limit, normalize_limit_target

    expr = sp.sympify(expr)
    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    terms = sp.Add.make_args(sp.expand_mul(expr))
    envelopes = []
    removed_any = False
    for term in terms:
        envelope, removed = _term_envelope(term)
        # Terms without a removable bounded factor are harmless when their own
        # absolute envelope is independently certified to vanish.
        removed_any = removed_any or removed
        envelopes.append(envelope)
    if not removed_any:
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "no supported bounded multiplicative factor",
        )
    for envelope in envelopes:
        result = limit(envelope, variables, target, domain=domain, return_result=True)
        if result.status is not LimitStatus.PROVED or result.value != 0:
            return NewtonFanLimitResult(
                expr,
                variables,
                target,
                None,
                AdvancedLimitStatus.UNKNOWN,
                (),
                "none",
                "not every absolute term envelope was certified to vanish",
            )
    return NewtonFanLimitResult(
        expr,
        variables,
        target,
        sp.S.Zero,
        AdvancedLimitStatus.CERTIFIED,
        (),
        "bounded_factor_envelope",
        "triangle inequality with independently bounded multiplicative factors and certified vanishing term envelopes",
    )


def relative_limit(expr, variables, target, *, domain, return_result=False):
    """Certified simultaneous real limit relative to an explicit approach set.

    The domain is part of the semantics, not a path-sampling filter.  This is a
    public wrapper around the core domain-aware epsilon/closure machinery.
    """
    from .limit_primitives import _normalize_variables
    from .limits import LimitStatus, limit, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    domain = sp.sympify(domain)
    core = limit(expr, variables, target, domain=domain, return_result=True)
    ok = core.status is LimitStatus.PROVED
    result = RelativeLimitResult(
        sp.sympify(expr),
        variables,
        target,
        domain,
        core.value if ok else None,
        AdvancedLimitStatus.CERTIFIED if ok else AdvancedLimitStatus.UNKNOWN,
        "limit:relative_domain" if ok else "none",
        "certified relative to the supplied local approach domain"
        if ok
        else "the relative-domain limit was not certified",
    )
    return result if return_result else (result.value if result.certified else result)


def _normalize_extended_target(variables, target):
    if isinstance(target, dict):
        target = tuple(sp.sympify(target[v]) for v in variables)
    elif len(variables) == 1 and not isinstance(target, (tuple, list, sp.Tuple)):
        target = (sp.sympify(target),)
    else:
        target = tuple(map(sp.sympify, target))
    if len(target) != len(variables):
        raise ValueError("target dimension must match variables")
    from .fixed_ray_branch_germs import DirectionalInfinity

    normalized = []
    for point in target:
        if point.is_Mul and any(a in (sp.oo, -sp.oo) for a in point.args):
            direction = (-1 if -sp.oo in point.args else 1) * sp.Mul(
                *(a for a in point.args if a not in (sp.oo, -sp.oo))
            )
            if direction.is_finite is True and direction.is_zero is False:
                point = DirectionalInfinity(direction / sp.Abs(direction))
        normalized.append(point)
    return tuple(normalized)


def projective_limit_chart(expr, variables, target, *, domain=True) -> CoordinateChart:
    """Normalize finite/infinite real targets to one finite reciprocal chart."""
    from .limit_primitives import _normalize_variables

    variables = _normalize_variables(variables)
    target = _normalize_extended_target(variables, target)
    new_vars, substitutions, finite_target, side = [], {}, [], []
    for i, (v, a) in enumerate(zip(variables, target, strict=True)):
        if a in (sp.oo, -sp.oo):
            t = sp.Dummy(f"_infinity_{i}", real=True)
            substitutions[v] = 1 / t
            new_vars.append(t)
            finite_target.append(sp.S.Zero)
            side.append(sp.Gt(t, 0) if a is sp.oo else sp.Lt(t, 0))
        else:
            new_vars.append(v)
            finite_target.append(a)
    transformed = sp.sympify(expr).subs(substitutions, simultaneous=True)
    transformed_domain = sp.sympify(domain).subs(substitutions, simultaneous=True)
    if side:
        transformed_domain = sp.And(transformed_domain, *side)
    return CoordinateChart(
        transformed,
        tuple(new_vars),
        tuple(finite_target),
        transformed_domain,
        tuple(substitutions.items()),
    )


def _finite_chart_transform(expr, variables, target):
    chart = projective_limit_chart(expr, variables, target)
    return chart.expression, chart.variables, chart.target, dict(chart.substitutions)


def extended_limit(expr, variables, target, *, domain=True, return_result=False):
    """Simultaneous limit allowing finite, +oo, -oo, and mixed real targets.

    Infinite coordinates are normalized by reciprocal projective charts.  The
    sign of each reciprocal coordinate is included in the transformed approach
    domain, so +oo and -oo remain distinct one-sided ends.
    """
    from .limit_primitives import _normalize_variables
    from .limits import limit

    variables = _normalize_variables(variables)
    chart = projective_limit_chart(expr, variables, target, domain=domain)
    result = limit(
        chart.expression,
        chart.variables,
        chart.target,
        domain=chart.domain,
        return_result=True,
    )
    return (
        result
        if return_result
        else (
            result.value if getattr(result.status, "value", "") == "proved" else result
        )
    )


def extended_newton_fan_limit(expr, variables, target, *, domain=True, order=2):
    """Run the Newton-fan dispatcher directly in a finite projective chart."""
    chart = projective_limit_chart(expr, variables, target, domain=domain)
    return newton_fan_limit(
        chart.expression,
        chart.variables,
        chart.target,
        domain=chart.domain,
        order=order,
    )


def extended_multivariate_cluster_set(expr, variables, target, *, domain=True):
    """Cluster-set analysis after finite projective normalization."""
    chart = projective_limit_chart(expr, variables, target, domain=domain)
    return multivariate_cluster_set(
        chart.expression, chart.variables, chart.target, domain=chart.domain
    )


def _semialgebraic_map_graph(expressions, variables, image_variables):
    """Build a joint semialgebraic graph for rational or algebraic coordinates."""
    try:
        from semialg import UnsupportedFunctionGraph, semialgebraic_function_graph
    except (ImportError, ModuleNotFoundError):
        return None

    clauses = []
    auxiliaries = []
    all_rational = True
    for z, expression in zip(image_variables, expressions, strict=True):
        expression = sp.sympify(expression)
        # Keep the cheap rational graph because it introduces no auxiliaries.
        rational = _rational_image_graph((expression,), variables, (z,))
        if rational is not None:
            clauses.append(rational)
            continue
        all_rational = False
        # Avoid graph auxiliaries for the common principal square-root case.
        # z = sqrt(a) is exactly z >= 0, z**2 = a; clearing a rational
        # radicand denominator keeps the graph polynomial for CAD/QE.
        if isinstance(expression, sp.Pow) and expression.exp == sp.Rational(1, 2):
            rad = sp.cancel(sp.together(expression.base))
            num, den = map(sp.expand, sp.fraction(rad))
            try:
                sp.Poly(num, *variables)
                sp.Poly(den, *variables)
            except sp.PolynomialError:
                pass
            else:
                clauses.append(sp.Ge(z, 0))
                if den != 1:
                    clauses.append(sp.Ne(den, 0))
                clauses.append(sp.Eq(sp.expand(z**2 * den - num), 0))
                continue
        try:
            graph = semialgebraic_function_graph(expression, z)
        except UnsupportedFunctionGraph:
            return None
        except (TypeError, ValueError, NotImplementedError, RuntimeError):
            return None
        clauses.append(graph.formula)
        auxiliaries.extend(graph.auxiliary_variables)
    return (
        (sp.And(*clauses) if clauses else sp.S.true),
        tuple(auxiliaries),
        all_rational,
    )


def _rational_image_graph(expressions, variables, image_variables):
    """Return a quantifier-free source/image graph for rational maps."""
    clauses = []
    for z, expression in zip(image_variables, expressions, strict=True):
        expression = sp.cancel(sp.together(expression))
        numerator, denominator = map(sp.expand, sp.fraction(expression))
        try:
            sp.Poly(numerator, *variables)
            sp.Poly(denominator, *variables)
        except sp.PolynomialError:
            return None
        if denominator != 1:
            clauses.append(sp.Ne(denominator, 0))
        clauses.append(sp.Eq(sp.expand(z * denominator - numerator), 0))
    return sp.And(*clauses) if clauses else sp.S.true


def _dnf_strata(formula):
    """Split a quantifier-free image formula into deterministic Boolean strata."""
    try:
        dnf = sp.to_dnf(sp.sympify(formula), simplify=True, force=True)
    except (TypeError, ValueError, NotImplementedError):
        dnf = sp.sympify(formula)
    if isinstance(dnf, sp.Or):
        return tuple(dnf.args)
    return (dnf,)


def _accumulating_image_strata(formula, image_variables, image_target):
    """Discard image strata whose closure misses the germ target."""
    try:
        from semialg import point_in_closure
    except (ImportError, ModuleNotFoundError):
        return tuple(
            LocalImageStratum(s, True, "unfiltered") for s in _dnf_strata(formula)
        )

    kept = []
    for stratum in _dnf_strata(formula):
        try:
            accumulating = bool(
                point_in_closure(stratum, image_target, image_variables)
            )
        except (
            ArithmeticError,
            TypeError,
            ValueError,
            NotImplementedError,
            RuntimeError,
        ):
            # Keeping an unresolved stratum is a sound over-approximation.
            kept.append(LocalImageStratum(stratum, True, "closure_unknown_kept"))
            continue
        if accumulating:
            kept.append(LocalImageStratum(stratum, True, "semialg_point_in_closure"))
    return tuple(kept)


def _explicit_equality_substitution(formula, candidates, protected=()):
    """Cheaply eliminate variables defined by unique explicit equalities."""
    formula = sp.sympify(formula)
    candidates = list(candidates)
    protected = set(protected)
    substitutions = {}
    changed = True
    while changed:
        changed = False
        current = sp.simplify(formula.subs(substitutions, simultaneous=True))
        atoms = list(current.args) if isinstance(current, sp.And) else [current]
        for atom in atoms:
            if not isinstance(atom, sp.Equality):
                continue
            for variable in tuple(candidates):
                if variable in protected or variable not in atom.free_symbols:
                    continue
                try:
                    solved = bounded_solve_one(atom, variable, allow_general=True) or ()
                except (NotImplementedError, TypeError, ValueError):
                    continue
                usable = [
                    sp.simplify(v)
                    for v in solved
                    if variable not in sp.sympify(v).free_symbols
                ]
                if len(usable) != 1:
                    continue
                value = usable[0].subs(substitutions, simultaneous=True)
                # Avoid replacing a source/aux variable by an expression that
                # still depends on another copy of itself through a cycle.
                if variable in value.free_symbols:
                    continue
                substitutions[variable] = value
                candidates.remove(variable)
                changed = True
                break
            if changed:
                break
    return (
        sp.simplify(formula.subs(substitutions, simultaneous=True)),
        tuple(candidates),
        substitutions,
    )


def _project_image_branch(formula, image_variables, eliminated):
    """Project one Boolean branch after cheap algebraic dimension reduction."""
    try:
        from semialg import quantifier_eliminate
    except (ImportError, ModuleNotFoundError):
        return None
    reduced, remaining, _ = _explicit_equality_substitution(
        formula, eliminated, protected=image_variables
    )
    if reduced in (sp.S.false, False):
        return sp.S.false
    # Variables removed by substitution must truly have disappeared; retain any
    # that survive due to a coupled equality.
    remaining = tuple(v for v in remaining if v in reduced.free_symbols)
    if not remaining:
        return sp.simplify(reduced)
    try:
        return sp.simplify(
            quantifier_eliminate(
                reduced,
                quantifiers=[*(("exists", variable) for variable in remaining)],
                variables=(*image_variables, *remaining),
            )
        )
    except (
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        RuntimeError,
        sp.PolynomialError,
    ):
        return None


def _planned_image_projection(source_formula, image_variables, eliminated):
    """Plan image projection branch-locally before monolithic QE.

    The planner performs explicit equality/graph-auxiliary substitution first,
    then projects DNF branches independently.  A monolithic QE call is used only
    when branch-local projection cannot complete the job.
    """
    branches = _dnf_strata(source_formula)
    projected = []
    if len(branches) <= 16:
        for branch in branches:
            result = _project_image_branch(branch, image_variables, eliminated)
            if result is None:
                projected = []
                break
            if result not in (sp.S.false, False):
                projected.append(result)
        if projected:
            return sp.simplify(sp.Or(*projected)), "branch_local_qe"
        if branches and all(b in (sp.S.false, False) for b in branches):
            return sp.S.false, "branch_local_qe"
    # Last resort: still perform cheap substitutions before one QE problem.
    reduced, remaining, _ = _explicit_equality_substitution(
        source_formula, eliminated, protected=image_variables
    )
    remaining = tuple(v for v in remaining if v in reduced.free_symbols)
    if not remaining:
        return sp.simplify(reduced), "equality_elimination"
    try:
        from semialg import quantifier_eliminate

        result = quantifier_eliminate(
            reduced,
            quantifiers=[*(("exists", variable) for variable in remaining)],
            variables=(*image_variables, *remaining),
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
    return sp.simplify(result), "reduced_monolithic_qe"


def _semialgebraic_image_constraints(
    expressions, variables, target, image_variables, image_target, domain
):
    """Compute a local semialgebraic image over-approximation and its strata.

    The source is first restricted to a fixed neighbourhood of the approach
    point.  Limit semantics are local, so any positive fixed neighbourhood is
    sufficient.  Source variables and graph auxiliaries are then eliminated by
    QE.  Finally, Boolean image strata whose closures miss the image target are
    discarded.  The retained union is therefore a sound local image-germ
    over-approximation, while avoiding irrelevant global branches.
    """
    # Pairwise resultants give necessary image equations for a rational curve.
    # They may include extra real branches, which is safe for an outer-limit
    # proof on the over-approximation. No surjectivity claim is made.
    if len(variables) == 1 and sp.sympify(domain) is sp.S.true:
        x = variables[0]
        polynomials = []
        for expression, z in zip(expressions, image_variables, strict=True):
            if not expression.is_rational_function(x):
                break
            n, d = sp.fraction(sp.cancel(expression))
            polynomials.append(sp.expand(z * d - n))
        else:
            equations = []
            for i, p in enumerate(polynomials):
                if not p.has(x):
                    equations.append(sp.Eq(p, 0))
                for q in polynomials[i + 1 :]:
                    if p.has(x) and q.has(x):
                        resultant = sp.factor(sp.resultant(p, q, x))
                        if resultant != 0:
                            equations.append(sp.Eq(resultant, 0))
            if equations:
                constraint = sp.And(*equations)
                return (
                    constraint,
                    "semialgebraic_local_image:rational_curve_resultants:strata",
                    (LocalImageStratum(constraint, True, "necessary_image_equations"),),
                    None,
                )
    graph_data = _semialgebraic_map_graph(expressions, variables, image_variables)
    if graph_data is None:
        return None
    graph, auxiliaries, all_rational = graph_data
    radius = sp.S.One
    distance2 = sp.Add(
        *(sp.expand((v - a) ** 2) for v, a in zip(variables, target, strict=True))
    )
    domain = sp.sympify(domain)
    # Preserve the old cheap path for unrestricted rational maps: their exact
    # global image is already a sound local over-approximation, and adding a
    # source ball can make CAD dramatically more expensive.  Localize when the
    # domain itself carries branches/inequalities or when algebraic graph
    # auxiliaries are present.
    localize = (domain not in (sp.S.true, True)) or not all_rational
    source_formula = (
        sp.And(domain, sp.Le(distance2, radius**2), graph) if localize else graph
    )
    eliminated = (*variables, *auxiliaries)
    planned = _planned_image_projection(source_formula, image_variables, eliminated)
    if planned is None:
        return None
    projected, projection_provider = planned
    projected = sp.simplify(projected)
    if projected in (sp.S.false, False):
        return None
    strata = _accumulating_image_strata(projected, image_variables, image_target)
    if not strata:
        return None
    local_constraint = sp.simplify(sp.Or(*(s.constraint for s in strata)))
    provider = f"semialgebraic_local_image:{projection_provider}:strata"
    return local_constraint, provider, strata, (radius if localize else None)


def _polynomial_image_constraints(expressions, variables, image_variables):
    """Certified polynomial equalities satisfied by the image germ.

    Groebner elimination returns necessary image equations, hence an over-approximation
    of the actual image.  Proving an outer limit on that larger set is sound.
    This remains the inexpensive fallback when semialgebraic QE is unavailable.
    """
    if not expressions:
        return sp.S.true, "empty_image"
    try:
        polys = [
            sp.Poly(z - e, *variables, *image_variables)
            for z, e in zip(image_variables, expressions, strict=True)
        ]
        gb = sp.groebner(
            [p.as_expr() for p in polys], *variables, *image_variables, order="lex"
        )
    except (sp.PolynomialError, ValueError, TypeError):
        return sp.S.true, "none"
    source = set(variables)
    rels = [
        sp.Eq(sp.expand(g.as_expr()), 0)
        for g in gb.polys
        if not (g.as_expr().free_symbols & source)
    ]
    if not rels:
        return sp.S.true, "none"
    return sp.And(*rels), "groebner_image_overapproximation"


def certify_vector_local_germ(
    expressions, variables, target, *, domain=True
) -> VectorLocalGerm:
    """Certify all components of a finite germ map at a common approach."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expressions = tuple(map(sp.sympify, expressions))
    germs = tuple(
        certify_local_germ(e, variables, target, domain=domain) for e in expressions
    )
    image_variables = tuple(
        sp.Dummy(f"_image_{i}", real=True) for i in range(len(expressions))
    )
    all_certified = all(g.certified for g in germs)
    image_target = (
        tuple(g.limit for g in germs)
        if all_certified
        else tuple(
            sp.sympify(e).subs(
                dict(zip(variables, target, strict=True)), simultaneous=True
            )
            for e in expressions
        )
    )
    # Image constraints are consumed only by certified composition/limit paths.
    # Avoid expensive QE for a map whose component germs are already unresolved.
    semialgebraic_image = (
        _semialgebraic_image_constraints(
            expressions, variables, target, image_variables, image_target, domain
        )
        if all_certified
        else None
    )
    if semialgebraic_image is not None:
        image_constraint, image_provider, image_strata, local_radius = (
            semialgebraic_image
        )
    else:
        image_constraint, image_provider = _polynomial_image_constraints(
            expressions, variables, image_variables
        )
        image_strata = (LocalImageStratum(image_constraint, True, image_provider),)
        local_radius = None
    if all_certified:
        return VectorLocalGerm(
            expressions,
            variables,
            target,
            tuple(g.limit for g in germs),
            sp.sympify(domain),
            AdvancedLimitStatus.CERTIFIED,
            "component_germs",
            "all components have certified finite local values on the common domain",
            image_variables,
            image_constraint,
            image_provider,
            image_strata,
            local_radius,
        )
    return VectorLocalGerm(
        expressions,
        variables,
        target,
        None,
        sp.sympify(domain),
        AdvancedLimitStatus.UNKNOWN,
        "none",
        "at least one component germ was not certified",
        image_variables,
        image_constraint,
        image_provider,
        image_strata,
        local_radius,
    )


def _reduce_image_constraint_equalities(
    expression, image_variables, target, constraint
):
    """Use explicit image equalities to lower the outer-limit dimension."""
    expression = sp.sympify(expression)
    constraint = sp.sympify(constraint)
    variables = list(image_variables)
    targets = dict(zip(image_variables, target, strict=True))
    substitutions = {}
    changed = True
    while changed:
        changed = False
        current = constraint.subs(substitutions, simultaneous=True)
        equations = list(current.args) if isinstance(current, sp.And) else [current]
        for equation in equations:
            if not isinstance(equation, sp.Equality):
                continue
            for variable in reversed(variables):
                if variable not in equation.free_symbols:
                    continue
                try:
                    solutions = (
                        bounded_solve_one(equation, variable, allow_general=True) or ()
                    )
                except (NotImplementedError, TypeError, ValueError):
                    continue
                usable = [
                    sp.simplify(solution)
                    for solution in solutions
                    if variable not in sp.sympify(solution).free_symbols
                ]
                if len(usable) != 1:
                    continue
                substitutions[variable] = usable[0].subs(
                    substitutions, simultaneous=True
                )
                variables.remove(variable)
                changed = True
                break
            if changed:
                break
    if not substitutions:
        return None
    reduced_expression = sp.simplify(expression.subs(substitutions, simultaneous=True))
    reduced_constraint = sp.simplify(constraint.subs(substitutions, simultaneous=True))
    atoms = (
        list(reduced_constraint.args)
        if isinstance(reduced_constraint, sp.And)
        else [reduced_constraint]
    )
    for variable in variables:
        for atom in atoms:
            if (
                isinstance(atom, sp.GreaterThan)
                and atom.lhs == variable
                and atom.rhs == 0
            ):
                reduced_expression = sp.refine(
                    reduced_expression, sp.Q.nonnegative(variable)
                )
            elif (
                isinstance(atom, sp.LessThan) and atom.lhs == variable and atom.rhs == 0
            ):
                reduced_expression = sp.refine(
                    reduced_expression, sp.Q.nonpositive(variable)
                )
    reduced_expression = sp.simplify(reduced_expression)
    reduced_target = tuple(targets[variable] for variable in variables)
    return reduced_expression, tuple(variables), reduced_target, reduced_constraint


def compose_vector_local_germ(
    outer, inners, variables, target, *, outer_variables=None, domain=True
) -> LocalGerm:
    """Compose a scalar outer germ with a jointly certified vector inner germ.

    The outer function may itself have a multivariate singularity: its
    simultaneous limit at the certified inner target is proved by the same core
    engine before composition is accepted.
    """
    from .limit_models import LimitStatus
    from .limits import limit

    vg = certify_vector_local_germ(inners, variables, target, domain=domain)
    if outer_variables is None:
        outer_variables = tuple(
            sp.Dummy(f"_outer_{i}", real=True) for i in range(len(vg.expressions))
        )
    else:
        outer_variables = tuple(outer_variables)
    outer_expr = sp.sympify(outer(*outer_variables) if callable(outer) else outer)
    composed = outer_expr.subs(
        dict(zip(outer_variables, vg.expressions, strict=True)), simultaneous=True
    )
    if not vg.certified:
        return LocalGerm(
            composed,
            vg.variables,
            vg.target,
            None,
            vg.domain,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "inner germ map was not certified",
        )
    if vg.image_constraint is not sp.S.true and vg.image_constraint != sp.S.true:
        image_outer = outer_expr.subs(
            dict(zip(outer_variables, vg.image_variables, strict=True)),
            simultaneous=True,
        )
        reduced_image = _reduce_image_constraint_equalities(
            image_outer, vg.image_variables, vg.limits, vg.image_constraint
        )
        if reduced_image is not None:
            reduced_outer, reduced_variables, reduced_target, reduced_domain = (
                reduced_image
            )
            if reduced_variables:
                reduced_limit = limit(
                    reduced_outer,
                    reduced_variables,
                    reduced_target,
                    domain=reduced_domain,
                    return_result=True,
                )
                if reduced_limit.status is LimitStatus.PROVED:
                    return LocalGerm(
                        composed,
                        vg.variables,
                        vg.target,
                        reduced_limit.value,
                        vg.domain,
                        AdvancedLimitStatus.CERTIFIED,
                        "vector_image_constraint_composition",
                        "outer limit certified after exact image-equality reduction",
                    )
        # Prefer stratum-wise certification: smaller Boolean cells are often
        # substantially cheaper than sending their disjunction back through CAD.
        if len(vg.image_strata) > 1:
            stratum_values = []
            all_proved = True
            for stratum in vg.image_strata:
                stratum_limit = limit(
                    image_outer,
                    vg.image_variables,
                    vg.limits,
                    domain=stratum.constraint,
                    return_result=True,
                )
                if stratum_limit.status is not LimitStatus.PROVED:
                    all_proved = False
                    break
                stratum_values.append(stratum_limit.value)
            if all_proved and stratum_values:
                first = stratum_values[0]
                if all(sp.simplify(value - first) == 0 for value in stratum_values[1:]):
                    return LocalGerm(
                        composed,
                        vg.variables,
                        vg.target,
                        first,
                        vg.domain,
                        AdvancedLimitStatus.CERTIFIED,
                        "vector_image_strata_composition",
                        "outer limit certified with the same value on every accumulating image stratum",
                    )

        outer_on_image = limit(
            image_outer,
            vg.image_variables,
            vg.limits,
            domain=vg.image_constraint,
            return_result=True,
        )
        if outer_on_image.status is LimitStatus.PROVED:
            return LocalGerm(
                composed,
                vg.variables,
                vg.target,
                outer_on_image.value,
                vg.domain,
                AdvancedLimitStatus.CERTIFIED,
                "vector_image_constraint_composition",
                "outer limit certified on a certified constraint containing the local image",
            )
    image_limit = limit(
        composed, vg.variables, vg.target, domain=vg.domain, return_result=True
    )
    if image_limit.status is LimitStatus.PROVED:
        return LocalGerm(
            composed,
            vg.variables,
            vg.target,
            image_limit.value,
            vg.domain,
            AdvancedLimitStatus.CERTIFIED,
            "vector_image_germ_composition",
            "the composed expression has a certified limit on the actual inner germ image",
        )
    outer_limit = limit(outer_expr, outer_variables, vg.limits, return_result=True)
    if outer_limit.status is not LimitStatus.PROVED:
        return LocalGerm(
            composed,
            vg.variables,
            vg.target,
            None,
            vg.domain,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "neither the actual image germ nor the ambient outer germ was certified",
        )
    return LocalGerm(
        composed,
        vg.variables,
        vg.target,
        outer_limit.value,
        vg.domain,
        AdvancedLimitStatus.CERTIFIED,
        "vector_local_germ_composition",
        "joint inner germ map composed through a certified outer simultaneous germ",
    )


def vector_limit(expressions, variables, target, *, domain=True, return_result=False):
    """Return a first-class certified simultaneous limit of a real vector map."""
    germ = certify_vector_local_germ(expressions, variables, target, domain=domain)
    result = VectorLimitResult(
        germ.expressions,
        germ.variables,
        germ.target,
        germ.limits if germ.certified else None,
        germ.domain,
        AdvancedLimitStatus.CERTIFIED
        if germ.certified
        else AdvancedLimitStatus.UNKNOWN,
        germ.provider if germ.certified else "none",
        "all vector components converge jointly on the common approach domain"
        if germ.certified
        else "at least one vector component limit is not certified",
        germ,
    )
    return result if return_result else (result.value if result.certified else result)


def vector_cluster_set(expressions, variables, target, *, domain=True):
    """Return a certified joint vector cluster set when currently representable.

    A convergent vector germ has the exact singleton joint cluster set.  The
    result stays UNKNOWN rather than replacing joint dependence by
    a Cartesian product of scalar cluster intervals.
    """
    expressions = tuple(map(sp.sympify, expressions))
    if any(e.has(sp.sin, sp.cos) for e in expressions):
        joint = joint_cluster_geometry(expressions, variables, target, domain=domain)
        if joint.certified:
            return VectorClusterSetResult(
                joint.expressions,
                joint.variables,
                joint.target,
                joint.cluster_set,
                AdvancedLimitStatus.CERTIFIED,
                joint.provider,
                joint.statement,
                None,
                coverage=joint.coverage,
            )

    limit = None
    if domain in (sp.S.true, True):
        limit = vector_limit(
            expressions, variables, target, domain=domain, return_result=True
        )
    if (
        limit is not None
        and limit.certified
        and all(
            not isinstance(v, sp.Set) and not v.has(sp.AccumBounds) for v in limit.value
        )
    ):
        point = sp.Tuple(*limit.value)
        return VectorClusterSetResult(
            limit.expressions,
            limit.variables,
            limit.target,
            sp.FiniteSet(point),
            AdvancedLimitStatus.CERTIFIED,
            "vector_limit_singleton_cluster",
            "a convergent vector germ has exactly one joint cluster value",
            limit.image_germ,
            coverage=CoverageCertificate.complete(
                "vector_limit_neighborhood_cover",
                "the certified vector limit applies to every admissible approach",
                ("relative_neighborhood",),
            ),
        )
    germ = limit.image_germ
    newton = vector_newton_fan_cluster_set(
        expressions, variables, target, domain=domain
    )
    if newton.certified:
        return newton
    # General correlated fallback: algebraic angular graphs, restricted-domain
    # strata, and lifted oscillatory phases share one limiting-image engine.
    joint = joint_cluster_geometry(expressions, variables, target, domain=domain)
    if joint.certified:
        return VectorClusterSetResult(
            tuple(map(sp.sympify, expressions)),
            limit.variables,
            limit.target,
            joint.cluster_set,
            AdvancedLimitStatus.CERTIFIED,
            joint.provider,
            joint.statement,
            germ,
            coverage=joint.coverage,
        )
    return VectorClusterSetResult(
        tuple(map(sp.sympify, expressions)),
        limit.variables,
        limit.target,
        newton.cluster_set,
        AdvancedLimitStatus.UNKNOWN,
        newton.provider,
        "joint cluster geometry was only partially certified",
        germ,
    )


def _cluster_interval_for_weight(expr, variables, target, domain, weight):
    """Certify one weighted angular cluster interval, or return None."""
    from .angular_extrema import _angular_squared_extremum, _certified_extremum

    data = _homogeneous_angular(expr, variables, target, weight)
    if data is None:
        return None
    angular, u = data
    _num, den = sp.fraction(angular)
    dmin = _angular_squared_extremum(sp.expand(den**2), u, kind="min")
    if dmin is None or bounded_ask(sp.Q.positive(dmin[0])) is not True:
        return None
    sphere = sp.Eq(sp.Add(*(x**2 for x in u)), 1)
    angular_domain = sphere
    if domain not in (sp.S.true, True):
        from .blowup_geometry import weighted_spherical_atlas
        from .domain_cluster_geometry import transform_domain_to_chart

        geometry = weighted_spherical_atlas(variables, target, weight)
        chart = geometry.charts[0]
        transformed = transform_domain_to_chart(domain, chart)
        replacements = dict(zip(chart.angular_variables, u, strict=True))
        angular_domain = sp.And(sphere, transformed.xreplace(replacements))
    lo = _certified_extremum(angular, angular_domain, u, kind="min")
    hi = _certified_extremum(angular, angular_domain, u, kind="max")
    if lo is None or hi is None:
        return None
    return sp.Interval(sp.simplify(lo[0]), sp.simplify(hi[0]))


def correlated_directional_outer_cluster_set(expr, variables, target, *, domain=True):
    """Certify radial/directional outer singularities from an exact angular image.

    Supported theorem: ``rho/acos(g(u))`` where ``rho`` is the Euclidean
    radius and ``g`` is a degree-zero directional germ whose certified angular
    image contains a full one-sided interval ending at 1.  Radial and angular
    coordinates are independent on the punctured ball.  Since
    ``acos(g)`` continuously fills ``[0, eps)`` near that endpoint, the ratio
    has both cluster value 0 and positive finite cluster values, hence no
    limit.  The angular endpoint/interval claim is obtained from semialg's
    correlated angular-map image, not sampled paths.
    """
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr = sp.sympify(expr)
    domain = sp.sympify(domain)
    if domain is not sp.S.true or any(a != 0 for a in target) or len(variables) < 2:
        return None
    rho = sp.sqrt(sp.Add(*(v**2 for v in variables)))
    num, den = sp.fraction(expr)
    if sp.simplify(num - rho) != 0 or den.func is not sp.acos:
        return None
    direction = den.args[0]
    u = sp.symbols(f"_dir0:{len(variables)}", real=True)
    directional = sp.simplify(
        direction.subs(
            {v: ui for v, ui in zip(variables, u, strict=True)}, simultaneous=True
        )
    )
    # Evaluate the degree-zero directional germ on the unit sphere.
    unit_rho = sp.sqrt(sp.Add(*(ui**2 for ui in u)))
    directional = sp.simplify(directional.subs(unit_rho, sp.S.One))
    if any(v in directional.free_symbols for v in variables):
        return None
    try:
        from semialg import angular_map_image, structured_proof_diagnostics

        image = angular_map_image((directional,), u)
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        sp.PolynomialError,
    ):
        return None
    if not image.certified or len(image.image_variables) != 1:
        return None
    z = image.image_variables[0]
    # Require the exact image formula to certify the canonical closed interval
    # [-1,1].  Equivalent more complicated images decline.
    canonical = sp.And(z >= -1, z <= 1)
    try:
        from semialg import implies

        same = (
            implies(image.formula, canonical) is True
            and implies(canonical, image.formula) is True
        )
    except (ImportError, TypeError, ValueError, NotImplementedError):
        same = sp.simplify(sp.Equivalent(image.formula, canonical)) is sp.S.true
    if not same:
        return None
    trace = structured_proof_diagnostics(image)
    coverage = CoverageCertificate.complete(
        "semialg_correlated_directional_outer_image",
        "exact correlated angular image is [-1,1]; radial and angular coordinates independently cover the punctured ball",
        covered=("angular_image[-1,1]", "radial_interval(0,epsilon)"),
    )
    return ClusterSetResult(
        expr,
        variables,
        target,
        sp.Interval(0, sp.oo),
        sp.S.Zero,
        sp.oo,
        AdvancedLimitStatus.CERTIFIED,
        "semialg_correlated_directional_outer_image:" + ">".join(trace.steps),
        "acos denominator approaches its directional zero through a complete angular interval independently of radius, so rho/acos(direction) has cluster set [0,oo)",
        coverage=coverage,
    )


def newton_fan_cluster_set(expr, variables, target, *, domain=True) -> ClusterSetResult:
    """Union certified cluster images over all discovered Newton weight regimes."""
    from .blowup_geometry import newton_valuation_rays
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr, domain = sp.sympify(expr), sp.sympify(domain)
    directional_outer = correlated_directional_outer_cluster_set(
        expr, variables, target, domain=domain
    )
    if directional_outer is not None and directional_outer.certified:
        return directional_outer
    radial_complete = multivariate_cluster_set(expr, variables, target, domain=domain)
    if radial_complete.certified:
        return radial_complete
    phase_complete = oscillatory_phase_cluster_theorem(
        expr, variables, target, domain=domain
    )
    if phase_complete is not None and phase_complete.certified:
        return phase_complete
    oscillatory_complete = oscillatory_cluster_set(
        expr, variables, target, domain=domain
    )
    if oscillatory_complete.certified:
        return oscillatory_complete
    rays, coverage, _fan = newton_valuation_rays((expr,), variables, target)
    weights = sorted(set(rays) | {(1,) * len(variables)}, key=lambda w: (sum(w), w))
    pieces = []
    used = []
    for weight in weights:
        interval = _cluster_interval_for_weight(expr, variables, target, domain, weight)
        if interval is not None:
            pieces.append(interval)
            used.append(weight)
    if not pieces:
        return multivariate_cluster_set(expr, variables, target, domain=domain)
    cluster = sp.Union(*pieces)
    # A finite normal fan is covered by the discovered radial/Newton regimes for
    # the supported rational homogeneous class.  Refuse to call a partial set
    # certified if any discovered regime could not be analyzed.
    if len(used) != len(weights) or not coverage.certified:
        return ClusterSetResult(
            expr,
            variables,
            target,
            cluster,
            None,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "newton_fan_partial_cluster_union",
            f"certified cluster pieces were obtained on {len(used)}/{len(weights)} regimes; {coverage.statement}",
        )
    infimum = sp.simplify(sp.Min(*(p.inf for p in pieces)))
    supremum = sp.simplify(sp.Max(*(p.sup for p in pieces)))
    return ClusterSetResult(
        expr,
        variables,
        target,
        cluster,
        infimum,
        supremum,
        AdvancedLimitStatus.CERTIFIED,
        "newton_fan_cluster_union",
        f"certified union of angular cluster images over {len(weights)} Newton/radial regimes",
        coverage=coverage_certificate(coverage),
    )


def _complex_holomorphic_value(expr, variables, target):
    """Direct certificate for expressions holomorphic near a finite target."""
    expr = sp.cancel(sp.sympify(expr))
    allowed = {sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh}

    def analytic(node):
        if node.is_Atom:
            return True
        if node.is_Add or node.is_Mul:
            return all(analytic(a) for a in node.args)
        if node.is_Pow:
            base, exponent = node.args
            if exponent.is_Integer:
                if exponent.is_negative:
                    bv = sp.simplify(
                        base.subs(dict(zip(variables, target, strict=True)))
                    )
                    return bv != 0 and analytic(base)
                return analytic(base)
            return False
        if node.func in allowed:
            return all(analytic(a) for a in node.args)
        return False

    if not analytic(expr):
        return None
    value = sp.simplify(expr.subs(dict(zip(variables, target, strict=True))))
    if value.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return None
    return value


def _complex_meromorphic_univariate_value(expr, variable, target):
    """Exact one-complex-variable meromorphic removable-singularity certificate."""
    expr = sp.cancel(sp.sympify(expr))
    try:
        value = bounded_limit(expr, variable, target, allow_general=True)
    except (NotImplementedError, ValueError, TypeError):
        return None
    if value is None or value.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return None
    return sp.simplify(value)


def complex_multivariate_limit(
    expr, variables, target, *, domain=True, return_result=False
):
    """Certify a complex multivariate limit through real and imaginary parts.

    Each complex approach variable z_j is represented by independent real
    coordinates x_j+i*y_j.  Both real and imaginary parts of the expression
    must have certified real simultaneous limits on the transformed domain.
    """
    from .limit_models import LimitStatus
    from .limits import limit

    variables = (
        tuple(variables) if isinstance(variables, (tuple, list)) else (variables,)
    )
    if isinstance(target, dict):
        target = tuple(sp.sympify(target[v]) for v in variables)
    elif len(variables) == 1 and not isinstance(target, (tuple, list, sp.Tuple)):
        target = (sp.sympify(target),)
    else:
        target = tuple(map(sp.sympify, target))
    if sp.sympify(domain) == sp.S.true:
        direct = _complex_holomorphic_value(expr, variables, target)
        provider = "complex_holomorphic_substitution"
        if direct is None and len(variables) == 1:
            direct = _complex_meromorphic_univariate_value(
                expr, variables[0], target[0]
            )
            provider = "complex_meromorphic_removable_singularity"
        if direct is not None:
            result = ComplexMultivariateLimitResult(
                sp.sympify(expr),
                tuple(variables),
                tuple(target),
                direct,
                AdvancedLimitStatus.CERTIFIED,
                provider,
                "certified directly in complex analytic/meromorphic coordinates",
            )
            return result if return_result else direct
        geometry = complex_meromorphic_geometry(expr, variables, target)
        if not (
            geometry.certified and geometry.classification in {"holomorphic", "zero"}
        ):
            from .multivariate_geometry_extended import complex_newton_geometry

            newton_geometry = complex_newton_geometry(expr, variables, target)
            if (
                newton_geometry.certified
                and newton_geometry.value is not None
                and newton_geometry.value not in (sp.zoo, sp.oo, -sp.oo)
            ):
                result = ComplexMultivariateLimitResult(
                    sp.sympify(expr),
                    tuple(variables),
                    tuple(target),
                    newton_geometry.value,
                    AdvancedLimitStatus.CERTIFIED,
                    newton_geometry.provider,
                    "certified by intrinsic complex Newton/divisor valuation geometry",
                )
                return result if return_result else newton_geometry.value
        if geometry.certified and geometry.classification in {"holomorphic", "zero"}:
            result = ComplexMultivariateLimitResult(
                sp.sympify(expr),
                tuple(variables),
                tuple(target),
                geometry.value,
                AdvancedLimitStatus.CERTIFIED,
                geometry.provider,
                "certified by exact complex zero/pole valuation geometry",
            )
            return result if return_result else geometry.value
    real_vars = []
    real_target = []
    subs = {}
    for i, (z, a) in enumerate(zip(variables, target, strict=True)):
        x = sp.Dummy(f"_complex_x{i}", real=True)
        y = sp.Dummy(f"_complex_y{i}", real=True)
        real_vars.extend((x, y))
        real_target.extend((sp.re(a), sp.im(a)))
        subs[z] = x + sp.I * y
    transformed = sp.expand_complex(sp.sympify(expr).subs(subs, simultaneous=True))
    transformed_domain = sp.sympify(domain).subs(subs, simultaneous=True)
    rr = limit(
        sp.re(transformed),
        tuple(real_vars),
        tuple(real_target),
        domain=transformed_domain,
        return_result=True,
    )
    ii = limit(
        sp.im(transformed),
        tuple(real_vars),
        tuple(real_target),
        domain=transformed_domain,
        return_result=True,
    )
    ok = rr.status is LimitStatus.PROVED and ii.status is LimitStatus.PROVED
    value = sp.simplify(rr.value + sp.I * ii.value) if ok else None
    result = ComplexMultivariateLimitResult(
        sp.sympify(expr),
        tuple(variables),
        tuple(target),
        value,
        AdvancedLimitStatus.CERTIFIED if ok else AdvancedLimitStatus.UNKNOWN,
        "real_imaginary_limits" if ok else "none",
        "real and imaginary parts have certified simultaneous limits"
        if ok
        else "real and imaginary part limits were not both certified",
    )
    return result if return_result else (value if ok else result)


def _condition_set(image_variables, condition):
    base = sp.ProductSet(*(sp.S.Reals for _ in image_variables))
    return sp.ConditionSet(sp.Tuple(*image_variables), sp.simplify(condition), base)


def _weighted_vector_angular_image(expressions, variables, target, weight):
    """Return the common zero-order angular vector map for one weight."""
    pieces = []
    angular_variables = None
    for expression in expressions:
        data = _homogeneous_angular(expression, variables, target, weight)
        if data is None:
            return None
        angular, u = data
        if angular_variables is None:
            angular_variables = u
        else:
            angular = angular.subs(
                dict(zip(u, angular_variables, strict=True)), simultaneous=True
            )
        pieces.append(sp.cancel(angular))
    return tuple(pieces), angular_variables


def _semialgebraic_angular_vector_image(expressions, angular_variables):
    """Project a rational angular map of the unit sphere to exact image coordinates."""
    image_variables = tuple(
        sp.Dummy(f"_cluster_{i}", real=True) for i in range(len(expressions))
    )
    # Fast exact theorem for the canonical squared-direction simplex map.
    # On the unit sphere, (u_i^2 / sum u_j^2)_i has exactly the probability
    # simplex as its image; no CAD is needed.
    sphere_sum = sp.Add(*(u**2 for u in angular_variables))
    if len(expressions) == len(angular_variables) and all(
        sp.simplify(e - u**2 / sphere_sum) == 0
        for e, u in zip(expressions, angular_variables, strict=True)
    ):
        condition = sp.And(
            sp.Eq(sp.Add(*image_variables), 1),
            *(sp.Ge(z, 0) for z in image_variables),
        )
        return (
            _condition_set(image_variables, condition),
            condition,
            "squared_direction_simplex",
        )
    # Prefer semialg's certified correlated angular-map image.  This keeps the
    # output coordinates correlated and exposes a structured proof trace.
    try:
        from semialg import angular_map_image, structured_proof_diagnostics

        exact = angular_map_image(
            expressions, angular_variables, image_variables=image_variables
        )
        if exact.certified:
            trace = structured_proof_diagnostics(exact)
            provider = "semialg:" + ">".join(trace.steps)
            return (
                _condition_set(image_variables, exact.formula),
                exact.formula,
                provider,
            )
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        TypeError,
        ValueError,
        NotImplementedError,
        sp.PolynomialError,
    ):
        pass
    graph = _rational_image_graph(expressions, angular_variables, image_variables)
    if graph is None:
        return None
    sphere = sp.Eq(sp.Add(*(u**2 for u in angular_variables)), 1)
    planned = _planned_image_projection(
        sp.And(sphere, graph), image_variables, angular_variables
    )
    if planned is None:
        return None
    condition, provider = planned
    return _condition_set(image_variables, condition), condition, provider


def vector_newton_fan_cluster_set(expressions, variables, target, *, domain=True):
    """Certify correlated joint cluster geometry over Newton/radial regimes.

    Each zero-order angular vector map is projected jointly, so algebraic
    correlations between components are retained instead of replacing the image
    by a Cartesian product of scalar cluster sets.
    """
    from .blowup_geometry import newton_valuation_rays
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expressions = tuple(map(sp.sympify, expressions))
    domain = sp.sympify(domain)
    if domain not in (sp.S.true, True):
        return VectorClusterSetResult(
            expressions,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "relative-domain vector Newton cluster geometry is not yet certified",
        )
    rays, coverage, _fan = newton_valuation_rays(expressions, variables, target)
    weights = set(rays) | {(1,) * len(variables)}
    pieces = []
    used = []
    for weight in sorted(weights, key=lambda w: (sum(w), w)):
        data = _weighted_vector_angular_image(expressions, variables, target, weight)
        if data is None:
            continue
        angular, u = data
        projected = _semialgebraic_angular_vector_image(angular, u)
        if projected is None:
            continue
        pieces.append(projected[0])
        used.append(weight)
    if not pieces or len(used) != len(weights) or not coverage.certified:
        return VectorClusterSetResult(
            expressions,
            variables,
            target,
            sp.Union(*pieces) if pieces else None,
            AdvancedLimitStatus.UNKNOWN,
            "vector_newton_partial_cluster_union" if pieces else "none",
            f"certified joint images on {len(used)}/{len(weights)} Newton/radial regimes",
        )
    return VectorClusterSetResult(
        expressions,
        variables,
        target,
        sp.Union(*pieces),
        AdvancedLimitStatus.CERTIFIED,
        "vector_newton_joint_cluster_union",
        f"certified correlated joint cluster geometry on {len(weights)} Newton/radial regimes",
        coverage=coverage_certificate(coverage),
    )


def _positive_radial_phase_denominator(expr, variables, target):
    """Recognize q>0 with q->0 in phases 1/q near a finite target."""
    expr = sp.factor(sp.sympify(expr))
    subs = dict(zip(variables, target, strict=True))
    if sp.simplify(expr.subs(subs)) != 0:
        return False
    try:
        sp.Poly(sp.expand(expr), *variables)
    except sp.PolynomialError:
        return False
    # Conservative certificate: a sum of positive even pure monomials after
    # translation.  This covers Euclidean/weighted radial phases without making
    # an unproved positivity inference.
    shifted = sp.Poly(
        sp.expand(expr.subs({v: v + a for v, a in subs.items()})), *variables
    )
    for monom, coeff in shifted.terms():
        if coeff.is_positive is not True or any(power % 2 for power in monom):
            return False
    return bool(shifted.terms())


def oscillatory_cluster_set(expr, variables, target, *, domain=True):
    """Certify basic nonvanishing oscillatory scalar cluster sets.

    Currently recognizes sin(1/q) and cos(1/q), where q is a certified positive
    radial polynomial tending to zero.  Their exact cluster set is [-1, 1].
    """
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr = sp.sympify(expr)
    if sp.sympify(domain) not in (sp.S.true, True):
        return ClusterSetResult(
            expr,
            variables,
            target,
            None,
            None,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "relative-domain oscillatory phase accumulation is not certified",
        )
    direct_phase = oscillatory_phase_cluster_theorem(
        expr, variables, target, domain=domain
    )
    if direct_phase is not None and direct_phase.certified:
        return direct_phase
    # Prefer a constant-amplitude atom. Unordered set iteration can otherwise
    # select a vanishing perturbation rather than the persistent oscillator,
    # making the early complete-cluster theorem depend on the hash seed.
    oscillatory_atoms = sorted(
        expr.atoms(sp.sin, sp.cos),
        key=lambda atom: (expr.coeff(atom).has(*variables), sp.default_sort_key(atom)),
    )
    oscillatory_atom = oscillatory_atoms[0] if oscillatory_atoms else None
    if oscillatory_atom is not None and expr != oscillatory_atom and expr.is_Add:
        coefficient = sp.simplify(expr.coeff(oscillatory_atom))
        remainder = sp.simplify(expr - coefficient * oscillatory_atom)
        # Try the cheap uniform envelope before the general local-germ
        # pipeline.  Radial oscillation plus an elementary vanishing remainder
        # is common and should not pay for algebraic/semialgebraic geometry.
        cheap_remainder = cheap_uniform_vanishing_limit(
            remainder, variables, target, domain=domain
        )
        rgerm = (
            None
            if cheap_remainder.certified
            else certify_local_germ(remainder, variables, target, domain=domain)
        )
        cgerm = certify_local_germ(coefficient, variables, target, domain=domain)
        base = oscillatory_cluster_set(
            oscillatory_atom, variables, target, domain=domain
        )
        remainder_zero = cheap_remainder.certified or (
            rgerm is not None and rgerm.certified and rgerm.limit == 0
        )
        if (
            remainder_zero
            and cgerm.certified
            and cgerm.limit is not None
            and cgerm.limit.is_real is True
            and base.certified
        ):
            radius = sp.Abs(cgerm.limit)
            return ClusterSetResult(
                expr,
                variables,
                target,
                sp.Interval(-radius, radius),
                -radius,
                radius,
                AdvancedLimitStatus.CERTIFIED,
                "vanishing_perturbation_cluster",
                "a convergent real coefficient and uniformly vanishing additive perturbation preserve the scaled complete oscillatory cluster interval",
                coverage=base.coverage,
            )
    if oscillatory_atom is not None and expr != oscillatory_atom:
        amplitude = sp.cancel(expr / oscillatory_atom)
        germ = certify_local_germ(amplitude, variables, target, domain=domain)
        if germ.certified and sp.sympify(germ.limit).is_real is True:
            base = oscillatory_cluster_set(
                oscillatory_atom, variables, target, domain=domain
            )
            if base.certified:
                radius = sp.Abs(germ.limit)
                return ClusterSetResult(
                    expr,
                    variables,
                    target,
                    sp.Interval(-radius, radius),
                    -radius,
                    radius,
                    AdvancedLimitStatus.CERTIFIED,
                    "convergent_amplitude_oscillatory_cluster",
                    "a convergent real amplitude scales the full sine/cosine phase cluster interval",
                    coverage=base.coverage,
                )
    if expr.func in (sp.sin, sp.cos):
        phase = sp.cancel(expr.args[0])
        num, den = sp.fraction(phase)
        if sp.simplify(num) in (1, -1) and _positive_radial_phase_denominator(
            den, variables, target
        ):
            return ClusterSetResult(
                expr,
                variables,
                target,
                sp.Interval(-1, 1),
                -1,
                1,
                AdvancedLimitStatus.CERTIFIED,
                "radial_oscillatory_phase_accumulation",
                "the reciprocal positive radial phase attains arbitrarily large values, so sine/cosine accumulate on [-1,1]",
                coverage=CoverageCertificate.complete(
                    "positive_radial_phase_cover",
                    "the positive radial phase denominator tends to zero along every punctured approach and its reciprocal exhausts the tail",
                    ("punctured_neighborhood",),
                ),
            )
    return ClusterSetResult(
        expr,
        variables,
        target,
        None,
        None,
        None,
        AdvancedLimitStatus.UNKNOWN,
        "none",
        "supported oscillatory phase accumulation was not recognized",
    )


def oscillatory_vector_cluster_set(expressions, variables, target, *, domain=True):
    """Certify the unit-circle cluster set for matched cosine/sine radial phases."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expressions = tuple(map(sp.sympify, expressions))
    if len(expressions) != 2 or sp.sympify(domain) not in (sp.S.true, True):
        return VectorClusterSetResult(
            expressions,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "none",
            "supported oscillatory vector geometry was not recognized",
        )
    a, b = expressions
    pair = None
    if (
        a.func is sp.cos
        and b.func is sp.sin
        and sp.simplify(a.args[0] - b.args[0]) == 0
    ) or (
        a.func is sp.sin
        and b.func is sp.cos
        and sp.simplify(a.args[0] - b.args[0]) == 0
    ):
        pair = a.args[0]
    if pair is not None:
        num, den = sp.fraction(sp.cancel(pair))
        if sp.simplify(num) in (1, -1) and _positive_radial_phase_denominator(
            den, variables, target
        ):
            u, v = sp.symbols("_osc_u _osc_v", real=True)
            circle = _condition_set((u, v), sp.Eq(u**2 + v**2, 1))
            return VectorClusterSetResult(
                expressions,
                variables,
                target,
                circle,
                AdvancedLimitStatus.CERTIFIED,
                "radial_oscillatory_circle",
                "matched sine/cosine phase accumulates on the full unit circle",
                coverage=CoverageCertificate.complete(
                    "matched_radial_phase_cover",
                    "the common reciprocal radial phase exhausts all phase angles modulo 2*pi",
                    ("phase_circle",),
                ),
            )
    return VectorClusterSetResult(
        expressions,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        "none",
        "supported oscillatory vector geometry was not recognized",
    )


def projective_cluster_atlas(expr, variables, ends, *, domain=True):
    """Assemble cluster geometry from a finite atlas of signed projective ends."""
    from .limit_primitives import _normalize_variables

    variables = _normalize_variables(variables)
    normalized_ends = tuple(_normalize_extended_target(variables, end) for end in ends)
    charts, pieces = [], []
    for end in normalized_ends:
        chart = projective_limit_chart(expr, variables, end, domain=domain)
        charts.append(chart)
        if len(variables) == 1 and sp.sympify(domain) in (sp.S.true, True):
            try:
                direct = bounded_limit(
                    sp.sympify(expr), variables[0], end[0], allow_general=True
                )
            except (NotImplementedError, ValueError, TypeError):
                direct = None
            if direct is not None and not sp.sympify(direct).has(sp.zoo, sp.nan):
                pieces.append(sp.FiniteSet(sp.simplify(direct)))
                continue
        result = newton_fan_cluster_set(
            chart.expression, chart.variables, chart.target, domain=chart.domain
        )
        if not result.certified:
            result = multivariate_cluster_set(
                chart.expression, chart.variables, chart.target, domain=chart.domain
            )
        if not result.certified:
            return ProjectiveClusterAtlasResult(
                sp.sympify(expr),
                variables,
                normalized_ends,
                sp.Union(*pieces) if pieces else None,
                AdvancedLimitStatus.UNKNOWN,
                "projective_partial_cluster_atlas",
                f"certified {len(pieces)}/{len(normalized_ends)} projective ends",
                tuple(charts),
            )
        pieces.append(result.cluster_set)
    return ProjectiveClusterAtlasResult(
        sp.sympify(expr),
        variables,
        normalized_ends,
        sp.Union(*pieces),
        AdvancedLimitStatus.CERTIFIED,
        "projective_cluster_atlas",
        f"certified cluster union over {len(charts)} signed projective ends",
        tuple(charts),
        coverage=CoverageCertificate.complete(
            "requested_projective_end_cover",
            "every requested signed projective end has a certified cluster image",
            normalized_ends,
        ),
    )


def _polynomial_vanishing_order(poly, variables, target):
    from .blowup_geometry import Valuation

    shifted = sp.expand(
        sp.sympify(poly).subs(
            {v: v + a for v, a in zip(variables, target, strict=True)}
        )
    )
    try:
        value = Valuation(tuple(variables), (1,) * len(variables)).polynomial(shifted)
    except ValueError:
        return None
    return None if value is sp.oo else value


def _coordinate_monomial_valuation(poly, variables, target):
    """Factor the largest coordinate monomial using the common valuation object."""
    from .blowup_geometry import Valuation

    shifted = sp.expand(
        sp.sympify(poly).subs(
            {v: v + a for v, a in zip(variables, target, strict=True)}
        )
    )
    valuation_engine = Valuation(tuple(variables), (1,) * len(variables))
    orders = valuation_engine.coordinate_orders(shifted)
    if orders is None:
        return None
    monomial = sp.Mul(*(v**k for v, k in zip(variables, orders, strict=True)))
    return orders, sp.cancel(shifted / monomial)


def complex_meromorphic_geometry(expr, variables, target):
    """Classify polynomial/rational complex germs by exact local valuations.

    Cancellation is performed first.  A denominator nonzero at the target gives
    a holomorphic quotient.  In one variable, numerator/denominator vanishing
    orders additionally certify removable zeros and poles without realification.
    """
    variables = (
        tuple(variables) if isinstance(variables, (tuple, list)) else (variables,)
    )
    target = _normalize_extended_target(variables, target)
    expr = sp.cancel(sp.together(sp.sympify(expr)))
    num, den = map(sp.expand, sp.fraction(expr))
    no = _polynomial_vanishing_order(num, variables, target)
    do = _polynomial_vanishing_order(den, variables, target)
    if no is None or do is None:
        return ComplexMeromorphicGeometryResult(
            expr,
            variables,
            target,
            no,
            do,
            "unknown",
            None,
            AdvancedLimitStatus.UNKNOWN,
            "none",
        )
    subs = dict(zip(variables, target, strict=True))
    den0 = sp.simplify(den.subs(subs))
    if den0 != 0:
        value = sp.simplify(num.subs(subs) / den0)
        return ComplexMeromorphicGeometryResult(
            expr,
            variables,
            target,
            no,
            0,
            "holomorphic",
            value,
            AdvancedLimitStatus.CERTIFIED,
            "complex_rational_regular_germ",
        )
    if len(variables) == 1:
        if no > do:
            return ComplexMeromorphicGeometryResult(
                expr,
                variables,
                target,
                no,
                do,
                "zero",
                sp.S.Zero,
                AdvancedLimitStatus.CERTIFIED,
                "complex_univariate_valuation",
            )
        if no < do:
            return ComplexMeromorphicGeometryResult(
                expr,
                variables,
                target,
                no,
                do,
                "pole",
                sp.zoo,
                AdvancedLimitStatus.CERTIFIED,
                "complex_univariate_valuation",
            )
    else:
        nv = _coordinate_monomial_valuation(num, variables, target)
        dv = _coordinate_monomial_valuation(den, variables, target)
        if nv is not None and dv is not None:
            exponents = tuple(a - b for a, b in zip(nv[0], dv[0], strict=True))
            zero_subs = dict.fromkeys(variables, 0)
            den_unit = sp.simplify(dv[1].subs(zero_subs))
            num_unit = sp.simplify(nv[1].subs(zero_subs))
            if den_unit != 0 and num_unit != 0:
                if all(e >= 0 for e in exponents) and any(e > 0 for e in exponents):
                    return ComplexMeromorphicGeometryResult(
                        expr,
                        variables,
                        target,
                        no,
                        do,
                        "normal_crossing_zero",
                        sp.S.Zero,
                        AdvancedLimitStatus.CERTIFIED,
                        "complex_normal_crossing_valuation",
                    )
                if all(e == 0 for e in exponents):
                    value = sp.simplify(num_unit / den_unit)
                    return ComplexMeromorphicGeometryResult(
                        expr,
                        variables,
                        target,
                        no,
                        do,
                        "normal_crossing_removable",
                        value,
                        AdvancedLimitStatus.CERTIFIED,
                        "complex_normal_crossing_valuation",
                    )
                if any(e < 0 for e in exponents) and all(e <= 0 for e in exponents):
                    return ComplexMeromorphicGeometryResult(
                        expr,
                        variables,
                        target,
                        no,
                        do,
                        "normal_crossing_pole",
                        sp.zoo,
                        AdvancedLimitStatus.CERTIFIED,
                        "complex_normal_crossing_valuation",
                    )
    return ComplexMeromorphicGeometryResult(
        expr,
        variables,
        target,
        no,
        do,
        "meromorphic_unresolved",
        None,
        AdvancedLimitStatus.UNKNOWN,
        "complex_multivariate_valuation",
    )


__all__ = [
    "AdvancedLimitStatus",
    "ClusterSetResult",
    "ComplexMeromorphicGeometryResult",
    "ComplexMultivariateLimitResult",
    "CoordinateChart",
    "LocalGerm",
    "NewtonFanLimitResult",
    "ProjectiveClusterAtlasResult",
    "RelativeLimitResult",
    "VectorClusterSetResult",
    "VectorLimitResult",
    "VectorLocalGerm",
    "bounded_factor_envelope_limit",
    "certify_local_germ",
    "certify_vector_local_germ",
    "complex_meromorphic_geometry",
    "complex_multivariate_limit",
    "compose_local_germ",
    "compose_vector_local_germ",
    "extended_limit",
    "extended_multivariate_cluster_set",
    "extended_newton_fan_limit",
    "multivariate_cluster_set",
    "multivariate_liminf",
    "multivariate_limsup",
    "newton_fan_cluster_set",
    "newton_fan_limit",
    "oscillatory_cluster_set",
    "oscillatory_vector_cluster_set",
    "projective_cluster_atlas",
    "projective_limit_chart",
    "relative_limit",
    "vector_cluster_set",
    "vector_limit",
    "vector_newton_fan_cluster_set",
]


def _radial_leading_value(expr, variables, target, weight, r, angular):
    """Return the finite exceptional-divisor value using a common blow-up chart."""
    from .blowup_geometry import BlowUpChart, Valuation

    chart = BlowUpChart(
        tuple(variables),
        tuple(target),
        Valuation(tuple(variables), tuple(weight)),
        r,
        tuple(angular),
        sp.Eq(sp.Add(*(u**2 for u in angular)), 1),
    )
    shifted = chart.transform(expr)
    try:
        value = sp.simplify(
            bounded_limit(shifted, r, 0, direction="+", allow_general=True)
        )
    except (NotImplementedError, ValueError, TypeError, RecursionError):
        return None
    if value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    return value


def _radial_relation_limit(rel, variables, target, weight, r, angular):
    """Take the leading positive-r germ of a polynomial relation."""
    if rel in (sp.S.true, sp.S.false, True, False):
        return sp.sympify(rel)
    if isinstance(rel, sp.And):
        parts = [
            _radial_relation_limit(a, variables, target, weight, r, angular)
            for a in rel.args
        ]
        return None if any(p is None for p in parts) else sp.And(*parts)
    if isinstance(rel, sp.Or):
        parts = [
            _radial_relation_limit(a, variables, target, weight, r, angular)
            for a in rel.args
        ]
        return None if any(p is None for p in parts) else sp.Or(*parts)
    if isinstance(rel, sp.Not):
        part = _radial_relation_limit(
            rel.args[0], variables, target, weight, r, angular
        )
        return None if part is None else sp.Not(part)
    if not getattr(rel, "is_Relational", False):
        return None
    delta = sp.expand(rel.lhs - rel.rhs).subs(
        {
            v: a + r**w * u
            for v, a, w, u in zip(variables, target, weight, angular, strict=True)
        },
        simultaneous=True,
    )
    try:
        poly = sp.Poly(sp.expand(delta), r)
    except sp.PolynomialError:
        return None
    terms = [(m[0], c) for m, c in poly.terms() if c != 0]
    if not terms:
        lead = sp.S.Zero
    else:
        degree = min(k for k, _ in terms)
        lead = sp.simplify(next(c for k, c in terms if k == degree))
    cls = rel.func
    return cls(lead, 0)


def _on_unit_sphere(expr, angular):
    """Simplify an angular expression using sum(u_i**2)=1 before graphing."""
    sphere_sum = sp.Add(*(u**2 for u in angular))
    expr = sp.factor(sp.cancel(sp.together(expr)))
    # Structural replacement also catches square roots of the sphere norm.
    expr = expr.xreplace({sphere_sum: sp.S.One, sp.sqrt(sphere_sum): sp.S.One})
    expr = sp.simplify(expr.subs(sphere_sum, 1))
    # A final powsimp often combines sqrt(A)/sqrt(sphere_sum) before substitution.
    expr = analytic_powsimp(expr)
    return sp.simplify(expr.subs(sphere_sum, 1))


def _semialgebraic_limiting_image(expressions, angular, angular_domain):
    """Project a finite algebraic angular map over a semialgebraic angular domain."""
    expressions = tuple(_on_unit_sphere(e, angular) for e in expressions)
    # Fast exact 2D even-angular parameterization.  The unit circle modulo sign
    # flips is t=u0**2 in [0,1], u1**2=1-t.  Keeping this as an ImageSet
    # preserves component correlations and avoids an unnecessary CAD image
    # projection for a very common Newton stratum.
    if len(angular) == 2:
        t = sp.Dummy("_angular_t", real=True)
        parameterized = []
        for expression in expressions:
            candidate = sp.expand(expression).subs(angular[1] ** 2, 1 - angular[0] ** 2)
            candidate = sp.simplify(candidate.subs(angular[0] ** 2, t))
            if candidate.has(*angular):
                parameterized = []
                break
            parameterized.append(candidate)
        sign_only_domain = angular_domain in (sp.S.true, True)
        if not sign_only_domain:
            atoms = (
                angular_domain.args
                if isinstance(angular_domain, sp.And)
                else (angular_domain,)
            )
            sign_only_domain = all(
                getattr(a, "is_Relational", False)
                and (a.lhs in angular or a.rhs in angular)
                and (a.lhs == 0 or a.rhs == 0)
                for a in atoms
            )
        if parameterized and sign_only_domain:
            image = sp.ImageSet(
                sp.Lambda(t, sp.Tuple(*parameterized)), sp.Interval(0, 1)
            )
            return image, sp.S.true, "even_2d_sphere_parameterization"
    # Exact higher-dimensional stereographic chart.  For an unrestricted
    # sphere this preserves the *joint* image without coordinate-wise bounds or
    # QE: R^(n-1) covers the sphere minus one pole, which is added explicitly.
    if len(angular) >= 3 and angular_domain in (sp.S.true, True):
        ts = tuple(
            sp.Dummy(f"_stereo_t{i}", real=True) for i in range(len(angular) - 1)
        )
        r2 = sp.Add(*(t**2 for t in ts))
        den = 1 + r2
        subs = {u: 2 * t / den for u, t in zip(angular[:-1], ts, strict=True)}
        subs[angular[-1]] = (1 - r2) / den
        vals = tuple(sp.cancel(e.subs(subs)) for e in expressions)
        pole_subs = dict.fromkeys(angular, sp.S.Zero)
        pole_subs[angular[-1]] = sp.S.NegativeOne
        pole = sp.Tuple(*(sp.simplify(e.subs(pole_subs)) for e in expressions))
        image = sp.ImageSet(sp.Lambda(ts, sp.Tuple(*vals)), *([sp.S.Reals] * len(ts)))
        return (
            sp.Union(image, sp.FiniteSet(pole)),
            sp.S.true,
            "stereographic_sphere_image",
        )
    image_variables = tuple(
        sp.Dummy(f"_joint_cluster_{i}", real=True) for i in range(len(expressions))
    )
    graph_data = _semialgebraic_map_graph(expressions, angular, image_variables)
    if graph_data is None:
        return None
    graph, auxiliaries, _ = graph_data
    sphere = sp.Eq(sp.Add(*(u**2 for u in angular)), 1)
    source = sp.And(sphere, angular_domain, graph)
    planned = _planned_image_projection(
        source, image_variables, (*angular, *auxiliaries)
    )
    if planned is None:
        return None
    condition, provider = planned
    condition = sp.simplify(condition)
    if condition in (sp.S.false, False):
        return None
    return _condition_set(image_variables, condition), condition, provider


def _oscillatory_lift(expressions, variables, target, weight, domain):
    """Lift one common divergent sine/cosine phase to a circle variable."""
    atoms = sorted(
        set().union(*(e.atoms(sp.sin, sp.cos) for e in expressions)),
        key=sp.default_sort_key,
    )
    if not atoms:
        return None
    phases = {sp.simplify(a.args[0]) for a in atoms}
    if len(phases) != 1:
        return None
    phase = next(iter(phases))
    num, den = sp.fraction(sp.cancel(phase))
    if sp.simplify(num) not in (1, -1) or not _positive_radial_phase_denominator(
        den, variables, target
    ):
        return None
    c, s = sp.symbols("_phase_c _phase_s", real=True)
    repl = {}
    for atom in atoms:
        repl[atom] = s if atom.func is sp.sin else c
    lifted = tuple(sp.expand(e.xreplace(repl)) for e in expressions)
    # The nonoscillatory coefficients must possess finite weighted angular limits.
    r = sp.Dummy("_cluster_r", positive=True)
    angular = tuple(
        sp.Dummy(f"_cluster_u{i}", real=True) for i in range(len(variables))
    )
    vals = tuple(
        _radial_leading_value(e, variables, target, weight, r, angular) for e in lifted
    )
    if any(v is None for v in vals):
        return None
    angular_domain = _radial_relation_limit(
        sp.sympify(domain), variables, target, weight, r, angular
    )
    if angular_domain is None:
        return None
    source_vars = (*angular, c, s)
    image_variables = tuple(
        sp.Dummy(f"_joint_cluster_{i}", real=True) for i in range(len(vals))
    )
    graph_data = _semialgebraic_map_graph(vals, source_vars, image_variables)
    if graph_data is None:
        return None
    graph, auxiliaries, _ = graph_data
    source = sp.And(
        sp.Eq(sp.Add(*(u**2 for u in angular)), 1),
        angular_domain,
        sp.Eq(c**2 + s**2, 1),
        graph,
    )
    planned = _planned_image_projection(
        source, image_variables, (*source_vars, *auxiliaries)
    )
    if planned is None:
        return None
    condition, provider = planned
    condition = sp.simplify(condition)
    return (
        _condition_set(image_variables, condition),
        condition,
        f"oscillatory_phase_lift:{provider}",
        angular_domain,
    )


def _joint_local_stratum(fan, weight, geometry):
    """Build a subsystem-neutral Newton -> image recursive stratum."""
    from .local_strata import LocalStratum, as_local_stratum, combine_local_strata

    cone = next((c for c in fan.cones if c.representative_weight == weight), None)
    children = []
    if cone is not None:
        children.append(as_local_stratum(cone))
    if geometry is not None:
        children.extend(as_local_stratum(s) for s in geometry.strata)
    if not children:
        return LocalStratum("cluster_fiber", valuation_data=weight)
    return combine_local_strata(*children, kind="newton_cluster_fiber")


def joint_cluster_geometry(expressions, variables, target, *, domain=True):
    """Certify joint cluster geometry from Newton strata, algebraic graphs and phases.

    The engine keeps component correlations by projecting each limiting angular
    image jointly.  Supported algebraic graph coordinates are handled through
    ``semialg``; a common reciprocal radial sine/cosine phase is lifted to the
    unit circle before projection.  Restricted semialgebraic domains are reduced
    to their leading angular germs on each weight stratum.
    """
    from .blowup_geometry import newton_valuation_rays
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expressions = tuple(map(sp.sympify, expressions))
    domain = sp.sympify(domain)
    # Exact n-dimensional direction map: (x-target)/||x-target|| has the
    # whole unit sphere as its correlated cluster set.  Recognizing it before
    # Newton/CAD avoids the 3-D timeout.
    shifted = tuple(sp.expand(v - a) for v, a in zip(variables, target, strict=True))
    radius = sp.sqrt(sum(q**2 for q in shifted))
    if (
        len(expressions) == len(variables)
        and domain in (sp.S.true, True)
        and all(
            sp.simplify(e - q / radius) == 0
            for e, q in zip(expressions, shifted, strict=True)
        )
    ):
        uu = tuple(sp.Dummy(f"_sphere_u{i}", real=True) for i in range(len(variables)))
        point = sp.Tuple(*uu)
        sphere = sp.ConditionSet(
            point,
            sp.Eq(sum(q**2 for q in uu), 1),
            sp.ProductSet(*([sp.S.Reals] * len(uu))),
        )
        stratum = JointClusterStratum(
            None, sp.Eq(sum(q**2 for q in uu), 1), sphere, "unit_sphere_direction_map"
        )
        return JointClusterGeometryResult(
            expressions,
            variables,
            target,
            sphere,
            (stratum,),
            AdvancedLimitStatus.CERTIFIED,
            "joint_cluster_unit_sphere",
            "exact correlated unit-sphere direction cluster set",
            None,
            coverage=CoverageCertificate.complete(
                "unit_sphere_direction_cover",
                "every direction is attained by a radial approach",
                (len(variables),),
            ),
        )
    if any(e.has(sp.sin, sp.cos) for e in expressions):
        from .multivariate_geometry_extended import periodic_phase_cluster_geometry

        phase_image = periodic_phase_cluster_geometry(
            expressions, variables, target, domain=domain
        )
        if phase_image is not None:
            cs, phase_data, provider = phase_image
            from .local_strata import as_local_stratum

            stratum = JointClusterStratum(
                None,
                phase_data.constraint,
                cs,
                provider,
                local_stratum=as_local_stratum(phase_data),
            )
            return JointClusterGeometryResult(
                expressions,
                variables,
                target,
                cs,
                (stratum,),
                AdvancedLimitStatus.CERTIFIED,
                "joint_cluster_phase_geometry",
                f"certified correlated cluster geometry on {len(phase_data.independent_phases)} phase-circle factors",
                None,
                coverage=CoverageCertificate.complete(
                    "periodic_phase_torus_cover",
                    "the independent phase-circle factors exhaust the periodic cluster geometry",
                    tuple(range(len(phase_data.independent_phases))),
                ),
            )
    limit = None
    if domain in (sp.S.true, True):
        limit = vector_limit(
            expressions, variables, target, domain=domain, return_result=True
        )
    if (
        limit is not None
        and limit.certified
        and all(
            not isinstance(v, sp.Set) and not v.has(sp.AccumBounds) for v in limit.value
        )
    ):
        point = sp.Tuple(*limit.value)
        cs = sp.FiniteSet(point)
        stratum = JointClusterStratum(None, sp.S.true, cs, "vector_limit_singleton")
        return JointClusterGeometryResult(
            expressions,
            variables,
            target,
            cs,
            (stratum,),
            AdvancedLimitStatus.CERTIFIED,
            "joint_cluster_limit",
            "the vector germ converges",
            coverage=CoverageCertificate.complete(
                "joint_limit_neighborhood_cover",
                "the convergent vector germ covers every admissible approach",
                ("relative_neighborhood",),
            ),
        )
    from .newton_geometry import (
        structured_semialgebraic_geometry,
    )

    rays, coverage, fan = newton_valuation_rays(expressions, variables, target)
    weights = set(rays)
    if not weights:
        weights = {(1,) * len(variables)}
    pieces, strata, failed = [], [], []
    for weight in sorted(weights, key=lambda w: (sum(w), w)):
        osc = _oscillatory_lift(expressions, variables, target, weight, domain)
        if osc is not None:
            cs, cond, provider, adom = osc
            pieces.append(cs)
            geometry = structured_semialgebraic_geometry(cs)
            strata.append(
                JointClusterStratum(
                    weight,
                    cond,
                    cs,
                    provider,
                    adom,
                    geometry,
                    _joint_local_stratum(fan, weight, geometry),
                )
            )
            continue
        r = sp.Dummy("_cluster_r", positive=True)
        angular = tuple(
            sp.Dummy(f"_cluster_u{i}", real=True) for i in range(len(variables))
        )
        vals = tuple(
            _radial_leading_value(e, variables, target, weight, r, angular)
            for e in expressions
        )
        if any(v is None for v in vals):
            failed.append(weight)
            continue
        adom = _radial_relation_limit(domain, variables, target, weight, r, angular)
        if adom is None:
            failed.append(weight)
            continue
        projected = _semialgebraic_limiting_image(vals, angular, adom)
        if projected is None:
            failed.append(weight)
            continue
        cs, cond, provider = projected
        pieces.append(cs)
        geometry = structured_semialgebraic_geometry(cs)
        strata.append(
            JointClusterStratum(
                weight,
                cond,
                cs,
                f"newton_algebraic_image:{provider}",
                adom,
                geometry,
                _joint_local_stratum(fan, weight, geometry),
            )
        )
    union = sp.Union(*pieces) if pieces else None
    if failed or not pieces or not coverage.certified:
        return JointClusterGeometryResult(
            expressions,
            variables,
            target,
            union,
            tuple(strata),
            AdvancedLimitStatus.UNKNOWN,
            "joint_cluster_partial",
            f"certified {len(strata)}/{len(weights)} Newton/radial strata",
            fan,
        )
    return JointClusterGeometryResult(
        expressions,
        variables,
        target,
        union,
        tuple(strata),
        AdvancedLimitStatus.CERTIFIED,
        "joint_cluster_newton_algebraic_phase",
        f"certified correlated cluster union on {len(strata)} Newton/radial strata",
        fan,
        coverage=coverage_certificate(coverage),
    )


# ---------------------------------------------------------------------------
# Uniform envelope and phase-image theorems used by the final cluster layer.


def _shifted_substitution(variables, target, t, exponents, signs=None):
    signs = signs or (1,) * len(variables)
    return {
        v: a + s * t**e
        for v, a, s, e in zip(variables, target, signs, exponents, strict=True)
    }


def _reciprocal_phase_tail_witness(phase, variables, target):
    """Find an exact monomial approach on which a real phase exhausts a tail.

    A phase whose pullback is ``c*t**(-m) + o(t**(-m))`` with real nonzero
    ``c`` and integer ``m>0`` is continuous for sufficiently small positive
    ``t`` and its image contains a real tail.  Consequently sine/cosine of the
    phase has the full cluster interval [-1, 1].
    """
    t = sp.Dummy("_phase_t", positive=True)
    n = len(variables)
    from itertools import product

    for exponents in product(range(1, 4), repeat=n):
        for signs in product((-1, 1), repeat=n):
            pulled = sp.cancel(
                sp.sympify(phase).subs(
                    _shifted_substitution(variables, target, t, exponents, signs)
                )
            )
            if pulled.has(sp.zoo, sp.nan):
                continue
            try:
                inv = sp.cancel(1 / pulled)
                _lead = bounded_limit(inv / t, t, 0, direction="+", allow_general=True)
                # The t test is only a fast path.  Determine the exact positive
                # valuation when it is a rational pullback.
                num, den = sp.fraction(inv)
                pn, pd = sp.Poly(num, t), sp.Poly(den, t)
                on = min(m[0] for m, c in pn.terms() if c != 0)
                od = min(m[0] for m, c in pd.terms() if c != 0)
                order = on - od
                if order <= 0:
                    continue
                coeff = bounded_limit(
                    inv / t**order, t, 0, direction="+", allow_general=True
                )
                if (
                    coeff.is_real is True
                    and coeff.is_zero is False
                    and coeff.is_finite is True
                ):
                    return (exponents, signs, sp.simplify(coeff), order)
            except (
                sp.PolynomialError,
                TypeError,
                ValueError,
                ZeroDivisionError,
                NotImplementedError,
            ):
                continue
    return None


def oscillatory_phase_cluster_theorem(expr, variables, target, *, domain=True):
    """Exact scalar cluster theorem for periodic atoms with an exhausting phase.

    The proof needs only one admissible approach whose phase image contains a
    tail: global boundedness supplies the reverse inclusion.  This is stronger
    and cheaper than requiring the phase to be radial.
    """
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr = sp.sympify(expr)
    if sp.sympify(domain) not in (sp.S.true, True):
        return None
    if expr.func not in (sp.sin, sp.cos):
        return None
    witness = _reciprocal_phase_tail_witness(expr.args[0], variables, target)
    if witness is None:
        return None
    return ClusterSetResult(
        expr,
        variables,
        target,
        sp.Interval(-1, 1),
        -1,
        1,
        AdvancedLimitStatus.CERTIFIED,
        "oscillatory_phase_tail_image",
        "an exact monomial approach makes the phase continuously exhaust a real tail; periodicity and the global sine/cosine bound give the exact cluster interval [-1,1]",
        coverage=CoverageCertificate.complete(
            "periodic_tail_image",
            "the witness phase exhausts a tail and sine/cosine are globally confined to [-1,1]",
            (witness,),
        ),
    )


def _globally_bounded_factor(factor):
    """Return a certified constant absolute bound for elementary real factors."""
    factor = sp.sympify(factor)
    if factor.func in (sp.sin, sp.cos, sp.sign):
        return sp.S.One
    if factor.func is sp.atan:
        return sp.pi / 2
    if factor.func is sp.exp and factor.args[0].is_nonpositive is True:
        return sp.S.One
    return None


def _elementary_local_majorant(expr):
    """Construct a sound elementary absolute majorant near zero.

    Returned inequalities use only standard real estimates: |sin u|<=|u|,
    |atan u|<=|u|, |1-cos u|<=u^2/2, and, locally, |asin u|<=2|u|,
    |tan u|<=2|u|, |exp(u)-1|<=2|u|.  The last three are used only when their
    arguments themselves are certified to tend to zero by the caller.
    """
    e = sp.sympify(expr)
    # Exact bounded factors first.
    b = _globally_bounded_factor(e)
    if b is not None:
        return b, ()
    if e.func is sp.Abs:
        m, obligations = _elementary_local_majorant(e.args[0])
        return m, obligations
    if e.func is sp.sin:
        return sp.Abs(e.args[0]), ()
    if e.func is sp.atan:
        # min(|u|,pi/2) is bounded by either choice; |u| is useful locally.
        return sp.Abs(e.args[0]), ()
    if e.func is sp.asin:
        return 2 * sp.Abs(e.args[0]), (e.args[0],)
    if e.func is sp.tan:
        return 2 * sp.Abs(e.args[0]), (e.args[0],)
    if e.func is sp.Add and len(e.args) == 2:
        # SymPy refinement may canonicalize 1-cos(u) as cos(u)-1.  Their
        # absolute values are identical, so recognize both orientations.
        for term in e.args:
            if term.func is sp.cos and sp.simplify(e - term) == -1:
                return sp.Abs(term.args[0]) ** 2 / 2, ()
            if (-term).func is sp.cos and sp.simplify(e - term) == 1:
                return sp.Abs((-term).args[0]) ** 2 / 2, ()
        if sp.S.One in e.args:
            other = sp.simplify(e - 1)
            if (-other).func is sp.cos:
                return sp.Abs((-other).args[0]) ** 2 / 2, ()
        a, b0 = e.args
        if a == 1 and b0.func is sp.Mul and -1 in b0.args:
            rest = -b0
            if rest.func is sp.cos:
                return sp.Abs(rest.args[0]) ** 2 / 2, ()
        if b0 == 1 and a.func is sp.Mul and -1 in a.args:
            rest = -a
            if rest.func is sp.cos:
                return sp.Abs(rest.args[0]) ** 2 / 2, ()
        # exp(u)-1 / 1-exp(u)
        for term in e.args:
            if term.func is sp.exp and sp.simplify(e - term) == -1:
                return 2 * sp.Abs(term.args[0]), (term.args[0],)
            if (-term).func is sp.exp and sp.simplify(e - term) == 1:
                return 2 * sp.Abs((-term).args[0]), ((-term).args[0],)
    if e.is_Mul:
        out = sp.S.One
        obligations = []
        for f in e.args:
            m, obs = _elementary_local_majorant(f)
            out *= m
            obligations.extend(obs)
        return out, tuple(obligations)
    if isinstance(e, sp.Pow):
        if e.exp.is_integer is True:
            m, obs = _elementary_local_majorant(e.base)
            if e.exp.is_nonnegative is True:
                return m**e.exp, obs
        return sp.Abs(e), ()
    if e.is_Add:
        pieces = []
        obligations = []
        for term in e.args:
            m, obs = _elementary_local_majorant(term)
            pieces.append(m)
            obligations.extend(obs)
        return sp.Add(*pieces), tuple(obligations)
    return sp.Abs(e), ()


def uniform_vanishing_amplitude_limit(expr, variables, target, *, domain=True):
    """Certify ``expr -> 0`` from a uniform elementary absolute majorant."""
    from .limit_primitives import _normalize_variables
    from .limits import LimitStatus, limit, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr = sp.sympify(expr)
    majorant, obligations = _elementary_local_majorant(expr)
    if sp.simplify(majorant - sp.Abs(expr)) == 0:
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "no sharper uniform majorant was derived",
        )
    q = _uniform_radial_order(majorant, variables, target)
    if not obligations and q is not None and q is not sp.oo and q > 0:
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            sp.S.Zero,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "uniform_vanishing_amplitude",
            f"an elementary absolute majorant has positive uniform radial order {q}",
        )
    for inner in obligations:
        g = limit(inner, variables, target, domain=domain, return_result=True)
        if g.status is not LimitStatus.PROVED or g.value != 0:
            return NewtonFanLimitResult(
                expr,
                variables,
                target,
                None,
                AdvancedLimitStatus.UNKNOWN,
                (),
                "none",
                "a local elementary-bound argument was not certified to tend to zero",
            )
    m = limit(majorant, variables, target, domain=domain, return_result=True)
    if m.status is LimitStatus.PROVED and m.value == 0:
        return NewtonFanLimitResult(
            expr,
            variables,
            target,
            sp.S.Zero,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "uniform_vanishing_amplitude",
            "a certified elementary absolute majorant tends uniformly to zero",
        )
    return NewtonFanLimitResult(
        expr,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "the derived absolute majorant was not certified to vanish",
    )


def coercive_outer_divergence_limit(expr, variables, target, *, domain=True):
    """Certify common log/reciprocal divergence from a vanishing inner germ."""
    from .limit_primitives import _normalize_variables
    from .limits import normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    expr = sp.sympify(expr)
    # log(|g|) and log(g) with g -> 0: on the real domain every admissible
    # sequence has |g| -> 0, hence log tends to -infinity.
    if expr.func is sp.log:
        inner = expr.args[0]
        base = inner.args[0] if inner.func is sp.Abs else inner
        # The shortcut is valid only for a continuous rational germ. Reduce
        # first so removable factors do not matter, then require the reduced
        # denominator to stay nonzero at the base point. In particular, a
        # homogeneous rational 0/0 germ must fall through to the general
        # multivariate solver instead of being certified from substitution.
        reduced = sp.cancel(base)
        _, denominator = sp.fraction(reduced)
        substitutions = dict(zip(variables, target, strict=True))
        denominator_value = sp.simplify(denominator.subs(substitutions))
        value = sp.simplify(reduced.subs(substitutions))
        continuous_rational_germ = denominator_value.is_zero is False
        # ``log(g) -> -oo`` is a real-germ theorem. For complex g -> 0 the
        # phase can remain direction-dependent, so magnitude decay alone is
        # not a scalar complex-limit certificate.
        real_germ = reduced.is_real is True or sp.im(reduced).equals(0) is True
        if (
            continuous_rational_germ
            and value == 0
            and real_germ
            and not reduced.has(sp.zoo, sp.nan)
        ):
            return NewtonFanLimitResult(
                expr,
                variables,
                target,
                -sp.oo,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "log_vanishing_inner",
                "the continuous logarithm argument tends to zero on its real punctured domain",
            )
    if expr.func is sp.exp:
        inner = sp.cancel(expr.args[0])
        num, den = sp.fraction(inner)
        nval = sp.simplify(num.subs(dict(zip(variables, target, strict=True))))
        dval = sp.simplify(den.subs(dict(zip(variables, target, strict=True))))
        if (
            nval.is_positive is True
            and dval == 0
            and _locally_nonnegative(den, variables, target)
        ):
            return NewtonFanLimitResult(
                expr,
                variables,
                target,
                sp.oo,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "exp_positive_reciprocal_divergence",
                "a positive numerator over a nonnegative vanishing denominator tends to +infinity, hence so does its exponential",
            )
    return NewtonFanLimitResult(
        expr,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no supported coercive outer divergence theorem applied",
    )


def _uniform_radial_order(expr, variables, target):
    """Certified radial vanishing order with a cheap decline path.

    This helper is used by early-dispatch certificates, so unsupported or
    structurally large germs must be cheaper to reject than the general
    geometry/QE machinery they precede.
    """

    raw = sp.sympify(expr)
    # Structural budget: do not simplify/factor a large expression merely to
    # discover that this conservative order calculus cannot handle it.
    max_nodes = 240
    max_depth = 28
    nodes = 0
    stack = [(raw, 1)]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            return None
        stack.extend((arg, depth + 1) for arg in node.args)

    u = sp.symbols(f"_rho0:{len(variables)}", real=True)
    e = raw.subs(
        dict(
            zip(variables, (a + z for a, z in zip(target, u, strict=True)), strict=True)
        )
    )
    uset = frozenset(u)
    r2 = sp.Add(*(z**2 for z in u))

    # Unknown applied functions are an immediate decline.  The supported
    # elementary atoms below have explicit bounded/vanishing semantics.
    supported_functions = {
        sp.Abs,
        sp.sin,
        sp.cos,
        sp.atan,
        sp.asin,
        sp.tan,
        sp.exp,
        sp.log,
        sp.sign,
    }
    for fn in e.atoms(sp.Function):
        if fn.func not in supported_functions:
            return None

    @functools.cache
    def order(f):
        if f == 0:
            return sp.oo
        if not (f.free_symbols & uset):
            return sp.S.Zero
        if f.func is sp.Abs:
            return order(f.args[0])
        if f.func is sp.sign:
            # Real sign is uniformly bounded by one; it need not converge.
            return sp.S.Zero
        if f.func is sp.log:
            arg = f.args[0]
            # log(1+g) has the same vanishing order as g when g -> 0.
            g = sp.expand(arg - 1)
            q = order(g)
            return q if q is not None and q > 0 else None
        if f.func in (sp.sin, sp.atan, sp.asin, sp.tan):
            q = order(f.args[0])
            return q if q is not None and q > 0 else None
        if f.func is sp.exp:
            # exp(g) is order-zero only when g stays bounded at the germ.
            # Treating every exponential as bounded is unsound: for example,
            # exp(-1/(x+y)) is unbounded on the x+y<0 side of the origin.
            q = order(f.args[0])
            return sp.S.Zero if q is not None and q >= 0 else None
        if f.func is sp.cos:
            # A standalone cosine is bounded but need not vanish.
            return sp.S.Zero
        if f.is_Add and len(f.args) == 2:
            # 1-cos(u), including SymPy's canonical -cos(u)+1 form.
            for term in f.args:
                neg = -term
                if neg.func is sp.cos and sp.simplify(f - term) == 1:
                    q = order(neg.args[0])
                    return None if q is None or q <= 0 else 2 * q
        if f.is_Add:
            qs = tuple(order(a) for a in f.args)
            return None if any(q is None for q in qs) else min(qs)
        if f.is_Mul:
            qs = tuple(order(a) for a in f.args)
            return None if any(q is None for q in qs) else sum(qs, sp.S.Zero)
        if isinstance(f, sp.Pow):
            base, ex = f.args
            if base == r2 and ex.is_Rational:
                return 2 * ex
            q = order(base)
            if q is None:
                return None
            if ex.is_nonnegative is True:
                return q * ex
            if q == 0:
                if base.has(sp.sign):
                    return None
                for power in base.atoms(sp.Pow):
                    if power.exp.is_integer is True:
                        continue
                    power_center = power.base.subs(dict.fromkeys(u, 0))
                    if not (
                        power_center.is_positive is True
                        or (
                            power_center.is_real is False
                            and power_center.is_finite is True
                            and power_center.is_zero is False
                        )
                    ):
                        return None
                try:
                    center = sp.simplify(base.subs(dict.fromkeys(u, 0)))
                except (TypeError, ValueError, ZeroDivisionError):
                    center = None
                if (
                    center is not None
                    and center.is_finite is True
                    and center.is_zero is False
                ):
                    return sp.S.Zero
            # Only pay for a Lojasiewicz check after the cheap structural
            # recursion has established a supported negative-power base.
            try:
                from ._multivariate_germ import _cheap_lojasiewicz_exponent_bound

                cert = _cheap_lojasiewicz_exponent_bound(base, u, (0,) * len(u))
                if cert.certified and ex.is_Rational is True:
                    return sp.Integer(cert.value) * ex
            except (TypeError, ValueError, sp.PolynomialError):
                pass
            return None
        # Polynomial fallback is exact and normally cheap after the budgets
        # above.  Avoid factor(), which was the principal pathological decline
        # cost on complicated unsupported germs.
        try:
            p = sp.Poly(f, *u)
            if p.is_zero:
                return sp.oo
            return sp.Integer(min(sum(m) for m, c in p.terms() if c != 0))
        except (sp.PolynomialError, TypeError, ValueError):
            return None

    return order(e)


def _analytic_jet_radial_order(expr, variables, target, *, max_order=5):
    """Return a certified lower radial order from a finite analytic Taylor jet.

    The remainder theorem for a real-analytic germ makes a first nonzero total
    degree m jet an O(r**m) bound locally.  This is bounded and
    declines on branch/nonanalytic germs.
    """
    raw = sp.sympify(expr)
    if len(variables) > 4 or sp.count_ops(raw) > 80:
        return None
    if raw.has(sp.Abs, sp.sign, sp.floor, sp.ceiling, sp.Heaviside, sp.Piecewise):
        return None
    u = sp.symbols(f"_jet0:{len(variables)}", real=True)
    shifted = raw.subs(
        dict(
            zip(variables, (a + z for a, z in zip(target, u, strict=True)), strict=True)
        )
    )
    zero = dict.fromkeys(u, 0)
    for power in shifted.atoms(sp.Pow):
        if power.exp.is_integer is True:
            continue
        try:
            center = sp.simplify(power.base.subs(zero))
        except (TypeError, ValueError, NotImplementedError):
            return None
        if not (
            center.is_positive is True
            or (
                center.is_real is False
                and center.is_finite is True
                and center.is_zero is False
            )
        ):
            return None
    entire_functions = {
        sp.sin,
        sp.cos,
        sp.exp,
        sp.sinh,
        sp.cosh,
        sp.erf,
        sp.erfc,
        sp.erfi,
        sp.airyai,
        sp.airybi,
        sp.airyaiprime,
        sp.airybiprime,
    }
    for atom in shifted.atoms(sp.Function):
        if atom.func not in entire_functions | {sp.log, sp.gamma}:
            return None
        if atom.func is sp.gamma and atom.args[0].subs(zero).is_positive is not True:
            return None
        if atom.func is sp.log:
            continue
        try:
            center = sp.simplify(atom.args[0].subs(zero)) if atom.args else sp.S.Zero
        except (TypeError, ValueError, NotImplementedError):
            return None
        if center.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or center.is_finite is False:
            return None
    for atom in shifted.atoms(sp.log):
        try:
            center = sp.simplify(atom.args[0].subs(zero))
        except (TypeError, ValueError, NotImplementedError):
            return None
        if center.is_positive is not True:
            return None
    try:
        center = sp.simplify(shifted.subs(zero))
    except (TypeError, ValueError, NotImplementedError):
        return None
    if center.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or center.is_finite is not True:
        return None
    if center != 0:
        return sp.S.Zero
    from itertools import product

    for degree in range(1, max_order + 1):
        for alpha in product(range(degree + 1), repeat=len(u)):
            if sum(alpha) != degree:
                continue
            derivative = shifted
            try:
                for z, count in zip(u, alpha, strict=True):
                    if count:
                        derivative = sp.diff(derivative, z, count)
                value = sp.simplify(derivative.subs(zero))
            except (TypeError, ValueError, NotImplementedError, RecursionError):
                return None
            if value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) or value.is_finite is not True:
                return None
            if value != 0 and value.is_zero is not True:
                return sp.Integer(degree)
    return None


def analytic_jet_radial_vanishing_limit(expr, variables, target, *, domain=True):
    """Certify zero after cancellation using analytic numerator/denominator jets."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    if sp.sympify(domain) is not sp.S.true:
        return NewtonFanLimitResult(
            sp.sympify(expr),
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "analytic jet radial theorem requires an ambient germ",
        )
    num, den = sp.fraction(sp.cancel(sp.together(sp.sympify(expr))))
    num_order = _analytic_jet_radial_order(num, variables, target)
    if num_order is None or num_order is sp.oo:
        return NewtonFanLimitResult(
            sp.sympify(expr),
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "no certified analytic numerator jet",
        )
    # A Taylor jet bounds the numerator from above. Division requires a
    # lower bound for the denominator, supplied by its reciprocal bound.
    # The denominator jet alone can vanish on approaching varieties.
    inverse_order = _uniform_radial_order(1 / den, variables, target)
    den_order = None if inverse_order is None else -inverse_order
    if den_order is not None and den_order is not sp.oo and num_order > den_order:
        return NewtonFanLimitResult(
            sp.sympify(expr),
            variables,
            target,
            sp.S.Zero,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "analytic_jet_radial_order",
            f"analytic numerator vanishes to order {num_order}, exceeding denominator radial order {den_order}",
        )
    return NewtonFanLimitResult(
        sp.sympify(expr),
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "analytic jet order does not dominate denominator order",
    )


def radial_order_vanishing_limit(expr, variables, target, *, domain=True):
    """Certify zero when a conservative uniform radial order is positive."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    q = _uniform_radial_order(expr, variables, target)
    if q is not None and q is not sp.oo and q > 0:
        return NewtonFanLimitResult(
            sp.sympify(expr),
            variables,
            target,
            sp.S.Zero,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "uniform_radial_order",
            f"a uniform radial bound has strictly positive order {q}",
        )
    if q is sp.oo:
        return NewtonFanLimitResult(
            sp.sympify(expr),
            variables,
            target,
            sp.S.Zero,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "uniform_radial_order",
            "the shifted germ vanishes identically",
        )
    return NewtonFanLimitResult(
        sp.sympify(expr),
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no positive uniform radial order was certified",
    )


def _locally_nonnegative(expr, variables, target):
    """Cheap structural proof that an expression is nonnegative locally.

    This does not infer a neighborhood sign merely from a
    positive value at the target: that inference needs a continuity proof.
    """
    e = sp.sympify(expr)
    if e.is_nonnegative is True:
        return True
    if e.func is sp.Abs:
        return True
    if isinstance(e, sp.Pow):
        if e.exp.is_even is True:
            return True
        if e.exp.is_Rational and e.exp.q == 2 and e.exp.p > 0:
            return True
    if (e.is_Mul or e.is_Add) and all(
        _locally_nonnegative(a, variables, target) for a in e.args
    ):
        return True
    # A polynomial is continuous everywhere, so strict positivity at the
    # target certifies positivity on some neighborhood.  Do not make this
    # inference for arbitrary discontinuous/transcendental expressions.
    try:
        if e.is_polynomial(*variables):
            value = e.subs(dict(zip(variables, target, strict=True)))
            if value.is_positive is True:
                return True
    except (TypeError, ValueError, sp.PolynomialError):
        pass
    return False


def positive_quotient_divergence_limit(expr, variables, target, *, domain=True):
    """Certify +infinity for a positive numerator over a positive vanishing germ."""
    from .limit_primitives import _normalize_variables
    from .limits import LimitStatus, limit, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    e = sp.sympify(expr)
    num, den = sp.fraction(sp.cancel(e))
    if den == 1:
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "not a quotient",
        )
    nr = limit(num, variables, target, domain=domain, return_result=True)
    dr = limit(den, variables, target, domain=domain, return_result=True)
    if (
        nr.status is LimitStatus.PROVED
        and nr.value is not None
        and nr.value.is_positive is True
        and dr.status is LimitStatus.PROVED
        and dr.value == 0
        and _locally_nonnegative(den, variables, target)
    ):
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            sp.oo,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "positive_vanishing_denominator",
            "a locally positive numerator is divided by a nonnegative denominator tending to zero",
        )
    return NewtonFanLimitResult(
        e,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "positive quotient divergence was not certified",
    )


def norm_comparison_divergence_limit(expr, variables, target, *, domain=True):
    """Certify two elementary norm-comparison poles used by local geometry."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    e = sp.sympify(expr)
    # |g|/g^2 = 1/|g| on its domain, for a continuous g vanishing at target.
    num, den = sp.fraction(e)
    if num.func is sp.Abs:
        g = num.args[0]
        if (
            sp.simplify(den - g**2) == 0
            and sp.simplify(g.subs(dict(zip(variables, target, strict=True)))) == 0
        ):
            return NewtonFanLimitResult(
                e,
                variables,
                target,
                sp.oo,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "absolute_reciprocal_pole",
                "|g|/g^2 equals 1/|g| off the zero set and g tends to zero",
            )
    # max_i |u_i| / sqrt(sum_i u_i^4) >= 1/(sqrt(n) max_i|u_i|).
    shifted = [sp.expand(v - a) for v, a in zip(variables, target, strict=True)]
    M = sp.Max(*(sp.Abs(u) for u in shifted))
    if sp.simplify(num - M) == 0:
        expected = sp.sqrt(sp.Add(*(u**4 for u in shifted)))
        if sp.simplify(den - expected) == 0:
            return NewtonFanLimitResult(
                e,
                variables,
                target,
                sp.oo,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "max_norm_quartic_pole",
                "the quartic norm is at most sqrt(n) times the square of the max norm, so the quotient diverges uniformly",
            )
    return NewtonFanLimitResult(
        e,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no supported norm-comparison pole applied",
    )


def denominator_term_vanishing_limit(expr, variables, target, *, domain=True):
    """Use one nonnegative denominator summand as a global quotient majorant."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    e = sp.sympify(expr)
    num, den = sp.fraction(e)
    if not den.is_Add:
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "denominator is not a nonnegative sum",
        )
    # The comparison den >= term is valid only when *every* denominator
    # summand is locally nonnegative.  Checking merely the selected term is
    # unsound in the presence of cancellation (for example A - B).
    if not all(_locally_nonnegative(term, variables, target) for term in den.args):
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "denominator contains a summand that is not locally nonnegative",
        )
    for term in den.args:
        # No simplify/cancel is needed for soundness; radial order handles the
        # quotient directly and has its own strict decline budget.
        bound = sp.Abs(num) / term
        q = _uniform_radial_order(bound, variables, target)
        if q is not None and q is not sp.oo and q > 0:
            return NewtonFanLimitResult(
                e,
                variables,
                target,
                sp.S.Zero,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "denominator_term_majorant",
                f"one nonnegative denominator summand gives a quotient majorant of positive radial order {q}",
            )
    return NewtonFanLimitResult(
        e,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no denominator summand yielded a vanishing majorant",
    )


def power_log_vanishing_limit(expr, variables, target, *, domain=True):
    """Certify the standard x^a log(x) -> 0+ composition inside products."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    e = sp.sympify(expr)
    for atom in e.atoms(sp.log):
        arg = atom.args[0]
        for v, a in zip(variables, target, strict=True):
            if arg == v and a == 0:
                quotient = sp.cancel(e / (v * atom))
                val = sp.simplify(
                    quotient.subs(dict(zip(variables, target, strict=True)))
                )
                if val.is_finite is True:
                    return NewtonFanLimitResult(
                        e,
                        variables,
                        target,
                        sp.S.Zero,
                        AdvancedLimitStatus.CERTIFIED,
                        (),
                        "power_log_vanishing",
                        "the standard one-sided real germ x*log(x) tends to zero and the remaining factor has a finite target value",
                    )
    return NewtonFanLimitResult(
        e,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no supported power-log vanishing factor",
    )


def _positive_lower_monomials(expr, variables, target, *, max_bounds=16):
    """Return a bounded set of cheap exact nonnegative denominator lower bounds."""
    e = sp.sympify(expr)
    if e.func is sp.Abs:
        return (e,)
    if (
        e.is_nonnegative is True
        and not e.is_Add
        and not e.is_Mul
        and not isinstance(e, sp.Pow)
    ):
        return (e,)
    if e.is_Add:
        if not all(_locally_nonnegative(a, variables, target) for a in e.args):
            return ()
        out = []
        seen = set()
        for a in e.args:
            candidates = _positive_lower_monomials(
                a, variables, target, max_bounds=max_bounds
            ) or (a,)
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    out.append(candidate)
                    if len(out) >= max_bounds:
                        return tuple(out)
        return tuple(out)
    if e.is_Mul:
        if not all(_locally_nonnegative(a, variables, target) for a in e.args):
            return ()
        choices = [
            _positive_lower_monomials(a, variables, target, max_bounds=max_bounds)
            for a in e.args
        ]
        if any(not c for c in choices):
            return ()
        out = [sp.S.One]
        for candidates in choices:
            if len(out) * len(candidates) > max_bounds:
                return ()
            out = [sp.Mul(a, b, evaluate=False) for a in out for b in candidates]
        return tuple(out)
    if isinstance(e, sp.Pow) and e.exp.is_Rational and e.exp.is_positive is True:
        base = e.base
        if base.is_Add and all(
            _locally_nonnegative(a, variables, target) for a in base.args
        ):
            return tuple(a**e.exp for a in base.args[:max_bounds])
        if _locally_nonnegative(base, variables, target):
            return (e,)
    if _locally_nonnegative(e, variables, target):
        return (e,)
    return ()


def weighted_fractional_angular_vanishing_limit(
    expr, variables, target, *, domain=True
):
    """Certify zero by an exact weighted/Abs denominator lower bound.

    For ``|N|/D`` with ``D>=m>=0`` on the punctured real domain, ``|N/D|`` is
    bounded by ``|N|/m`` wherever the original quotient is defined.  A positive
    uniform radial order of that majorant proves the limit.  This is the
    weighted-angular theorem needed for fractional norms such as
    ``sqrt(x**4+y**2)`` and products of Euclidean/L1 factors.
    """
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    e = sp.sympify(expr)
    if domain is not sp.S.true:
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "weighted denominator squeezing requires the unrestricted real approach domain",
        )
    # Avoid cancel/simplify on the decline path.  This theorem only needs the
    # displayed quotient; algebraic cancellation is unnecessary for soundness.
    num, den = sp.fraction(e)
    if den == 1 or not _locally_nonnegative(den, variables, target):
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            (),
            "none",
            "no certified nonnegative weighted denominator was found",
        )
    for lower in _positive_lower_monomials(den, variables, target):
        if lower == 0:
            continue
        majorant = sp.Abs(num) / lower
        # Tiny candidate majorants benefit from canonical Abs cancellation
        # (e.g. Abs(x*y)/Abs(y) -> Abs(x)); never simplify a large decline.
        if sp.count_ops(majorant) <= 32:
            try:
                majorant = sp.simplify(majorant)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                # Simplification is optional canonicalization, not part of the
                # proof.  SymPy logcombine can fail on fractional Abs forms.
                pass
        q = _uniform_radial_order(majorant, variables, target)
        if q is not None and q is not sp.oo and q > 0:
            return NewtonFanLimitResult(
                e,
                variables,
                target,
                sp.S.Zero,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "weighted_fractional_angular_bound",
                f"the nonnegative denominator is bounded below by {sp.sstr(lower)}; hence |f| <= {sp.sstr(majorant)}, whose uniform radial order is {q}",
            )
    return NewtonFanLimitResult(
        e,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no denominator lower bound produced a positive-order uniform majorant",
    )


def cheap_uniform_vanishing_limit(expr, variables, target, *, domain=True):
    """Nonrecursive uniform-zero certificate for early dispatch."""
    from .limit_primitives import _normalize_variables, normalize_limit_target

    variables = _normalize_variables(variables)
    target = normalize_limit_target(variables, target)
    e = sp.sympify(expr)
    weighted = weighted_fractional_angular_vanishing_limit(
        e, variables, target, domain=domain
    )
    if weighted.status is AdvancedLimitStatus.CERTIFIED:
        return weighted
    q = _uniform_radial_order(e, variables, target)
    if q is not None and q is not sp.oo and q > 0:
        return NewtonFanLimitResult(
            e,
            variables,
            target,
            sp.S.Zero,
            AdvancedLimitStatus.CERTIFIED,
            (),
            "uniform_radial_order",
            f"a uniform radial bound has strictly positive order {q}",
        )
    majorant, obligations = _elementary_local_majorant(e)
    if not obligations and majorant != sp.Abs(e):
        q = _uniform_radial_order(majorant, variables, target)
        if q is not None and q is not sp.oo and q > 0:
            return NewtonFanLimitResult(
                e,
                variables,
                target,
                sp.S.Zero,
                AdvancedLimitStatus.CERTIFIED,
                (),
                "uniform_vanishing_amplitude",
                f"an elementary absolute majorant has positive uniform radial order {q}",
            )
    return NewtonFanLimitResult(
        e,
        variables,
        target,
        None,
        AdvancedLimitStatus.UNKNOWN,
        (),
        "none",
        "no nonrecursive uniform vanishing bound applied",
    )
