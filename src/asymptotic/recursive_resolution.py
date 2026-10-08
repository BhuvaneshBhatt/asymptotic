"""Recursive blow-up resolution of unresolved exceptional-divisor strata."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from ._symbolic_policy import bounded_limit, bounded_solve_system
from .blowup_geometry import BlowUpChart, weighted_spherical_atlas
from .coverage import CoverageCertificate, CoverageObligation
from .semialgebraic_angular import semialgebraic_angular_image


@dataclass(frozen=True)
class ExceptionalStratum:
    condition: sp.Expr
    reason: str
    defining_equations: tuple[sp.Expr, ...]
    dimension_hint: int | None = None


@dataclass(frozen=True)
class DimensionDescentCertificate:
    parent_dimension: int
    child_dimension: int
    provider: str = "semialg_algebraic_dimension"

    @property
    def certified(self):
        return self.child_dimension < self.parent_dimension


@dataclass(frozen=True)
class StratumLocalModel:
    point: tuple[sp.Expr, ...]
    tangent_basis: tuple[tuple[sp.Expr, ...], ...]
    normal_basis: tuple[tuple[sp.Expr, ...], ...]
    rank: int
    certified_smooth: bool
    provider: str


def _exact_stratum_point(stratum, variables):
    equations = tuple(sp.Eq(e, 0) for e in stratum.defining_equations)
    if not equations:
        return None
    try:
        from semialg import solve_semialgebraic

        result = solve_semialgebraic(sp.And(*equations), variables, samples=1)
        samples = getattr(result, "samples", ()) or ()
        if samples:
            sample = samples[0]
            if isinstance(sample, dict):
                return tuple(sp.simplify(sample[v]) for v in variables)
            if isinstance(sample, (tuple, list)) and len(sample) >= len(variables):
                return tuple(map(sp.simplify, sample[: len(variables)]))
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        pass
    try:
        sols = bounded_solve_system(
            stratum.defining_equations, variables, allow_general=True
        )
        for sol in sols:
            if all(
                v in sol and not sol[v].free_symbols and sol[v].is_real is not False
                for v in variables
            ):
                return tuple(sp.simplify(sol[v]) for v in variables)
    except (NotImplementedError, TypeError, ValueError):
        pass
    return None


def _stratum_local_model(stratum, variables):
    point = _exact_stratum_point(stratum, variables)
    if point is None:
        return None
    subs = dict(zip(variables, point, strict=True))
    jac = sp.Matrix(
        [
            [sp.diff(g, v).subs(subs) for v in variables]
            for g in stratum.defining_equations
        ]
    )
    rank = jac.rank()
    if rank <= 0:
        return None
    tangent = tuple(tuple(sp.simplify(x) for x in vec) for vec in jac.nullspace())
    normal = tuple(tuple(sp.simplify(x) for x in vec) for vec in jac.T.columnspace())
    # Jacobian rank at the exact point certifies a smooth local complete
    # intersection when the defining equations have this independent rank.
    smooth = rank == len(stratum.defining_equations)
    return StratumLocalModel(
        point, tangent, normal, rank, smooth, "exact_jacobian_local_model"
    )


def _centered_normal_chart(parent, stratum, index):
    vars_ = tuple(v for v in parent.angular_variables if isinstance(v, sp.Symbol))
    model = _stratum_local_model(stratum, vars_)
    if model is None or not model.certified_smooth or not model.normal_basis:
        return None
    # Coordinates split into tangent directions along the stratum and normal
    # directions that are blown up. This is a local tubular chart centered at
    # an exact smooth stratum point.
    tangent_symbols = sp.symbols(f"_tau{index}_0:{len(model.tangent_basis)}", real=True)
    normal_symbols = sp.symbols(f"_nu{index}_0:{len(model.normal_basis)}", real=True)
    rho = sp.Symbol(f"_rho{index}", positive=True)
    substitution = {}
    for j, v in enumerate(vars_):
        along = sum(
            tangent_symbols[k] * model.tangent_basis[k][j]
            for k in range(len(tangent_symbols))
        )
        transverse = sum(
            normal_symbols[k] * model.normal_basis[k][j]
            for k in range(len(normal_symbols))
        )
        substitution[v] = sp.simplify(model.point[j] + along + rho * transverse)
    return model, substitution, rho, tangent_symbols, normal_symbols


@dataclass(frozen=True)
class SmoothStratumChart:
    condition: sp.Expr
    minor: sp.Expr
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]
    base_symbols: tuple[sp.Symbol, ...]
    normal_symbols: tuple[sp.Symbol, ...]
    radial_symbol: sp.Symbol
    substitution: tuple[tuple[sp.Symbol, sp.Expr], ...]
    base_equations: tuple[sp.Expr, ...] = ()
    provider: str = "jacobian_minor_tubular_chart"

    @property
    def radial_variable(self):
        return self.radial_symbol

    @property
    def angular_variables(self):
        return self.base_symbols + self.normal_symbols

    @property
    def angular_constraint(self):
        return sp.And(self.condition, sp.Eq(sum(n**2 for n in self.normal_symbols), 1))

    @property
    def sector(self):
        return sp.S.true

    @property
    def dominant_coordinate(self):
        return None

    def transform(self, expression):
        return sp.cancel(
            sp.together(
                sp.sympify(expression).subs(dict(self.substitution), simultaneous=True)
            )
        )

    def pullback_domain(self, domain):
        return sp.simplify(
            sp.sympify(domain).subs(dict(self.substitution), simultaneous=True)
        )

    @property
    def exceptional_domain(self):
        return sp.And(self.angular_constraint, self.sector)


@dataclass(frozen=True)
class StratumAtlas:
    rank: int
    smooth_charts: tuple[SmoothStratumChart, ...]
    smooth_condition: sp.Expr
    singular_stratum: ExceptionalStratum | None
    coverage: CoverageCertificate
    singular_components: tuple[ExceptionalStratum, ...] = ()
    dimension_descent: tuple[DimensionDescentCertificate, ...] = ()
    provider: str = "jacobian_minor_stratum_atlas"


def _nonzero_minors(jac, rank):
    if rank <= 0:
        return ()
    out = []
    for rows in combinations(range(jac.rows), rank):
        for cols in combinations(range(jac.cols), rank):
            m = sp.factor(jac.extract(rows, cols).det())
            if m != 0:
                out.append((tuple(rows), tuple(cols), m))
    return tuple(out)


def _tubular_chart_for_minor(stratum, variables, jac, rows, cols, minor, index):
    base = sp.symbols(f"_base{index}_0:{len(variables)}", real=True)
    normal = sp.symbols(f"_normal{index}_0:{len(rows)}", real=True)
    rho = sp.Symbol(f"_normal_r{index}", positive=True)
    bsubs = dict(zip(variables, base, strict=True))
    # Independent constraint gradients give a normal frame throughout the
    # minor-open patch. The nonzero minor certifies their independence.
    gradients = []
    for row in rows:
        gradients.append(
            tuple(sp.simplify(jac[row, j].subs(bsubs)) for j in range(jac.cols))
        )
    substitution = []
    for j, v in enumerate(variables):
        displacement = sum(normal[a] * gradients[a][j] for a in range(len(rows)))
        substitution.append((v, sp.simplify(base[j] + rho * displacement)))
    base_eq = tuple(sp.Eq(g.subs(bsubs), 0) for g in stratum.defining_equations)
    # Preserve every domain-relative inequality/branch-side predicate on the
    # tubular base, not only the algebraic equations defining its closure.
    base_domain = sp.sympify(stratum.condition).subs(bsubs, simultaneous=True)
    patch = sp.And(base_domain, *base_eq, sp.Ne(minor.subs(bsubs), 0))
    return SmoothStratumChart(
        patch,
        minor,
        rows,
        cols,
        tuple(base),
        tuple(normal),
        rho,
        tuple(substitution),
        tuple(sp.expand(g.subs(bsubs)) for g in stratum.defining_equations),
    )


def _certified_locus_dimension(equations, variables):
    try:
        from semialg.optimization_geometry import polynomial_locus_dimension

        d = polynomial_locus_dimension(tuple(equations), tuple(variables))
        return int(d) if d is not None else None
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        return None


def _decompose_singular_stratum(stratum, variables, parent_dimension):
    """Certified irreducible/equidimensional children of a singular remainder."""
    try:
        from semialg import is_satisfiable
        from semialg.algebraic_geometry import irreducible_components

        components = irreducible_components(stratum.defining_equations, variables)
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        return (), ()
    out = []
    certs = []
    for component in components:
        d = int(component.dimension)
        cond = sp.And(stratum.condition, *(sp.Eq(g, 0) for g in component.equations))
        try:
            if not is_satisfiable(cond, variables):
                continue
        except (
            ArithmeticError,
            NotImplementedError,
            RuntimeError,
            TypeError,
            ValueError,
            sp.PolynomialError,
        ):
            continue
        cert = DimensionDescentCertificate(parent_dimension, d)
        if not cert.certified:
            continue
        out.append(
            ExceptionalStratum(
                cond,
                "certified singular irreducible component",
                tuple(component.equations),
                d,
            )
        )
        certs.append(cert)
    return tuple(out), tuple(certs)


def finite_stratum_atlas(stratum, variables):
    """Finite Jacobian-minor atlas of the smooth locus plus its singular locus.

    The cover is algebraic rather than sample based: every rank-r smooth point
    lies in at least one patch on which an r-minor is nonzero.  The complement
    is exactly the rank-drop locus cut out by all r-minors.
    """
    variables = tuple(variables)
    equations = tuple(sp.expand(g) for g in stratum.defining_equations)
    if not equations:
        return None
    jac = sp.Matrix([[sp.diff(g, v) for v in variables] for g in equations])
    try:
        rank = int(jac.rank())
    except (TypeError, ValueError, NotImplementedError):
        return None
    minors = _nonzero_minors(jac, rank)
    if rank <= 0 or not minors:
        singular = ExceptionalStratum(
            stratum.condition,
            "Jacobian rank is zero",
            equations,
            stratum.dimension_hint,
        )
        return StratumAtlas(
            rank,
            (),
            sp.S.false,
            singular,
            CoverageCertificate.complete(
                "jacobian_minor_stratum_atlas",
                "rank-zero locus is entirely singular",
                ("singular",),
            ),
        )
    charts = tuple(
        _tubular_chart_for_minor(stratum, variables, jac, rows, cols, m, i)
        for i, (rows, cols, m) in enumerate(minors)
    )
    smooth = sp.And(stratum.condition, sp.Or(*(sp.Ne(m, 0) for _, _, m in minors)))
    singular_condition = sp.And(stratum.condition, *(sp.Eq(m, 0) for _, _, m in minors))
    singular = None
    try:
        from semialg import is_satisfiable

        has_singular = bool(is_satisfiable(singular_condition, variables))
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        has_singular = True
    singular_components = ()
    descent = ()
    if has_singular:
        parent_dimension = stratum.dimension_hint
        if parent_dimension is None:
            parent_dimension = _certified_locus_dimension(equations, variables)
        singular = ExceptionalStratum(
            singular_condition,
            "Jacobian rank drops",
            equations + tuple(m for _, _, m in minors),
            None,
        )
        if parent_dimension is not None:
            singular_components, descent = _decompose_singular_stratum(
                singular, variables, parent_dimension
            )
            if singular_components:
                singular = ExceptionalStratum(
                    singular_condition,
                    "Jacobian rank drops",
                    singular.defining_equations,
                    max(c.dimension_hint for c in singular_components),
                )
    obligations = tuple(
        CoverageObligation(
            "smooth_minor_patch", c.condition, True, "nonzero Jacobian minor"
        )
        for c in charts
    )
    coverage = CoverageCertificate.complete(
        "jacobian_minor_stratum_atlas",
        "nonzero maximal minors cover the smooth rank locus; their common zero set is the separate singular locus",
        tuple(c.condition for c in charts)
        + ((singular.condition,) if singular else ()),
        obligations,
    )
    return StratumAtlas(
        rank, charts, smooth, singular, coverage, singular_components, descent
    )


def singular_locus_chain(stratum, variables, max_depth=3):
    """Recursively stratify rank-drop loci only with certified dimension descent."""
    chain = []
    frontier = (stratum,)
    for _ in range(max_depth):
        next_frontier = []
        for current in frontier:
            atlas = finite_stratum_atlas(current, variables)
            if atlas is None:
                continue
            chain.append(atlas)
            if (
                atlas.singular_components
                and atlas.dimension_descent
                and all(c.certified for c in atlas.dimension_descent)
            ):
                next_frontier.extend(atlas.singular_components)
        if not next_frontier:
            break
        frontier = tuple(next_frontier)
    return tuple(chain)


@dataclass(frozen=True)
class ResolutionNode:
    expression: sp.Expr
    chart: BlowUpChart
    angular_expression: sp.Expr | None
    regular_image: sp.Set | None
    unresolved_strata: tuple[ExceptionalStratum, ...]
    children: tuple[ResolutionNode, ...]
    coverage: CoverageCertificate
    depth: int
    provider: str

    @property
    def certified(self):
        return self.coverage.certified and all(
            child.certified for child in self.children
        )


def _reduce_base_coefficients(expression, chart):
    equations = getattr(chart, "base_equations", ())
    base = getattr(chart, "base_symbols", ())
    if not equations or not base:
        return sp.expand(expression)
    try:
        gb = sp.groebner(equations, *base, order="grevlex", domain=sp.EX)
    except (sp.PolynomialError, ValueError, TypeError, NotImplementedError):
        try:
            gb = sp.groebner(equations, *base, domain=sp.EX)
        except (sp.PolynomialError, ValueError, TypeError, NotImplementedError):
            return sp.expand(expression)
    r = chart.radial_variable
    try:
        poly = sp.Poly(sp.expand(expression), r)
        reduced = sp.S.Zero
        for (power,), coeff in poly.terms():
            try:
                coeff = gb.reduce(sp.expand(coeff))[1]
            except (sp.PolynomialError, ValueError, TypeError, NotImplementedError):
                pass
            reduced += sp.expand(coeff) * r**power
        return sp.expand(reduced)
    except sp.PolynomialError:
        return sp.expand(expression)


def _radial_initial(expr, chart):
    transformed = sp.cancel(chart.transform(expr))
    r = chart.radial_variable
    num, den = map(sp.expand, sp.fraction(transformed))
    num = _reduce_base_coefficients(num, chart)
    den = _reduce_base_coefficients(den, chart)
    try:
        pn = sp.Poly(num, r)
        pd = sp.Poly(den, r)
        kn = min(m[0] for m, c in pn.terms() if c)
        kd = min(m[0] for m, c in pd.terms() if c)
        n0 = sp.expand(
            bounded_limit(num / r**kn, r, 0, direction="+", allow_general=True)
        )
        d0 = sp.expand(
            bounded_limit(den / r**kd, r, 0, direction="+", allow_general=True)
        )
        return kn - kd, n0, d0
    except (sp.PolynomialError, ValueError, TypeError, NotImplementedError):
        return None


def _singular_strata(numerator, denominator, angular, constraint, variables):
    strata = []
    # denominator-zero divisor: angular map is unresolved/polar there.
    if denominator != 1 and denominator != 0:
        strata.append(
            ExceptionalStratum(
                sp.And(constraint, sp.Eq(denominator, 0)),
                "leading denominator vanishes",
                (denominator,),
            )
        )
    # common zero is a genuinely indeterminate exceptional stratum.
    if numerator != 0 and denominator != 0:
        strata.append(
            ExceptionalStratum(
                sp.And(constraint, sp.Eq(numerator, 0), sp.Eq(denominator, 0)),
                "leading numerator and denominator vanish",
                (numerator, denominator),
            )
        )
    # critical singular locus of angular rational map.
    derivs = tuple(
        sp.together(sp.diff(angular, v)) for v in variables if isinstance(v, sp.Symbol)
    )
    if derivs:
        critical = sp.And(constraint, *(sp.Eq(sp.fraction(d)[0], 0) for d in derivs))
        strata.append(
            ExceptionalStratum(
                critical,
                "angular differential vanishes",
                tuple(sp.fraction(d)[0] for d in derivs),
            )
        )
    # discard provably empty strata.
    out = []
    for s in strata:
        try:
            from semialg import is_satisfiable

            if not is_satisfiable(s.condition, variables):
                continue
        except (
            ImportError,
            AttributeError,
            ArithmeticError,
            NotImplementedError,
            TypeError,
            ValueError,
            sp.PolynomialError,
        ):
            pass
        out.append(s)
    return tuple(out)


def _child_chart(parent, stratum, index):
    """Choose a stratum-centered normal blow-up when smooth geometry is certified."""
    centered = _centered_normal_chart(parent, stratum, index)
    if centered is not None:
        return ("centered", centered)
    # No arbitrary origin-centered fallback: without a certified local center
    # and tangent/normal split the stratum remains an explicit obligation.
    return None


def resolve_exceptional_strata(
    expr, variables, target=None, *, weights=None, domain=sp.S.true, max_depth=3
):
    """Recursively resolve singular/vanishing loci on exceptional divisors.

    Certification is strict: every accumulating unresolved
    stratum must either have a certified child cover or remain an explicit
    missing coverage obligation.
    """
    variables = tuple(variables)
    target = tuple(sp.S.Zero for _ in variables) if target is None else tuple(target)
    weights = (1,) * len(variables) if weights is None else tuple(weights)
    atlas = weighted_spherical_atlas(variables, target, weights)
    nodes = []
    for chart in atlas.charts:
        nodes.append(
            _resolve_chart(sp.sympify(expr), chart, sp.sympify(domain), 0, max_depth)
        )
    coverage = atlas.coverage
    obligations = tuple(
        CoverageObligation("exceptional_chart", i, n.certified, n.provider)
        for i, n in enumerate(nodes)
    )
    if all(n.certified for n in nodes):
        coverage = CoverageCertificate.complete(
            "recursive_exceptional_resolution",
            "all initial blow-up charts and every accumulating child stratum were discharged",
            tuple(range(len(nodes))),
            obligations,
        )
    else:
        coverage = CoverageCertificate.partial(
            "recursive_exceptional_resolution",
            "one or more exceptional strata remain unresolved",
            tuple(i for i, n in enumerate(nodes) if n.certified),
            tuple(i for i, n in enumerate(nodes) if not n.certified),
            obligations,
        )
    return tuple(nodes), coverage


def _resolve_stratum_atlas_images(expr, atlas, variables, depth, max_depth):
    """Execute all smooth tubular charts and recursively resolve singular pieces."""
    children = []
    obligations = []
    smooth_children = []
    for tubular in atlas.smooth_charts:
        child = _resolve_chart(expr, tubular, sp.S.true, depth, max_depth)
        children.append(child)
        smooth_children.append(child)
    smooth_empty = (
        atlas.smooth_condition is sp.S.false or atlas.smooth_condition == sp.S.false
    )
    smooth_ok = smooth_empty or (
        bool(atlas.smooth_charts) and all(c.certified for c in smooth_children)
    )
    obligations.append(
        CoverageObligation(
            "smooth_exceptional_stratum",
            atlas.smooth_condition,
            smooth_ok,
            f"{len(atlas.smooth_charts)} executable Jacobian-minor tubular charts",
        )
    )
    if atlas.singular_stratum is not None:
        descent_ok = (
            bool(atlas.singular_components)
            and bool(atlas.dimension_descent)
            and all(c.certified for c in atlas.dimension_descent)
        )
        singular_ok = descent_ok
        if descent_ok:
            for component in atlas.singular_components:
                subatlas = finite_stratum_atlas(component, variables)
                if subatlas is None:
                    singular_ok = False
                    continue
                from .resolution_progress import progress_certificate

                progress = progress_certificate(
                    atlas.dimension, subatlas.dimension, expr, variables
                )
                if not progress.certified:
                    singular_ok = False
                    obligations.append(
                        CoverageObligation(
                            "resolution_progress",
                            component.condition,
                            False,
                            progress.reason,
                        )
                    )
                    continue
                subchildren, subobligations = (
                    _resolve_stratum_atlas_images(
                        expr, subatlas, variables, depth + 1, max_depth
                    )
                    if depth < max_depth
                    else ((), ())
                )
                children.extend(subchildren)
                obligations.extend(subobligations)
                if (
                    depth >= max_depth
                    or not subobligations
                    or not all(o.discharged for o in subobligations)
                ):
                    singular_ok = False
        obligations.append(
            CoverageObligation(
                "singular_exceptional_locus",
                atlas.singular_stratum.condition,
                singular_ok,
                "certified irreducible decomposition, strict dimension descent, and recursively discharged child atlases",
            )
        )
    return tuple(children), tuple(obligations)


def _resolve_chart(expr, chart, domain, depth, max_depth):
    initial = _radial_initial(expr, chart)
    if initial is None:
        return ResolutionNode(
            expr,
            chart,
            None,
            None,
            (),
            (),
            CoverageCertificate.unknown(
                "recursive_resolution", "radial initial form unavailable"
            ),
            depth,
            "radial_initial_unknown",
        )
    order, num, den = initial
    if order > 0:
        image = sp.FiniteSet(0)
        strata = ()
    elif order < 0:
        image = None
        angular = sp.cancel(num / den)
        strata = _singular_strata(
            num, den, angular, chart.angular_constraint, chart.angular_variables
        )
        return ResolutionNode(
            expr,
            chart,
            angular,
            image,
            strata,
            (),
            CoverageCertificate.partial(
                "recursive_resolution",
                "negative radial order requires extended/polar child analysis",
                (),
                tuple(s.condition for s in strata) or ("polar",),
            ),
            depth,
            "negative_radial_order",
        )
    else:
        angular = sp.cancel(num / den)
        regular_domain = sp.And(chart.angular_constraint, chart.sector, domain)
        result = semialgebraic_angular_image(
            angular,
            tuple(v for v in chart.angular_variables if isinstance(v, sp.Symbol)),
            regular_domain,
        )
        image = result.image
        strata = _singular_strata(
            num,
            den,
            angular,
            regular_domain,
            tuple(v for v in chart.angular_variables if isinstance(v, sp.Symbol)),
        )
    if order > 0:
        return ResolutionNode(
            expr,
            chart,
            None,
            image,
            (),
            (),
            CoverageCertificate.complete(
                "recursive_resolution",
                "positive radial order vanishes uniformly on the chart",
                (chart.dominant_coordinate,),
            ),
            depth,
            "positive_radial_order",
        )
    if not strata and result.certified:
        return ResolutionNode(
            expr,
            chart,
            angular,
            image,
            (),
            (),
            result.coverage,
            depth,
            "regular_exceptional_image",
        )
    if depth >= max_depth:
        return ResolutionNode(
            expr,
            chart,
            angular,
            image,
            strata,
            (),
            CoverageCertificate.partial(
                "recursive_resolution",
                "resource depth guard reached after otherwise progress-certified recursion",
                (),
                tuple(s.condition for s in strata),
            ),
            depth,
            "resolution_depth_limit",
        )
    children = []
    child_obligations = []
    angular_vars = tuple(v for v in chart.angular_variables if isinstance(v, sp.Symbol))
    for stratum in strata:
        atlas = finite_stratum_atlas(stratum, angular_vars)
        if atlas is None:
            child_obligations.append(
                CoverageObligation(
                    "exceptional_stratum",
                    stratum.condition,
                    False,
                    "Jacobian-minor atlas unavailable",
                )
            )
            continue
        subchildren, subobligations = _resolve_stratum_atlas_images(
            angular, atlas, angular_vars, depth + 1, max_depth
        )
        children.extend(subchildren)
        child_obligations.extend(subobligations)
    if (
        result.certified
        and child_obligations
        and all(o.discharged for o in child_obligations)
    ):
        coverage = result.coverage.combine(
            CoverageCertificate.complete(
                "recursive_child_strata",
                "all singular exceptional strata discharged",
                tuple(s.condition for s in strata),
                child_obligations,
            ),
            provider="recursive_exceptional_resolution",
            statement="regular exceptional image plus every singular child stratum is complete",
        )
    elif result.certified and not strata:
        coverage = result.coverage
    else:
        coverage = CoverageCertificate.partial(
            "recursive_exceptional_resolution",
            "regular image or child-stratum coverage remains incomplete",
            (),
            tuple(s.condition for s in strata),
            tuple(child_obligations),
        )
    return ResolutionNode(
        expr,
        chart,
        angular,
        image,
        strata,
        tuple(children),
        coverage,
        depth,
        "recursive_exceptional_resolution",
    )


__all__ = [
    "DimensionDescentCertificate",
    "ExceptionalStratum",
    "ResolutionNode",
    "SmoothStratumChart",
    "StratumAtlas",
    "StratumLocalModel",
    "finite_stratum_atlas",
    "resolve_exceptional_strata",
    "singular_locus_chain",
]
