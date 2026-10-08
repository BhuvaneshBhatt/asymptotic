"""Branch-divisor and monodromy geometry for complex cluster sets."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask
from .blowup_geometry import BlowUpChart, Valuation
from .coverage import CoverageCertificate
from .multivariate_limits_advanced import AdvancedLimitStatus


@dataclass(frozen=True)
class BranchDivisor:
    """Local branch divisor induced by a multivalued analytic operation."""

    argument: sp.Expr
    kind: str
    cut_angle: sp.Expr
    target_value: sp.Expr
    operation: sp.Expr
    parameter_condition: sp.Expr = sp.S.true


@dataclass(frozen=True)
class MonodromyTransition:
    """Transition between the two local sides of one branch divisor."""

    divisor: BranchDivisor
    upper_value: sp.Expr
    lower_value: sp.Expr
    winding_increment: int
    statement: str


@dataclass(frozen=True)
class ComplexBranchChart:
    """One branch sector attached to the common blow-up chart geometry."""

    blowup_chart: BlowUpChart
    side: str
    constraint: sp.Expr
    value: sp.Expr
    transitions: tuple[MonodromyTransition, ...]
    branch_indices: tuple[int, ...]
    provider: str = "branch_divisor_chart"


@dataclass(frozen=True)
class ComplexClusterSetResult:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    cluster_set: sp.Set | None
    charts: tuple[ComplexBranchChart, ...]
    status: AdvancedLimitStatus
    provider: str
    statement: str
    divisors: tuple[BranchDivisor, ...] = ()
    coverage: CoverageCertificate | None = None

    @property
    def certified(self):
        return (
            self.status is AdvancedLimitStatus.CERTIFIED
            and self.coverage is not None
            and self.coverage.certified
        )

    @property
    def limit_semantics(self):
        from .cluster_semantics import cluster_limit_semantics

        return cluster_limit_semantics(self)


def _side_from_domain(domain, variable):
    domain = sp.sympify(domain)
    if domain is sp.S.true:
        return None
    y = sp.im(variable)
    clauses = domain.args if isinstance(domain, sp.And) else (domain,)
    for clause in clauses:
        if clause in (sp.Gt(y, 0), sp.Ge(y, 0)):
            return "upper"
        if clause in (sp.Lt(y, 0), sp.Le(y, 0)):
            return "lower"
    return None


def _base_complex_chart(variable, target):
    r = sp.Symbol("_branch_r", positive=True)
    u = sp.Symbol("_branch_u")
    return BlowUpChart(
        (variable,),
        (target,),
        Valuation((variable,), (1,)),
        r,
        (u,),
        sp.S.true,
        sp.S.true,
        0,
        "complex_branch_blowup",
    )


def _boundary_values(node, variable, target, memo):
    """Recursively evaluate upper/lower boundary germs and collect divisors."""
    key = (node, target)
    if key in memo:
        return memo[key]
    node = sp.sympify(node)
    if node == variable:
        out = (target, target, (), ())
        memo[key] = out
        return out
    if not node.has(variable):
        out = (node, node, (), ())
        memo[key] = out
        return out

    child_data = [_boundary_values(a, variable, target, memo) for a in node.args]
    if any(d is None for d in child_data):
        return None
    upper_args = tuple(d[0] for d in child_data)
    lower_args = tuple(d[1] for d in child_data)
    divisors = tuple(dv for d in child_data for dv in d[2])
    transitions = tuple(tr for d in child_data for tr in d[3])

    if node.func is sp.log:
        au, al = upper_args[0], lower_args[0]
        # A branch divisor occurs when the child lands on the nonpositive real
        # principal cut.  The strict negative case has two finite boundary values.
        if au == al and au.is_real is True and bounded_ask(sp.Q.negative(au)) is True:
            radius = sp.Abs(au)
            upper = sp.log(radius) + sp.I * sp.pi
            lower = sp.log(radius) - sp.I * sp.pi
            divisor = BranchDivisor(node.args[0], "log", sp.pi, au, node)
            transition = MonodromyTransition(
                divisor,
                upper,
                lower,
                -1,
                "principal logarithm changes by -2*pi*I across its negative-real divisor",
            )
            out = (upper, lower, divisors + (divisor,), transitions + (transition,))
            memo[key] = out
            return out
        try:
            out = (sp.log(au), sp.log(al), divisors, transitions)
            memo[key] = out
            return out
        except (TypeError, ValueError):
            return None

    if node.func is sp.arg:
        au, al = upper_args[0], lower_args[0]
        if au == al and au.is_real is True and bounded_ask(sp.Q.negative(au)) is True:
            divisor = BranchDivisor(node.args[0], "arg", sp.pi, au, node)
            transition = MonodromyTransition(
                divisor,
                sp.pi,
                -sp.pi,
                -1,
                "principal argument jumps by -2*pi across its negative-real divisor",
            )
            out = (sp.pi, -sp.pi, divisors + (divisor,), transitions + (transition,))
            memo[key] = out
            return out
        out = (sp.arg(au), sp.arg(al), divisors, transitions)
        memo[key] = out
        return out

    if node.is_Pow:
        bu, bl = upper_args[0], lower_args[0]
        exponent = node.exp
        if exponent.has(variable):
            return None
        if bu == bl and bu.is_real is True and bounded_ask(sp.Q.negative(bu)) is True:
            if exponent.is_integer is True:
                value = sp.simplify(bu**exponent)
                out = (value, value, divisors, transitions)
                memo[key] = out
                return out
            radius = sp.Abs(bu)
            upper = sp.simplify(radius**exponent * sp.exp(sp.I * sp.pi * exponent))
            lower = sp.simplify(radius**exponent * sp.exp(-sp.I * sp.pi * exponent))
            divisor = BranchDivisor(node.base, "power", sp.pi, bu, node)
            transition = MonodromyTransition(
                divisor,
                upper,
                lower,
                -1,
                "principal power acquires exp(-2*pi*I*exponent) monodromy across its divisor",
            )
            out = (upper, lower, divisors + (divisor,), transitions + (transition,))
            memo[key] = out
            return out
        out = (
            sp.simplify(bu**exponent),
            sp.simplify(bl**exponent),
            divisors,
            transitions,
        )
        memo[key] = out
        return out

    # Analytic outer composition propagates already-computed branch sides.
    if node.func in (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.tan):
        out = (
            sp.simplify(node.func(*upper_args)),
            sp.simplify(node.func(*lower_args)),
            divisors,
            transitions,
        )
        memo[key] = out
        return out

    # Algebraic arithmetic composes branch germs structurally.
    if node.is_Add or node.is_Mul:
        out = (
            sp.simplify(node.func(*upper_args)),
            sp.simplify(node.func(*lower_args)),
            divisors,
            transitions,
        )
        memo[key] = out
        return out
    return None


def branch_divisor_atlas(expr, variable, target, *, domain=True):
    """Build a local branched atlas and its monodromy transitions."""
    expr, variable, target = map(sp.sympify, (expr, variable, target))
    data = _boundary_values(expr, variable, target, {})
    if data is None:
        return (
            (),
            (),
            CoverageCertificate.unknown(
                "branch_divisor_unknown", "branch divisor structure was not certified"
            ),
        )
    upper, lower, divisors, transitions = data
    side = _side_from_domain(domain, variable)
    chart = _base_complex_chart(variable, target)
    upper_chart = ComplexBranchChart(
        chart,
        "upper",
        sp.Gt(sp.im(variable), 0),
        upper,
        transitions,
        tuple(0 for _ in transitions),
    )
    lower_chart = ComplexBranchChart(
        chart,
        "lower",
        sp.Lt(sp.im(variable), 0),
        lower,
        transitions,
        tuple(t.winding_increment for t in transitions),
    )
    if side == "upper":
        return (
            (upper_chart,),
            divisors,
            CoverageCertificate.complete(
                "branched_halfplane_cover",
                "upper branch sector is exhausted",
                ("upper",),
            ),
        )
    if side == "lower":
        return (
            (lower_chart,),
            divisors,
            CoverageCertificate.complete(
                "branched_halfplane_cover",
                "lower branch sector is exhausted",
                ("lower",),
            ),
        )
    if sp.sympify(domain) is sp.S.true:
        return (
            (upper_chart, lower_chart),
            divisors,
            CoverageCertificate.complete(
                "branch_divisor_monodromy_cover",
                "the two local sides of every detected divisor exhaust the punctured branch atlas",
                ("upper", "lower", "divisor"),
            ),
        )
    # General relative domains are intersected with the branch sides.  For a
    # complex scalar represented by z, semialgebraic predicates in re(z),im(z)
    # are reduced to real local coordinates before component analysis.
    xr, yi = sp.symbols("_branch_x _branch_y", real=True)
    real_domain = sp.sympify(domain).subs({sp.re(variable): xr, sp.im(variable): yi})
    if not real_domain.has(variable):
        # A conjunction of strict polynomial inequalities that is strictly true
        # at the target contains a full local neighbourhood.  In that case the
        # relative domain does not remove either side of the punctured branch
        # atlas, and no semialgebraic component backend is needed.
        clauses = sp.And.make_args(real_domain)
        target_substitution = {xr: sp.re(target), yi: sp.im(target)}
        if clauses and all(
            isinstance(clause, (sp.StrictLessThan, sp.StrictGreaterThan))
            and sp.simplify(clause.subs(target_substitution)) is sp.S.true
            for clause in clauses
        ):
            return (
                (upper_chart, lower_chart),
                divisors,
                CoverageCertificate.complete(
                    "branch_relative_domain_interior",
                    "strict relative domain contains a full punctured neighbourhood",
                    ("upper", "lower"),
                ),
            )
        try:
            from .domain_cluster_geometry import local_domain_components

            geometry = local_domain_components(
                real_domain, (xr, yi), (sp.re(target), sp.im(target))
            )
            if geometry.certified:
                selected = []
                from semialg import is_satisfiable

                for branch_chart, side_condition in (
                    (upper_chart, yi > 0),
                    (lower_chart, yi < 0),
                ):
                    if any(
                        bool(
                            is_satisfiable(
                                sp.And(c.condition, side_condition), (xr, yi)
                            )
                        )
                        for c in geometry.components
                    ):
                        selected.append(branch_chart)
                if selected:
                    return (
                        tuple(selected),
                        divisors,
                        CoverageCertificate.complete(
                            "branch_domain_component_cover",
                            "branch sides were intersected with all accumulating semialgebraic domain components",
                            tuple(c.side for c in selected),
                        ),
                    )
        except (
            ImportError,
            AttributeError,
            NotImplementedError,
            TypeError,
            ValueError,
        ):
            pass
    return (
        (),
        divisors,
        CoverageCertificate.unknown(
            "branch_relative_domain_unknown",
            "relative complex domain is not a certified branch-sector intersection",
        ),
    )


def complex_branch_cluster_set(expr, variable, target, *, domain=True):
    """Compute complete finite cluster images over a certified branched atlas."""
    expr, variable, target = map(sp.sympify, (expr, variable, target))
    charts, divisors, coverage = branch_divisor_atlas(
        expr, variable, target, domain=domain
    )
    if not charts or not coverage.certified:
        return ComplexClusterSetResult(
            expr,
            (variable,),
            (target,),
            None,
            charts,
            AdvancedLimitStatus.UNKNOWN,
            "branch_divisor_geometry",
            "branched local geometry was not certified",
            divisors,
            coverage=coverage,
        )
    cluster = sp.FiniteSet(*(chart.value for chart in charts))
    return ComplexClusterSetResult(
        expr,
        (variable,),
        (target,),
        cluster,
        charts,
        AdvancedLimitStatus.CERTIFIED,
        "branch_divisor_monodromy_cluster",
        "complete cluster image over a certified branch-divisor atlas",
        divisors,
        coverage=coverage,
    )


@dataclass(frozen=True)
class PulledBranchDivisor:
    """A branch divisor pulled back through one common local geometry chart."""

    operation: sp.Expr
    argument: sp.Expr
    transformed_argument: sp.Expr
    cut_condition: sp.Expr
    branch_point_condition: sp.Expr
    kind: str


@dataclass(frozen=True)
class BranchedBlowUpChart:
    """Common blow-up chart enriched by branch divisors and sector data."""

    blowup_chart: BlowUpChart
    domain: sp.Expr
    divisors: tuple[PulledBranchDivisor, ...]
    sectors: tuple[sp.Expr, ...]
    monodromy: tuple[MonodromyTransition, ...]
    coverage: CoverageCertificate
    provider: str = "branched_blowup_geometry"

    def transform(self, expression):
        return self.blowup_chart.transform(expression)

    def pullback_domain(self, domain):
        return self.blowup_chart.pullback_domain(domain)

    @property
    def exceptional_domain(self):
        return sp.And(self.blowup_chart.exceptional_domain, self.domain)


def _branch_nodes(expr):
    out = []
    for node in sp.preorder_traversal(sp.sympify(expr)):
        if node.func in (sp.log, sp.arg):
            out.append((node, node.args[0], node.func.__name__))
        elif node.is_Pow and node.exp.is_integer is not True:
            out.append((node, node.base, "power"))
    # inner operations first makes nested branch geometry deterministic
    return tuple(reversed(out))


def _pulled_branch_divisor(node, arg, kind, chart):
    transformed = sp.cancel(sp.together(chart.transform(arg)))
    re_arg = sp.simplify(sp.re(transformed))
    im_arg = sp.simplify(sp.im(transformed))
    cut = sp.And(sp.Eq(im_arg, 0), sp.Le(re_arg, 0))
    point = sp.And(sp.Eq(re_arg, 0), sp.Eq(im_arg, 0))
    return PulledBranchDivisor(node, arg, transformed, cut, point, kind)


def _branch_sectors(divisors, base_domain):
    """Finite exact sign sectors of pulled principal cuts.

    Empty sectors may remain; they do not compromise coverage and avoiding a
    CAD call here keeps branch-atlas construction a cheap structural step.
    """
    sectors = (sp.sympify(base_domain),)
    for divisor in divisors:
        re_arg = sp.re(divisor.transformed_argument)
        im_arg = sp.im(divisor.transformed_argument)
        candidates = (
            sp.Gt(im_arg, 0),
            sp.Lt(im_arg, 0),
            sp.And(sp.Eq(im_arg, 0), sp.Gt(re_arg, 0)),
            sp.And(sp.Eq(im_arg, 0), sp.Lt(re_arg, 0)),
        )
        sectors = tuple(sp.And(old, side) for old in sectors for side in candidates)
    return tuple(dict.fromkeys(sectors))


def branched_blowup_atlas(expr, variables, target, *, domain=True, weights=None):
    """Pull all principal branch divisors through a real multivariate blow-up.

    This is structural branch geometry: nested log/power operations share the
    same pulled-back divisors, domain and coverage protocol as real limits.
    """
    from .blowup_geometry import weighted_spherical_atlas

    variables = tuple(variables)
    target = tuple(map(sp.sympify, target))
    weights = (1,) * len(variables) if weights is None else tuple(weights)
    atlas = weighted_spherical_atlas(variables, target, weights)
    nodes = tuple(
        item
        for item in _branch_nodes(expr)
        if sp.sympify(item[1]).free_symbols.intersection(variables)
    )
    charts = []
    obligations = []
    for index, chart in enumerate(atlas.charts):
        pulled_domain = chart.pullback_domain(domain)
        exceptional_domain = sp.simplify(pulled_domain.subs(chart.radial_variable, 0))
        divisors = tuple(
            _pulled_branch_divisor(node, arg, kind, chart) for node, arg, kind in nodes
        )
        sectors = _branch_sectors(
            divisors, sp.And(chart.exceptional_domain, exceptional_domain)
        )
        # The asymptotic germ is punctured at the target.  For supported
        # logarithm/principal-power divisors, the explicit side sectors cover
        # that punctured chart; the omitted branch point is the target itself.
        supported = all(divisor.kind in {"log", "power"} for divisor in divisors)
        complete = not divisors or (supported and bool(sectors))
        coverage = (
            CoverageCertificate.complete(
                "branched_chart_sector_cover",
                "supported branch-side sectors cover the punctured branched chart",
                sectors if divisors else (exceptional_domain,),
            )
            if complete
            else CoverageCertificate.partial(
                "branched_chart_sector_cover",
                "branch-sector coverage remains unresolved for an unsupported divisor",
                sectors,
                (index,),
            )
        )
        charts.append(
            BranchedBlowUpChart(
                chart, exceptional_domain, divisors, sectors, (), coverage
            )
        )
        obligations.append(coverage)
    if atlas.coverage.certified and all(c.certified for c in obligations):
        coverage = CoverageCertificate.complete(
            "branched_blowup_atlas_cover",
            "the base blow-up atlas and every pulled branch-sector decomposition are complete",
            tuple(range(len(charts))),
        )
    else:
        coverage = CoverageCertificate.partial(
            "branched_blowup_atlas_cover",
            "one or more branch charts remain unresolved",
            (),
            tuple(i for i, c in enumerate(obligations) if not c.certified),
        )
    return tuple(charts), coverage


@dataclass(frozen=True)
class MonodromyAction:
    additive_shift: sp.Expr = sp.S.Zero
    multiplier: sp.Expr = sp.S.One
    statement: str = "trivial"


@dataclass(frozen=True)
class MonodromyGenerator:
    index: int
    divisor: PulledBranchDivisor
    winding_symbol: sp.Symbol
    relation: sp.Expr
    action: MonodromyAction = MonodromyAction()


def independent_monodromy_generators(
    expr, variables, target, *, domain=True, weights=None
):
    """Return one explicit winding generator per distinct pulled branch divisor.

    The result is structural: no relation between distinct generators is
    invented. Coincident pulled divisors share a generator only when their
    transformed arguments are symbolically identical.
    """
    charts, coverage = branched_blowup_atlas(
        expr, variables, target, domain=domain, weights=weights
    )
    if not charts:
        return (), coverage
    seen = []
    out = []
    for chart in charts:
        for d in chart.divisors:
            key = (d.kind, sp.simplify(d.transformed_argument))
            if any(k[0] == key[0] and sp.simplify(k[1] - key[1]) == 0 for k in seen):
                continue
            seen.append(key)
            n = sp.Symbol(f"_winding_{len(out)}", integer=True)
            op = d.operation
            if getattr(op, "func", None) is sp.log:
                action = MonodromyAction(
                    2 * sp.pi * sp.I * n,
                    sp.S.One,
                    "principal logarithm gains 2*pi*i per winding",
                )
            elif getattr(op, "is_Pow", False):
                action = MonodromyAction(
                    sp.S.Zero,
                    sp.exp(2 * sp.pi * sp.I * op.exp * n),
                    "power acquires exp(2*pi*i*exponent*n)",
                )
            else:
                action = MonodromyAction(
                    statement="detected divisor has no supported explicit monodromy action"
                )
            out.append(
                MonodromyGenerator(
                    len(out), d, n, sp.Contains(n, sp.S.Integers), action
                )
            )
    # Intersections are covered only when the branched atlas itself certified
    # every sector.  Independent generators then form the product action; no
    # commutation relation beyond integer windings is invented.
    if coverage.certified and all(
        g.action.statement
        != "detected divisor has no supported explicit monodromy action"
        for g in out
    ):
        coverage = CoverageCertificate.complete(
            "monodromy_action_cover",
            "all pulled branch divisors have explicit integer-winding actions and the branched atlas covers their sector intersections",
            tuple(g.index for g in out),
        )
    elif out:
        coverage = CoverageCertificate.partial(
            "monodromy_action_cover",
            "one or more branch-divisor actions or intersections remain unsupported",
            tuple(
                g.index
                for g in out
                if g.action.statement.startswith(("principal", "power"))
            ),
            ("unsupported_monodromy_action",),
        )
    return tuple(out), coverage


__all__ = [
    "BranchDivisor",
    "BranchedBlowUpChart",
    "ComplexBranchChart",
    "ComplexClusterSetResult",
    "MonodromyAction",
    "MonodromyGenerator",
    "MonodromyTransition",
    "PulledBranchDivisor",
    "branch_divisor_atlas",
    "branched_blowup_atlas",
    "complex_branch_cluster_set",
    "independent_monodromy_generators",
]
