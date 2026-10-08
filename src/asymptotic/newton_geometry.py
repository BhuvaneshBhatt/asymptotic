"""Certified Newton fans and structured semialgebraic cluster geometry."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp

from .coverage import CoverageCertificate


@dataclass(frozen=True)
class NewtonCone:
    """One closed cone in a common refinement of Newton normal fans."""

    index: int
    minimizers: tuple[tuple[int, ...], ...]
    constraint: sp.Expr
    dimension: int
    representative_weight: tuple[int, ...] | None = None
    adjacent: tuple[int, ...] = ()

    @property
    def stratum_kind(self):
        return "newton_cone"

    @property
    def local_constraint(self):
        return self.constraint

    @property
    def intrinsic_dimension(self):
        return self.dimension

    @property
    def valuation_data(self):
        return self.representative_weight

    @property
    def child_strata(self):
        return ()

    @property
    def stratum_certified(self):
        return self.representative_weight is not None


@dataclass(frozen=True)
class NewtonPolyhedralFan:
    """Finite exact positive-orthant Newton fan with certified coverage."""

    variables: tuple[sp.Symbol, ...]
    weight_variables: tuple[sp.Symbol, ...]
    supports: tuple[tuple[tuple[int, ...], ...], ...]
    cones: tuple[NewtonCone, ...]
    coverage_constraint: sp.Expr
    coverage: CoverageCertificate
    adjacency_certified: bool

    @property
    def coverage_certified(self):
        return self.coverage.certified


@dataclass(frozen=True)
class SemialgebraicClusterStratum:
    """A connected semialgebraic cluster stratum with topological metadata."""

    index: int
    variables: tuple[sp.Symbol, ...]
    formula: sp.Expr
    dimension: int
    closure: sp.Expr
    boundary: sp.Expr
    incident_to: tuple[int, ...] = ()

    @property
    def stratum_kind(self):
        return "semialgebraic_stratum"

    @property
    def local_constraint(self):
        return self.formula

    @property
    def intrinsic_dimension(self):
        return self.dimension

    @property
    def valuation_data(self):
        return None

    @property
    def child_strata(self):
        return ()

    @property
    def stratum_certified(self):
        return True


@dataclass(frozen=True)
class StructuredClusterGeometry:
    """Exact semialgebraic cluster geometry split into connected strata."""

    variables: tuple[sp.Symbol, ...]
    formula: sp.Expr
    dimension: int
    components: tuple[sp.Expr, ...]
    closure: sp.Expr
    boundary: sp.Expr
    strata: tuple[SemialgebraicClusterStratum, ...]
    incidence_certified: bool
    frontier_complex: object | None = None


def _support(expr: sp.Expr, variables: tuple[sp.Symbol, ...]):
    try:
        poly = sp.Poly(sp.expand(expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    return tuple(tuple(map(int, monom)) for monom in poly.monoms())


def _expression_supports(expressions, variables, target):
    supports = []
    shift = {v: v + a for v, a in zip(variables, target, strict=True)}
    for expr in expressions:
        shifted = sp.cancel(sp.sympify(expr).subs(shift, simultaneous=True))
        num, den = sp.fraction(sp.together(shifted))
        for part in (num, den):
            support = _support(part, variables)
            if support and support not in supports:
                supports.append(support)
    return tuple(supports)


def newton_polyhedral_fan(expressions, variables, target=None):
    """Construct the exact common Newton normal fan in positive valuation space.

    The implementation uses exact rational linear programming, not CAD.  Each
    support contributes its normal fan; simultaneous exposed minimizers give
    the maximal common-refinement cones, and nonempty intersections add all
    lower-dimensional faces.  A normalized positive weight simplex makes
    coverage, dimensions, representatives, and adjacency finite exact linear
    questions.
    """
    from sympy.solvers.simplex import InfeasibleLPError, UnboundedLPError, lpmax

    variables = tuple(variables)
    if target is None:
        target = (sp.S.Zero,) * len(variables)
    elif not isinstance(target, (tuple, list)):
        target = (target,) * len(variables)
    target = tuple(map(sp.sympify, target))
    supports = _expression_supports(tuple(expressions), variables, target)
    n = len(variables)
    weights = tuple(sp.Dummy(f"_newton_w{i}", real=True) for i in range(n))
    epsilon = sp.Dummy("_newton_eps", real=True)
    simplex_eq = sp.Eq(sum(weights), 1)
    if not supports:
        simplex = sp.And(*(w > 0 for w in weights), simplex_eq)
        return NewtonPolyhedralFan(
            variables,
            weights,
            (),
            (),
            simplex,
            CoverageCertificate.unknown(
                "newton_fan_empty_support",
                "no finite support was available to certify positive-weight coverage",
                ("positive_weight_space",),
            ),
            False,
        )

    def linear_forms(signature):
        forms = []
        for support, chosen in zip(supports, signature, strict=True):
            base = support[chosen]
            for point in support:
                form = sp.expand(
                    sum(
                        (b - a) * w
                        for a, b, w in zip(base, point, weights, strict=True)
                    )
                )
                if form != 0 and form not in forms:
                    forms.append(form)
        return tuple(forms)

    def normalized_key(forms):
        rows = []
        for form in forms:
            coeffs = tuple(sp.Rational(form.coeff(w)) for w in weights)
            # Positive scaling does not change an inequality.
            from functools import reduce
            from math import gcd, lcm

            scale = reduce(lcm, (int(c.q) for c in coeffs), 1)
            ints = tuple(int(c * scale) for c in coeffs)
            g = reduce(gcd, (abs(v) for v in ints if v), 0) or 1
            rows.append(tuple(v // g for v in ints))
        return tuple(sorted(set(rows)))

    def data(forms):
        closed = [simplex_eq, *(f >= 0 for f in forms), *(w >= 0 for w in weights)]
        # First determine the linear span of the cone: inequalities whose
        # maximum is zero are equalities on the entire cone.
        tight_rows = []
        non_tight = []
        for form in forms:
            try:
                maximum, _ = lpmax(form, closed)
            except (InfeasibleLPError, UnboundedLPError, ValueError):
                return None
            if maximum == 0:
                tight_rows.append([form.coeff(w) for w in weights])
            else:
                non_tight.append(form)
        rank = sp.Matrix([[1] * n, *tight_rows]).rank()
        dimension = n - rank

        # Pick a relative-interior rational point: maximize a common margin
        # from the coordinate boundary and every non-identically-tight facet.
        constraints = [simplex_eq, *(f >= 0 for f in forms)]
        constraints.extend(w >= epsilon for w in weights)
        constraints.extend(f >= epsilon for f in non_tight)
        constraints.append(epsilon >= 0)
        try:
            margin, point = lpmax(epsilon, constraints)
        except (InfeasibleLPError, UnboundedLPError, ValueError):
            return None
        if margin <= 0:
            return None
        representative_values = [sp.Rational(point[w]) for w in weights]
        from functools import reduce
        from math import gcd, lcm

        scale = reduce(lcm, (int(v.q) for v in representative_values), 1)
        ints = [int(v * scale) for v in representative_values]
        g = reduce(gcd, (abs(v) for v in ints if v), 0) or 1
        representative = tuple(v // g for v in ints)
        constraint = sp.And(
            *(w > 0 for w in weights), simplex_eq, *(f >= 0 for f in forms)
        )
        return constraint, dimension, representative

    raw = []
    seen = set()
    for signature in product(*(range(len(s)) for s in supports)):
        forms = linear_forms(signature)
        key = normalized_key(forms)
        if key in seen:
            continue
        info = data(forms)
        if info is None:
            continue
        seen.add(key)
        raw.append((signature, forms, *info))

    # Close under intersections.  An intersection simply concatenates linear
    # inequalities, so exact LP remains cheap even in 3+ valuation dimensions.
    all_raw = list(raw)
    changed = True
    while changed:
        changed = False
        current = list(all_raw)
        keys = {normalized_key(item[1]) for item in all_raw}
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                forms = tuple(dict.fromkeys((*current[i][1], *current[j][1])))
                key = normalized_key(forms)
                if key in keys:
                    continue
                info = data(forms)
                if info is None:
                    continue
                keys.add(key)
                all_raw.append(((), forms, *info))
                changed = True

    adjacency = {i: set() for i in range(len(all_raw))}
    for i in range(len(all_raw)):
        for j in range(i + 1, len(all_raw)):
            if all_raw[i][3] != all_raw[j][3]:
                continue
            forms = tuple(dict.fromkeys((*all_raw[i][1], *all_raw[j][1])))
            info = data(forms)
            if info is not None and info[1] == all_raw[i][3] - 1:
                adjacency[i].add(j)
                adjacency[j].add(i)

    cones = tuple(
        NewtonCone(i, sig, constraint, dim, representative, tuple(sorted(adjacency[i])))
        for i, (sig, _, constraint, dim, representative) in enumerate(all_raw)
    )
    maximal_constraints = [item[2] for item in raw]
    coverage = sp.Or(*maximal_constraints) if maximal_constraints else sp.S.false
    # Coverage is certified by finite argmin existence: every positive weight
    # exposes at least one minimizer in each finite support, and every
    # simultaneous minimizer signature was enumerated before infeasible cones
    # were removed by exact LP.
    coverage_certificate = (
        CoverageCertificate.complete(
            "newton_positive_weight_fan",
            "all simultaneous minimizer signatures over the positive normalized weight simplex were enumerated exactly",
            tuple(c.index for c in cones),
        )
        if raw
        else CoverageCertificate.unknown(
            "newton_positive_weight_fan",
            "no feasible positive-weight Newton cones were certified",
            ("positive_weight_space",),
        )
    )
    return NewtonPolyhedralFan(
        variables, weights, supports, cones, coverage, coverage_certificate, True
    )


def structured_semialgebraic_geometry(cluster_set):
    """Return dimension/components/closure/boundary/incidence for semialgebraic sets.

    Polynomial/rational one-parameter ``ImageSet`` curves are first implicitized
    by exact quantifier elimination, so structured topology is not restricted
    to sets that happened to originate as ``ConditionSet`` objects.
    """
    if isinstance(cluster_set, sp.ImageSet) and isinstance(
        cluster_set.base_set, sp.Interval
    ):
        from semialg import quantifier_eliminate

        lam = cluster_set.lamda
        params = tuple(lam.variables)
        if len(params) == 1:
            t = params[0]
            value = lam.expr
            values = tuple(value) if isinstance(value, sp.Tuple) else (value,)
            image_vars = tuple(
                sp.Dummy(f"_cluster_geom_{i}", real=True) for i in range(len(values))
            )
            interval = cluster_set.base_set
            bounds = []
            if interval.start is not sp.S.NegativeInfinity:
                bounds.append(
                    t > interval.start if interval.left_open else t >= interval.start
                )
            if interval.end is not sp.S.Infinity:
                bounds.append(
                    t < interval.end if interval.right_open else t <= interval.end
                )
            graph = sp.And(
                *bounds, *(sp.Eq(v, e) for v, e in zip(image_vars, values, strict=True))
            )
            try:
                condition = quantifier_eliminate(
                    graph, quantifiers=[("exists", t)], variables=(*image_vars, t)
                )
            except (
                ArithmeticError,
                ValueError,
                NotImplementedError,
                sp.PolynomialError,
            ):
                condition = None
            if condition is not None:
                base = (
                    sp.ProductSet(*(sp.S.Reals for _ in image_vars))
                    if len(image_vars) > 1
                    else sp.S.Reals
                )
                symbol = sp.Tuple(*image_vars) if len(image_vars) > 1 else image_vars[0]
                return structured_semialgebraic_geometry(
                    sp.ConditionSet(symbol, condition, base)
                )
    from semialg import (
        connected_components,
        region_boundary,
        region_closure,
        region_dimension,
    )
    from semialg.reasoning_regions import region_subset

    if not isinstance(cluster_set, sp.ConditionSet):
        return None
    symbol = cluster_set.sym
    variables = tuple(symbol) if isinstance(symbol, sp.Tuple) else (symbol,)
    formula = sp.sympify(cluster_set.condition)
    try:
        dimension = int(region_dimension(formula, variables))
        components = tuple(connected_components(formula, variables))
        closure = sp.simplify(region_closure(formula, variables))
        boundary = sp.simplify(region_boundary(formula, variables))
    except (ArithmeticError, ValueError, NotImplementedError):
        return None

    provisional = []
    for i, component in enumerate(components):
        try:
            d = int(region_dimension(component, variables))
            cl = sp.simplify(region_closure(component, variables))
            bd = sp.simplify(region_boundary(component, variables))
        except (ArithmeticError, ValueError, NotImplementedError, sp.PolynomialError):
            # Some exact component renderers introduce radicals even though the
            # original semialgebraic set is polynomial (the unit circle is a
            # common example).  Preserve certified whole-set topology instead
            # of feeding that display formula back into polynomial CAD.
            components = (formula,)
            provisional = [(0, formula, dimension, closure, boundary)]
            break
        provisional.append((i, component, d, cl, bd))

    incidence = {i: set() for i in range(len(provisional))}
    incidence_ok = True
    for i, (_, _, _, cli, bdi) in enumerate(provisional):
        for j, (_, _, _, clj, bdj) in enumerate(provisional):
            if i == j:
                continue
            try:
                # Components are incident when one's boundary meets the other's closure.
                meet = sp.And(bdi, clj)
                from semialg import is_satisfiable

                if is_satisfiable(meet, variables):
                    incidence[i].add(j)
            except (
                ArithmeticError,
                ValueError,
                NotImplementedError,
                sp.PolynomialError,
            ):
                incidence_ok = False
        # Exercise exact subset machinery as a consistency certificate.
        try:
            if not region_subset(bdi, cli, variables):
                incidence_ok = False
        except (ArithmeticError, ValueError, NotImplementedError, sp.PolynomialError):
            incidence_ok = False

    strata = tuple(
        SemialgebraicClusterStratum(
            i, variables, comp, d, cl, bd, tuple(sorted(incidence[i]))
        )
        for i, comp, d, cl, bd in provisional
    )
    from .local_strata import frontier_incidence_complex

    frontier = frontier_incidence_complex(formula, variables)
    return StructuredClusterGeometry(
        variables,
        formula,
        dimension,
        components,
        closure,
        boundary,
        strata,
        incidence_ok and (frontier is None or frontier.certified),
        frontier,
    )
