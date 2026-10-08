"""Proof-aware simultaneous real multivariate limits.

A simultaneous Euclidean limit ranges over the full admissible approach domain,
not an iterated sequence of one-variable limits. The solver combines exact
reductions, uniform bounds, path certificates, and local geometric methods;
unsupported cases return an explicit unknown status.
"""

from __future__ import annotations

from itertools import product

import sympy as sp

from ._limit_composition import _exact_equal, _two_sided_limit
from ._symbolic_policy import bounded_limit, bounded_polynomial_roots, bounded_solve_one
from .limit_models import (
    LimitEvidence,
)


def _candidate_paths(variables, target):
    """Generate deterministic exact paths; agreement is never proof of existence."""
    t = sp.Dummy("_limit_t", positive=True)
    n = len(variables)
    seen: set[tuple[sp.Expr, ...]] = set()

    # Generic signed rays.  All coordinates move, avoiding paths that can lie
    # wholly in a singular coordinate hyperplane.
    for signs in product((-1, 1), repeat=n):
        path = tuple(a + s * t for a, s in zip(target, signs, strict=True))
        if path not in seen:
            seen.add(path)
            yield t, path, "ray"

    # Unequal-slope rays expose angular dependence that sign-only rays miss.
    if n == 2:
        for scales in ((1, 2), (2, 1)):
            for signs in product((-1, 1), repeat=2):
                path = tuple(
                    a + s * c * t for a, s, c in zip(target, signs, scales, strict=True)
                )
                if path not in seen:
                    seen.add(path)
                    yield t, path, "scaled_ray"

    # Low-degree monomial curves catch failures invisible to straight rays,
    # including x^2*y/(x^4+y^2) along y=x^2.
    if n == 2:
        for powers in ((1, 2), (2, 1)):
            for signs in product((-1, 1), repeat=2):
                path = tuple(
                    a + s * t**p for a, s, p in zip(target, signs, powers, strict=True)
                )
                if path not in seen:
                    seen.add(path)
                    yield t, path, "monomial_curve"

    # Higher-order and tangent-refinement curves expose singular algebraic
    # branches missed by low-degree Newton rays (e.g. y=o(x) and y=+-x+o(x)).
    if n == 2:
        a0, a1 = target
        for k in (3, 4, 5, 6):
            for path in (
                (a0 + t, a1 + t**k),
                (a0 + t, a1 - t + t**k),
                (a0 + t, a1 + t + t**k),
            ):
                if path not in seen:
                    seen.add(path)
                    yield t, path, "algebraic_refinement"

    # Three-variable tangent refinements catch cancellation surfaces such as
    # z = x*y that ordinary rays miss.  These are negative witnesses only:
    # they can disprove existence when exact path limits conflict, never prove it.
    if n == 3:
        for dependent in range(3):
            for sign in (-1, 1):
                path = [target[j] + t for j in range(3)]
                path[dependent] = target[dependent] + sign * t**2 + t**5
                path = tuple(path)
                if path not in seen:
                    seen.add(path)
                    yield t, path, "algebraic_refinement_3d"
        # Linear cancellation planes need a tangent perturbation as well.
        for dependent in range(3):
            for k in (3, 4):
                path = [target[j] + t for j in range(3)]
                path[dependent] = target[dependent] - 2 * t + t**k
                path = tuple(path)
                if path not in seen:
                    seen.add(path)
                    yield t, path, "linear_tangent_refinement_3d"

    # Coordinate-axis paths are useful when the expression is defined there.
    for i in range(n):
        path = tuple(a + t if j == i else a for j, a in enumerate(target))
        if path not in seen:
            seen.add(path)
            yield t, path, "axis"


def _polynomial_support(poly_expr: sp.Expr, variables: tuple[sp.Symbol, ...]):
    """Return exponent vectors of a polynomial, or an empty tuple if unsupported."""
    try:
        poly = sp.Poly(sp.expand(poly_expr), *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return ()
    return tuple(tuple(int(e) for e in monom) for monom in poly.monoms())


def _newton_weight_vectors(expr, variables, target) -> tuple[tuple[int, ...], ...]:
    """Return candidate Newton rays from the unified geometry engine."""
    from .blowup_geometry import newton_candidate_rays

    return newton_candidate_rays(expr, tuple(variables), tuple(target))


def _exceptional_rational_paths(expr, variables, target):
    """Yield exact projective directions where rational leading forms cancel.

    The pre-pass is restricted to modest bivariate polynomial germs.  Roots of
    the lowest homogeneous numerator and denominator forms identify directions
    where generic-ray behavior can change.  Both affine projective charts are
    examined, and a quadratic transverse perturbation avoids following a
    singular divisor itself.
    """
    if len(variables) != 2 or any(
        sp.sympify(point).is_finite is False for point in target
    ):
        return
    x, y = variables
    u, v = sp.Dummy("_exceptional_u"), sp.Dummy("_exceptional_v")
    try:
        shifted = sp.cancel(
            expr.subs({x: u + target[0], y: v + target[1]}, simultaneous=True)
        )
        num, den = sp.fraction(shifted)
        pnum, pden = sp.Poly(num, u, v), sp.Poly(den, u, v)
    except (sp.PolynomialError, TypeError, ValueError):
        return
    if (
        max(pnum.total_degree(), pden.total_degree()) > 12
        or len(pnum.terms()) + len(pden.terms()) > 48
    ):
        return

    def lowest_form(poly):
        terms = poly.terms()
        degree = min(sum(monom) for monom, _ in terms)
        return sp.Add(
            *(
                coeff * u ** monom[0] * v ** monom[1]
                for monom, coeff in terms
                if sum(monom) == degree
            )
        )

    parameter = sp.Dummy("_exceptional_s", real=True)
    directions = set()
    for form in (lowest_form(pnum), lowest_form(pden)):
        for chart, polynomial in (
            ("y_over_x", form.subs({u: 1, v: parameter})),
            ("x_over_y", form.subs({u: parameter, v: 1})),
        ):
            try:
                poly = sp.Poly(polynomial, parameter)
            except (sp.PolynomialError, TypeError, ValueError):
                continue
            if not 0 < poly.degree() <= 6:
                continue
            try:
                roots = sp.roots(poly.as_expr(), parameter)
            except (sp.PolynomialError, NotImplementedError):
                continue
            for root in roots:
                if root.is_real is True:
                    directions.add((chart, sp.simplify(root)))

    t = sp.Dummy("_exceptional_t", positive=True)
    for chart, slope in sorted(
        directions, key=lambda item: (item[0], sp.default_sort_key(item[1]))
    ):
        for sign in (-1, 1):
            if chart == "y_over_x":
                path = (target[0] + t, target[1] + slope * t + sign * t**2)
            else:
                path = (target[0] + slope * t + sign * t**2, target[1] + t)
            yield t, path, f"exceptional_direction:{chart}:{slope}"


def _exceptional_algebraic_paths(expr, variables, target):
    """Lift low-degree denominator cancellation directions to algebraic curves.

    Solving the shifted denominator for either coordinate exposes curved
    divisors missed by straight projective rays.  A higher-order transverse
    perturbation stays off the divisor while retaining its asymptotic contact.
    The routine is a negative-witness generator only.
    """
    if len(variables) != 2 or any(sp.sympify(a).is_finite is False for a in target):
        return
    x, y = variables
    u = sp.Dummy("_curve_u", positive=True)
    v = sp.Dummy("_curve_v", real=True)
    try:
        shifted = sp.cancel(
            expr.subs({x: target[0] + u, y: target[1] + v}, simultaneous=True)
        )
        _, den = sp.fraction(shifted)
        poly = sp.Poly(den, u, v)
    except (sp.PolynomialError, TypeError, ValueError):
        return
    if poly.total_degree() > 6 or len(poly.terms()) > 24:
        return
    t = sp.Dummy("_curve_t", positive=True)
    for dependent, independent, chart in ((v, u, "y_of_x"), (u, v, "x_of_y")):
        try:
            roots = bounded_polynomial_roots(poly.as_expr(), dependent) or ()
        except (NotImplementedError, TypeError, ValueError):
            continue
        for root in roots[:6]:
            if root.has(sp.RootOf):
                continue
            try:
                branch = sp.series(root.subs(independent, t), t, 0, 5).removeO()
            except (NotImplementedError, TypeError, ValueError, sp.PoleError):
                continue
            # Limit variables are real.  Algebraic roots whose real-valuedness
            # on t>0 is not certified cannot be used as approach witnesses.
            if branch.is_real is not True:
                continue
            # The lifted branch must itself approach the translated target.
            # Other roots of the denominator describe remote divisor components
            # and are not local approach paths.
            try:
                if bounded_limit(branch, t, 0, direction="+", allow_general=True) != 0:
                    continue
            except (TypeError, ValueError, NotImplementedError, RecursionError):
                continue
            for sign in (-1, 1):
                if chart == "y_of_x":
                    path = (target[0] + t, target[1] + branch + sign * t**5)
                else:
                    path = (target[0] + branch + sign * t**5, target[1] + t)
                yield t, path, f"exceptional_curve:{chart}"


def _newton_paths(expr, variables, target):
    """Yield deterministic polynomial curves from Newton-support balances."""
    t = sp.Dummy("_newton_t", positive=True)
    for weights in _newton_weight_vectors(expr, variables, target):
        # Small exact coefficients catch both sign-sensitive balances and
        # coefficient-dependent cancellation without turning this into random
        # path sampling.  Path agreement remains non-probative.
        for coefficients in product((-2, -1, 1, 2), repeat=len(variables)):
            path = tuple(
                a + c * t**w
                for a, c, w in zip(target, coefficients, weights, strict=True)
            )
            yield t, path, weights


def _all_candidate_paths(expr, variables, target):
    yield from _candidate_paths(variables, target)
    seen = {path for _, path, _ in _candidate_paths(variables, target)}
    for t, path, kind in _exceptional_rational_paths(expr, variables, target):
        if path in seen:
            continue
        seen.add(path)
        yield t, path, kind
    for t, path, kind in _exceptional_algebraic_paths(expr, variables, target):
        if path in seen:
            continue
        seen.add(path)
        yield t, path, kind
    for t, path, weights in _newton_paths(expr, variables, target):
        if path in seen:
            continue
        seen.add(path)
        yield t, path, f"newton_curve:{','.join(map(str, weights))}"


def _unsafe_bessely_path(expr, substitutions, t):
    """Reject SymPy path limits across the negative-real Bessel-Y cut at zero.

    For integer-order Y_n, SymPy may drop the singular continuation term on
    a path z=-t and return a spurious finite value.  Such a path is not usable
    as negative evidence; reviewed local-germ rules handle the removable
    products instead.
    """
    sub = dict(substitutions)
    for atom in expr.atoms(sp.bessely):
        if atom.args[0].is_integer is not True:
            continue
        try:
            z = sp.simplify(atom.args[1].subs(sub, simultaneous=True))
            if bounded_limit(z, t, 0, direction="+", allow_general=True) != 0:
                continue
            # SymPy's direct limit for t*Y_n(t) can also miss the singular
            # leading term on the positive side.  Any path into z=0 is
            # therefore unsuitable as path-conflict evidence.
            return True
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            continue
    return False


def _iter_path_limits(expr, variables, target, domain=sp.S.true):
    """Yield admissible path limits so a conflict can stop further evaluation."""
    for t, path, kind in _all_candidate_paths(expr, variables, target):
        substitutions = tuple(zip(variables, path, strict=True))
        try:
            along_domain = sp.simplify(
                sp.sympify(domain).subs(dict(substitutions), simultaneous=True)
            )
            along = expr.subs(dict(substitutions), simultaneous=True)
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            continue
        if along_domain is not sp.S.true:
            continue
        direct = None
        if not _unsafe_bessely_path(expr, substitutions, t):
            direct = bounded_limit(along, t, 0, direction="+", allow_general=True)
            if direct is not None and (
                isinstance(direct, (sp.Limit, sp.Set, sp.AccumBounds))
                or direct.has(t)
                or direct in (sp.nan, sp.zoo)
            ):
                direct = None
        yield substitutions, kind, direct


def _path_bank(expr, variables, target, domain=sp.S.true):
    return tuple(_iter_path_limits(expr, variables, target, domain))


def _iter_expansion_limits(expr, variables, target, domain=sp.S.true):
    """Yield certified expansions along the canonical path subset."""
    from .local_expansion import adaptive_path_limit

    for t, path, kind in _candidate_paths(variables, target):
        substitutions = tuple(zip(variables, path, strict=True))
        try:
            along_domain = sp.simplify(
                sp.sympify(domain).subs(dict(substitutions), simultaneous=True)
            )
            along = expr.subs(dict(substitutions), simultaneous=True)
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            continue
        if along_domain is not sp.S.true:
            continue
        expanded = adaptive_path_limit(along, t)
        if expanded is not None:
            value = _finite_candidate(expanded[0], variables)
            if value is not None:
                yield substitutions, kind, value, expanded[1].depth


def _expansion_path_conflict(expr, variables, target, domain=sp.S.true):
    """Disprove a limit as soon as two certified path expansions disagree."""
    from .instrumentation import record_symbolic_event

    record_symbolic_event("path_conflict_calls")
    witnessed = []
    for substitutions, kind, value, depth in _iter_expansion_limits(
        expr, tuple(variables), tuple(target), sp.sympify(domain)
    ):
        if value is None:
            continue
        evidence = LimitEvidence(
            "local_expansion_path",
            f"certified {kind} germ through order {depth}",
            substitutions,
            value,
        )
        for previous in witnessed:
            if _exact_equal(previous.value, value) is False:
                return previous, evidence
        witnessed.append(evidence)
    return None


def _path_conflict(expr, variables, target, domain=sp.S.true):
    witnessed = []
    for substitutions, kind, direct in _iter_path_limits(
        expr, tuple(variables), tuple(target), sp.sympify(domain)
    ):
        value = direct
        if value is None:
            continue
        evidence = LimitEvidence(kind, f"limit along {kind}", substitutions, value)
        for previous in witnessed:
            if _exact_equal(previous.value, value) is False:
                return previous, evidence
        witnessed.append(evidence)
    return None


def _finite_candidate(
    value: sp.Expr | None, variables: tuple[sp.Symbol, ...]
) -> sp.Expr | None:
    if value is None:
        return None
    value = sp.sympify(value)
    if (
        value.has(*variables)
        or value.has(sp.nan, sp.zoo)
        or value in (sp.nan, sp.zoo, sp.oo, -sp.oo)
    ):
        return None
    # Sets and generalized limit envelopes are evidence of nonconvergence or
    # unresolved behavior, not scalar candidate values.
    if isinstance(value, (sp.Set, sp.AccumBounds)):
        return None
    return value


def _candidate_values(
    expr, variables, target, domain=sp.S.true
) -> tuple[tuple[sp.Expr, LimitEvidence], ...]:
    """Generate finite candidate values without treating any candidate as proof."""
    candidates: list[tuple[sp.Expr, LimitEvidence]] = []

    def add(value, method, statement, substitutions=()):
        value = _finite_candidate(value, variables)
        if value is None:
            return
        for old, _ in candidates:
            if _exact_equal(old, value) is True:
                return
        candidates.append(
            (value, LimitEvidence(method, statement, substitutions, value))
        )

    try:
        direct = expr.subs(dict(zip(variables, target, strict=True)), simultaneous=True)
    except (TypeError, ValueError, NotImplementedError, RecursionError):
        direct = None
    add(
        direct, "candidate_substitution", "finite value obtained by direct substitution"
    )

    path_values: list[tuple[sp.Expr, tuple[tuple[sp.Symbol, sp.Expr], ...], str]] = []
    for substitutions, kind, direct in _iter_path_limits(
        expr, tuple(variables), tuple(target), sp.sympify(domain)
    ):
        value = _finite_candidate(direct, variables)
        if value is not None:
            path_values.append((value, substitutions, kind))
    if path_values:
        first = path_values[0][0]
        if all(_exact_equal(first, value) is True for value, _, _ in path_values[1:]):
            add(
                first,
                "candidate_path_consensus",
                "tested exact paths agree; value is a candidate only",
                path_values[0][1],
            )

    # Simple equality-constrained domains can generate candidates after exact
    # substitution.  This is still candidate generation only; semialgebraic
    # certification must prove the relative-domain limit.
    domain_expr = sp.sympify(domain)
    equalities = domain_expr.args if isinstance(domain_expr, sp.And) else (domain_expr,)
    for relation in equalities:
        if not isinstance(relation, sp.Equality):
            continue
        for solved_var in variables:
            try:
                solutions = (
                    bounded_solve_one(relation, solved_var, allow_general=True) or ()
                )
            except (NotImplementedError, TypeError, ValueError):
                continue
            for solution in solutions:
                if solution.has(solved_var):
                    continue
                reduced_expr = sp.cancel(expr.subs(solved_var, solution))
                remaining = tuple(
                    v for v in variables if v != solved_var and reduced_expr.has(v)
                )
                if not remaining:
                    add(
                        reduced_expr,
                        "candidate_domain_equality",
                        "candidate from exact domain-equality substitution",
                    )
                elif len(remaining) == 1:
                    point = target[variables.index(remaining[0])]
                    value = _two_sided_limit(reduced_expr, remaining[0], point)
                    add(
                        value,
                        "candidate_domain_equality",
                        "candidate from exact domain-equality substitution",
                    )

    # Iterated limits are useful candidate generators but never simultaneous-limit proofs.
    if len(variables) > 1:
        for order in (variables, tuple(reversed(variables))):
            current = expr
            ok = True
            for variable in order:
                point = target[variables.index(variable)]
                value = _two_sided_limit(current, variable, point)
                if value is None:
                    ok = False
                    break
                current = value
            if ok:
                add(
                    current,
                    "candidate_iterated_limit",
                    "iterated limit used only as a candidate",
                )
    return tuple(candidates)
