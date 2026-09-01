"""Asymptotic optimization of parameter-dependent scalar objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from ._symbolic_policy import bounded_ask, bounded_limit
from .context import AsymptoticContext
from .solve import asymptotic_solve


@dataclass(frozen=True)
class OptimizationCertificate:
    """Replayable evidence for a globally covered scalar optimization problem."""

    objective: sp.Expr
    variable: sp.Symbol
    parameter: sp.Symbol
    point: sp.Expr
    domain: sp.Set
    assumptions: sp.Expr
    sense: Literal["min", "max"]
    candidates: tuple[tuple[sp.Expr, sp.Expr, bool], ...]
    best_value: sp.Expr
    stationary_complete: bool
    globally_covered: bool

    def replay(self) -> bool | None:
        if not self.stationary_complete or not self.globally_covered:
            return False
        context = AsymptoticContext(self.parameter, point=self.point)
        for location, value, attained in self.candidates:
            if attained:
                actual = _truncate(
                    self.objective.subs(self.variable, location),
                    self.parameter,
                    self.point,
                    6,
                )
                if sp.simplify(actual - value) != 0:
                    return False
            comparison = _eventual_compare(
                value, self.best_value, context=context, sense=self.sense
            )
            if comparison == 1 or comparison is None:
                return False
        return True


@dataclass(frozen=True)
class AsymptoticOptimizationResult:
    """Result of a univariate asymptotic optimization problem."""

    optimum_value: sp.Expr
    optimizers: tuple[sp.Expr, ...]
    variable: sp.Symbol
    parameter: sp.Symbol
    point: sp.Expr
    sense: Literal["min", "max"]
    status: str
    method: str
    conditions: tuple[sp.Expr, ...] = ()
    certificate: object | None = None
    approached_boundaries: tuple[sp.Expr, ...] = ()

    @property
    def attained(self) -> bool:
        return bool(self.optimizers)

    @property
    def certified(self) -> bool:
        return self.status in {"EXACT", "CERTIFIED"}


def _truncate(expr: sp.Expr, parameter: sp.Symbol, point: sp.Expr, terms: int) -> sp.Expr:
    expr = sp.sympify(expr)
    if parameter not in expr.free_symbols:
        return expr
    try:
        return sp.series(expr, parameter, point, terms).removeO()
    except (ValueError, TypeError, NotImplementedError):
        return expr


def _domain_contains(value: sp.Expr, domain: sp.Set, assumptions: sp.Expr) -> bool | None:
    if domain == sp.S.Reals:
        answer = bounded_ask(sp.Q.real(value), assumptions)
        if answer in (True, False):
            return answer
    try:
        contains = sp.refine(sp.Contains(value, domain), assumptions)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if contains is sp.S.true:
        return True
    if contains is sp.S.false:
        return False
    return None


def _interval_endpoints(domain: sp.Set) -> tuple[sp.Expr, ...]:
    if not isinstance(domain, sp.Interval):
        return ()
    out: list[sp.Expr] = []
    if not domain.left_open and domain.start not in (-sp.oo, sp.oo):
        out.append(domain.start)
    if not domain.right_open and domain.end not in (-sp.oo, sp.oo) and domain.end != domain.start:
        out.append(domain.end)
    return tuple(out)


def _interval_boundary_limits(
    objective: sp.Expr,
    variable: sp.Symbol,
    domain: sp.Set,
) -> tuple[tuple[sp.Expr, sp.Expr], ...]:
    """Return limits at interval boundaries that are not attained points."""

    if not isinstance(domain, sp.Interval):
        return ()
    boundaries: list[tuple[sp.Expr, sp.Expr]] = []
    candidates = (
        (domain.start, "+", domain.left_open or domain.start is -sp.oo),
        (domain.end, "-", domain.right_open or domain.end is sp.oo),
    )
    for boundary, direction, needed in candidates:
        if not needed:
            continue
        value = bounded_limit(
            objective, variable, boundary, direction=direction, allow_general=True
        )
        if value is None:
            continue
        if isinstance(value, sp.Limit) or value.has(sp.nan, sp.zoo):
            continue
        boundaries.append((boundary, sp.sympify(value)))
    return tuple(boundaries)


def _eventual_compare(
    left: sp.Expr,
    right: sp.Expr,
    *,
    context: AsymptoticContext,
    sense: Literal["min", "max"],
) -> int | None:
    """Return -1/0/1 when left is eventually worse/equal/better than right."""
    sign = context.eventual_sign(sp.simplify(left - right))
    if sign is None:
        return None
    if sign == 0:
        return 0
    if sense == "min":
        return 1 if sign < 0 else -1
    return 1 if sign > 0 else -1


def _candidate_stationary_points(
    objective: sp.Expr,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    *,
    point: sp.Expr,
    terms: int,
    domain: sp.Set,
    assumptions: sp.Expr,
) -> tuple[tuple[tuple[sp.Expr, str, object | None], ...], bool]:
    """Enumerate exact, relaxed-lattice, and boundary candidates for one optimizer."""
    derivative = sp.diff(objective, variable)
    candidates: list[tuple[sp.Expr, str, object | None]] = []

    # ``solveset`` distinguishes a complete finite root set from an unresolved
    # ConditionSet/ImageSet.  That distinction matters for certification.
    solve_variable = variable
    solve_derivative = derivative
    solve_domain = domain
    if domain == sp.S.Integers:
        # A Symbol declared integer causes solveset(..., domain=Reals) to retain
        # the integer assumption and reject the continuous stationary point.
        # Solve the relaxation in an assumption-free real dummy instead.
        solve_variable = sp.Dummy(f"{variable.name}_continuous", real=True)
        solve_derivative = derivative.xreplace({variable: solve_variable})
        solve_domain = sp.S.Reals
    try:
        exact_set = sp.solveset(sp.Eq(solve_derivative, 0), solve_variable, domain=solve_domain)
    except (NotImplementedError, ValueError, TypeError):
        exact_set = None
    exact_complete = isinstance(exact_set, sp.FiniteSet) or exact_set is sp.S.EmptySet
    if exact_complete:
        for root in exact_set:
            root = sp.sympify(root)
            if domain == sp.S.Integers:
                lattice = (sp.floor(root), sp.ceiling(root))
                for candidate in lattice:
                    candidate = sp.refine(candidate, assumptions)
                    if not any(sp.simplify(candidate - old[0]) == 0 for old in candidates):
                        candidates.append((candidate, "lattice-rounded-stationary", None))
            elif _domain_contains(root, domain, assumptions) is not False:
                candidates.append((root, "exact-stationary", None))

    # The asymptotic solver is important when exact roots are unresolved or
    # exact radicals/transcendentals would obscure the Hardy-field balance.
    if not exact_complete or derivative.has(sp.exp, sp.log):
        try:
            solved = asymptotic_solve(
                derivative,
                variable,
                parameter=parameter,
                point=point,
                terms=terms,
                domain=domain,
                assumptions=assumptions,
            )
        except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
            solved = None
        if solved is not None:
            for branch in solved.branches:
                root = branch.as_dict().get(variable)
                if root is None or _domain_contains(root, domain, assumptions) is False:
                    continue
                if not any(sp.simplify(root - old[0]) == 0 for old in candidates):
                    candidates.append((root, branch.method, branch.certificate))

    return tuple(candidates), exact_complete


def _global_curvature_decision(
    objective: sp.Expr,
    variable: sp.Symbol,
    *,
    assumptions: sp.Expr,
    sense: Literal["min", "max"],
) -> bool | None:
    """Conservatively decide global convexity/concavity from the Hessian scalar."""
    second = sp.refine(sp.diff(objective, variable, 2), assumptions)
    predicate = sp.Q.nonnegative(second) if sense == "min" else sp.Q.nonpositive(second)
    answer = bounded_ask(predicate, assumptions)
    if answer in (True, False):
        return answer
    if variable not in second.free_symbols:
        sign = sp.sign(sp.simplify(second))
        if sense == "min" and sign in (0, 1):
            return True
        if sense == "max" and sign in (0, -1):
            return True
    return None


def _asymptotic_optimize(
    objective: sp.Expr,
    variable: sp.Symbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    domain: sp.Set = sp.S.Reals,
    assumptions: sp.Expr | bool = sp.S.true,
    sense: Literal["min", "max"],
) -> AsymptoticOptimizationResult:
    """Compare optimizer candidates asymptotically and certify global coverage when possible."""
    if terms < 1:
        raise ValueError("terms must be positive")
    objective = sp.sympify(objective)
    variable = sp.sympify(variable)
    parameter = sp.sympify(parameter)
    if not isinstance(variable, sp.Symbol) or not isinstance(parameter, sp.Symbol):
        raise TypeError("variable and parameter must be SymPy symbols")
    if variable == parameter:
        raise ValueError("optimization variable and asymptotic parameter must differ")
    assumptions = sp.sympify(assumptions)
    domain = sp.sympify(domain)
    ctx = AsymptoticContext(parameter, point=point)

    stationary_candidates, stationary_complete = _candidate_stationary_points(
        objective,
        variable,
        parameter,
        point=point,
        terms=terms,
        domain=domain,
        assumptions=assumptions,
    )
    raw_candidates = [
        (location, method, certificate, True)
        for location, method, certificate in stationary_candidates
    ]
    raw_candidates.extend(
        (endpoint, "finite-endpoint", None, True) for endpoint in _interval_endpoints(domain)
    )

    evaluated: list[tuple[sp.Expr, sp.Expr, str, object | None, bool]] = []
    for location, method, certificate, attained in raw_candidates:
        value = _truncate(objective.subs(variable, location), parameter, point, terms)
        if value.has(sp.nan, sp.zoo):
            continue
        evaluated.append((location, value, method, certificate, attained))
    for boundary, boundary_value in _interval_boundary_limits(objective, variable, domain):
        value = _truncate(boundary_value, parameter, point, terms)
        if value.has(sp.nan, sp.zoo):
            continue
        evaluated.append((boundary, value, "open-boundary-limit", None, False))

    if not evaluated:
        return AsymptoticOptimizationResult(
            sp.nan, (), variable, parameter, point, sense, "UNKNOWN", "no-candidates"
        )

    best = [evaluated[0]]
    undecided = False
    for candidate in evaluated[1:]:
        cmp = _eventual_compare(candidate[1], best[0][1], context=ctx, sense=sense)
        if cmp == 1:
            best = [candidate]
        elif cmp == 0:
            best.append(candidate)
        elif cmp is None:
            undecided = True

    locations = tuple(item[0] for item in best if item[4])
    approached_boundaries = tuple(item[0] for item in best if not item[4])
    value = best[0][1]
    curvature = _global_curvature_decision(
        objective, variable, assumptions=assumptions, sense=sense
    )
    all_stationary_found = stationary_complete and all(
        item[2]
        in {
            "exact-stationary",
            "lattice-rounded-stationary",
            "finite-endpoint",
            "open-boundary-limit",
        }
        for item in evaluated
    )
    globally_covered = isinstance(domain, sp.Interval) or curvature is True
    status = (
        "CERTIFIED" if not undecided and all_stationary_found and globally_covered else "FORMAL"
    )
    methods = {item[2] for item in best}
    method = methods.pop() if len(methods) == 1 else "mixed-stationary-comparison"
    certificate = None
    if status == "CERTIFIED":
        certificate = OptimizationCertificate(
            objective=objective,
            variable=variable,
            parameter=parameter,
            point=point,
            domain=domain,
            assumptions=assumptions,
            sense=sense,
            candidates=tuple((item[0], item[1], item[4]) for item in evaluated),
            best_value=value,
            stationary_complete=all_stationary_found,
            globally_covered=globally_covered,
        )
    return AsymptoticOptimizationResult(
        value,
        locations,
        variable,
        parameter,
        point,
        sense,
        status,
        method,
        certificate=certificate,
        approached_boundaries=approached_boundaries,
    )


def asymptotic_minimize(
    objective: sp.Expr,
    variable: sp.Symbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    domain: sp.Set = sp.S.Reals,
    assumptions: sp.Expr | bool = sp.S.true,
) -> AsymptoticOptimizationResult:
    """Asymptotically minimize a univariate parameter-dependent objective."""
    return _asymptotic_optimize(
        objective,
        variable,
        parameter=parameter,
        point=point,
        terms=terms,
        domain=domain,
        assumptions=assumptions,
        sense="min",
    )


def asymptotic_maximize(
    objective: sp.Expr,
    variable: sp.Symbol,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    domain: sp.Set = sp.S.Reals,
    assumptions: sp.Expr | bool = sp.S.true,
) -> AsymptoticOptimizationResult:
    """Asymptotically maximize a univariate parameter-dependent objective."""
    return _asymptotic_optimize(
        objective,
        variable,
        parameter=parameter,
        point=point,
        terms=terms,
        domain=domain,
        assumptions=assumptions,
        sense="max",
    )


def asymptotic_argmin(*args, **kwargs) -> tuple[sp.Expr, ...]:
    """Return the asymptotic minimizers of a scalar objective."""
    return asymptotic_minimize(*args, **kwargs).optimizers


def asymptotic_argmax(*args, **kwargs) -> tuple[sp.Expr, ...]:
    """Return the asymptotic maximizers of a scalar objective."""
    return asymptotic_maximize(*args, **kwargs).optimizers
