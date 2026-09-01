"""Bounded symbolic operations for internal asymptotic algorithms.

This module centralizes the package's policy for potentially expensive SymPy
operations.  Internal certification and search code should prefer the cheap,
verified routines here and return ``None`` when they cannot decide.  General
SymPy fallbacks are opt-in and guarded by a small expression-complexity budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_primitives import certification_primitive
from .instrumentation import record_symbolic_event


@dataclass(frozen=True)
class SymbolicPolicy:
    """Complexity budgets for optional general SymPy fallbacks."""

    simplify_ops: int = 160
    limit_ops: int = 120
    solve_ops: int = 100
    integrate_ops: int = 80
    assumption_ops: int = 80
    satisfiable_ops: int = 40
    polynomial_degree: int = 4


DEFAULT_SYMBOLIC_POLICY = SymbolicPolicy()


def _within(expr: sp.Expr, budget: int) -> bool:
    try:
        return int(sp.count_ops(expr, visual=False)) <= budget
    except SYMBOLIC_ERRORS:
        return False


def bounded_simplify(
    expr: sp.Expr,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
    general: bool = True,
) -> sp.Expr:
    """Simplify *expr* without forcing a large expression through ``simplify``.

    Rational expressions first use ``cancel``.  The general simplifier is used
    only below the configured operation budget.  This is an optimization
    helper, not a zero oracle: callers requiring proof must replay an identity.
    """

    record_symbolic_event("simplify_calls")
    expr = sp.sympify(expr)
    if expr.is_Atom:
        return expr
    if isinstance(expr, sp.Expr):
        try:
            if expr.is_rational_function(*tuple(expr.free_symbols)):
                expr = sp.cancel(expr)
        except SYMBOLIC_ERRORS:
            pass
    if general and _within(expr, policy.simplify_ops):
        record_symbolic_event("general_simplify_calls")
        try:
            return sp.simplify(expr)
        except (NotImplementedError, TypeError, ValueError, RecursionError):
            pass
    return expr


def bounded_limit(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    *,
    direction: str = "+",
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
    allow_general: bool = True,
) -> sp.Expr | None:
    """Return a limit when a cheap exact route or a bounded fallback succeeds.

    The rational fast path is deliberately gated by ``is_rational_function``;
    attempting ``cancel``/``Poly`` on arbitrary nested exp-log expressions is
    itself expensive and defeats the purpose of this policy layer.
    """

    record_symbolic_event("limit_calls")
    expr = sp.sympify(expr)
    if variable not in expr.free_symbols:
        return expr

    is_rational = False
    try:
        is_rational = bool(expr.is_rational_function(variable))
    except SYMBOLIC_ERRORS:
        is_rational = False

    # Direct substitution is cheap at regular finite points.  Only simplify a
    # rational result, where cancellation is deterministic and bounded.
    if point not in (sp.oo, -sp.oo):
        try:
            direct = expr.subs(variable, point)
            if is_rational:
                direct = sp.cancel(direct)
            if direct not in (sp.nan, sp.zoo, sp.oo, -sp.oo) and not direct.has(sp.nan, sp.zoo):
                return direct
        except SYMBOLIC_ERRORS:
            direct = None

    if is_rational:
        try:
            local = sp.Dummy("_h", positive=True)
            if point is sp.oo:
                transformed = sp.cancel(expr.subs(variable, 1 / local))
            elif point is -sp.oo:
                transformed = sp.cancel(expr.subs(variable, -1 / local))
            else:
                sign = -1 if direction == "-" else 1
                transformed = sp.cancel(expr.subs(variable, point + sign * local))
            num, den = sp.fraction(transformed)
            pnum = sp.Poly(num, local)
            pden = sp.Poly(den, local)
            if pden.is_zero:
                return None

            def lowest(poly: sp.Poly) -> tuple[int, sp.Expr] | None:
                terms = sorted(poly.terms(), key=lambda item: item[0][0])
                if not terms:
                    return None
                (power,), coeff = terms[0]
                return int(power), coeff

            nlead = lowest(pnum)
            dlead = lowest(pden)
            if nlead is not None and dlead is not None:
                npow, nc = nlead
                dpow, dc = dlead
                power = npow - dpow
                ratio = sp.cancel(nc / dc)
                if power > 0:
                    return sp.S.Zero
                if power == 0:
                    return ratio
                if ratio.is_positive is True:
                    return sp.oo
                if ratio.is_negative is True:
                    return -sp.oo
        except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError):
            pass

    if allow_general and _within(expr, policy.limit_ops):
        record_symbolic_event("general_limit_calls")
        try:
            value = sp.limit(expr, variable, point, dir=direction)
        except (NotImplementedError, TypeError, ValueError, RecursionError):
            return None
        return None if isinstance(value, sp.Limit) else value
    if allow_general:
        record_symbolic_event("declined_by_budget")
    return None


def bounded_assumption_sign(
    expr: sp.Expr,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
) -> int | None:
    """Return a sign from SymPy assumptions only for a small expression.

    This helper intentionally does not try to prove a sign by limits or root
    isolation.  Those belong to the caller's asymptotic logic.
    """

    record_symbolic_event("assumption_sign_calls")
    expr = sp.sympify(expr)
    if not (expr.is_Atom or _within(expr, policy.assumption_ops)):
        return None
    try:
        if expr.is_positive is True:
            return 1
        if expr.is_negative is True:
            return -1
        if expr.is_zero is True:
            return 0
    except (TypeError, ValueError, RecursionError):
        return None
    return None


def bounded_assumption_entails(
    condition: sp.Expr,
    assumptions: sp.Expr = sp.S.true,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
) -> bool | None:
    """Use SymPy's assumptions/SAT engines only inside explicit budgets.

    Structural Boolean implication should be handled by the caller first.
    This routine is the deliberately bounded expensive fallback.
    """

    record_symbolic_event("assumption_entails_calls")
    condition = sp.sympify(condition)
    assumptions = sp.sympify(assumptions)
    total_ops = int(sp.count_ops(condition, visual=False)) + int(
        sp.count_ops(assumptions, visual=False)
    )
    if total_ops > policy.assumption_ops:
        record_symbolic_event("declined_by_budget")
        return None
    condition = bounded_simplify(condition, policy=policy)
    assumptions = bounded_simplify(assumptions, policy=policy)
    record_symbolic_event("ask_calls")
    try:
        asked = sp.ask(sp.Q.is_true(condition), assumptions)
    except (TypeError, ValueError, IndexError, NotImplementedError, RecursionError):
        asked = None
    if asked is not None:
        return bool(asked)

    if total_ops <= policy.satisfiable_ops:
        record_symbolic_event("satisfiable_calls", 2)
        try:
            counterexample = sp.satisfiable(sp.And(assumptions, sp.Not(condition)))
            if counterexample is False:
                return True
            witness = sp.satisfiable(sp.And(assumptions, condition))
            if witness is False:
                return False
        except (TypeError, ValueError, IndexError, NotImplementedError, RecursionError):
            pass

    implication = bounded_simplify(sp.Implies(assumptions, condition), policy=policy)
    if implication is sp.S.true:
        return True
    if implication is sp.S.false:
        return False
    return None


def bounded_polynomial_roots(
    equation: sp.Expr,
    variable: sp.Symbol,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
) -> tuple[sp.Expr, ...] | None:
    """Solve a univariate polynomial exactly up to the configured degree.

    Multiplicities are intentionally collapsed because all current asymptotic
    callers use candidate roots rather than a factored polynomial basis.
    ``None`` means the routine declined the problem; ``()`` means no roots.
    """

    expr = sp.sympify(equation)
    if isinstance(expr, sp.Equality):
        expr = expr.lhs - expr.rhs
    if not _within(expr, 4 * policy.solve_ops):
        return None
    try:
        poly = sp.Poly(expr, variable)
    except sp.PolynomialError:
        return None
    degree = poly.degree()
    if degree < 1:
        return ()
    if degree > policy.polynomial_degree:
        return None
    if degree == 1:
        a, b = poly.all_coeffs()
        if a == 0:
            return None
        return (sp.cancel(-b / a),)
    if not _within(poly.as_expr(), policy.solve_ops):
        return None
    try:
        roots = sp.roots(poly.as_expr(), variable)
    except (NotImplementedError, TypeError, ValueError, RecursionError):
        return None
    if sum(int(m) for m in roots.values()) != degree:
        return None
    return tuple(roots)


def bounded_solve_one(
    equation: sp.Expr,
    variable: sp.Symbol,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
    allow_general: bool = False,
) -> tuple[sp.Expr, ...] | None:
    """Bounded solver for one unknown, preferring exact polynomial methods."""

    record_symbolic_event("solve_one_calls")
    roots = bounded_polynomial_roots(equation, variable, policy=policy)
    if roots is not None:
        return roots
    expr = (
        equation.lhs - equation.rhs if isinstance(equation, sp.Equality) else sp.sympify(equation)
    )
    if allow_general and _within(expr, policy.solve_ops):
        record_symbolic_event("general_solve_calls")
        try:
            return tuple(sp.solve(equation, variable))
        except (NotImplementedError, TypeError, ValueError, RecursionError):
            return None
    if allow_general:
        record_symbolic_event("declined_by_budget")
    return None


def bounded_solve_system(
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
    allow_general: bool = False,
) -> tuple[dict[sp.Symbol, sp.Expr], ...] | None:
    """Solve a small polynomial/linear system with bounded general fallback."""

    record_symbolic_event("solve_system_calls")
    equations = tuple(
        eq.lhs - eq.rhs if isinstance(eq, sp.Equality) else sp.sympify(eq) for eq in equations
    )
    variables = tuple(variables)
    try:
        polys = [sp.Poly(eq, *variables) for eq in equations]
        if all(poly.total_degree() <= 1 for poly in polys):
            solution_set = sp.linsolve(equations, variables)
            if solution_set is sp.EmptySet:
                return ()
            out = []
            for row in solution_set:
                if any(value.free_symbols & set(variables) for value in row):
                    return None
                out.append(dict(zip(variables, row)))
            return tuple(out)
    except (sp.PolynomialError, ValueError, TypeError):
        pass

    total_ops = sum(int(sp.count_ops(eq, visual=False)) for eq in equations)
    if allow_general and total_ops <= policy.solve_ops:
        record_symbolic_event("general_solve_calls")
        try:
            solved = sp.solve(equations, variables, dict=True)
        except (NotImplementedError, TypeError, ValueError, RecursionError):
            return None
        return tuple(dict(sol) for sol in solved)
    if allow_general:
        record_symbolic_event("declined_by_budget")
    return None


def bounded_primitive(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
    allow_general: bool = False,
    risch: bool = False,
) -> sp.Expr | None:
    """Find a verified primitive with an explicit policy for general integration."""

    record_symbolic_event("primitive_calls")
    expr = sp.sympify(expr)
    primitive = certification_primitive(expr, variable)
    if primitive is not None:
        return primitive
    if not allow_general or not _within(expr, policy.integrate_ops):
        if allow_general:
            record_symbolic_event("declined_by_budget")
        return None
    record_symbolic_event("general_integrate_calls")
    try:
        candidate = (
            sp.integrate(expr, variable, risch=risch) if risch else sp.integrate(expr, variable)
        )
    except (NotImplementedError, TypeError, ValueError, RecursionError):
        return None
    if candidate.has(sp.Integral):
        return None
    defect = bounded_simplify(sp.diff(candidate, variable) - expr, policy=policy)
    return candidate if defect == 0 else None


def bounded_rsolve(
    recurrence: sp.Expr | sp.Equality,
    sequence: sp.Expr,
    *,
    initial_conditions: dict | None = None,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
    allow_general: bool = False,
) -> sp.Expr | None:
    """Run SymPy's general recurrence solver only inside the symbolic budget."""

    record_symbolic_event("rsolve_calls")
    expr = (
        recurrence.lhs - recurrence.rhs
        if isinstance(recurrence, sp.Equality)
        else sp.sympify(recurrence)
    )
    if not allow_general or not _within(expr, policy.solve_ops):
        if allow_general:
            record_symbolic_event("declined_by_budget")
        return None
    record_symbolic_event("general_rsolve_calls")
    try:
        return sp.rsolve(recurrence, sequence, init=initial_conditions)
    except (NotImplementedError, TypeError, ValueError, RecursionError, sp.PolynomialError):
        return None


def bounded_ask(
    predicate: sp.Expr,
    assumptions: sp.Expr = sp.S.true,
    *,
    policy: SymbolicPolicy = DEFAULT_SYMBOLIC_POLICY,
) -> bool | None:
    """Evaluate one SymPy assumption predicate inside the configured budget."""

    predicate = sp.sympify(predicate)
    assumptions = sp.sympify(assumptions)
    total_ops = int(sp.count_ops(predicate, visual=False)) + int(
        sp.count_ops(assumptions, visual=False)
    )
    if total_ops > policy.assumption_ops:
        record_symbolic_event("declined_by_budget")
        return None
    record_symbolic_event("ask_calls")
    try:
        answer = sp.ask(predicate, assumptions)
    except (TypeError, ValueError, IndexError, NotImplementedError, RecursionError):
        return None
    return bool(answer) if answer in (True, False) else None
