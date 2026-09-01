"""Asymptotic solving for parameter-dependent algebraic systems."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import (
    bounded_assumption_entails,
    bounded_solve_one,
    bounded_solve_system,
)
from .context import AsymptoticContext
from .implicit import implicit_asymptotic


@dataclass(frozen=True)
class AsymptoticSolutionBranch:
    solution: tuple[tuple[sp.Symbol, sp.Expr], ...]
    conditions: tuple[sp.Expr, ...]
    status: str
    multiplicity: int = 1
    method: str = "exact-algebraic"
    certificate: object | None = None

    def as_dict(self) -> dict[sp.Symbol, sp.Expr]:
        return dict(self.solution)


@dataclass(frozen=True)
class AsymptoticSolveResult:
    branches: tuple[AsymptoticSolutionBranch, ...]
    parameter: sp.Symbol
    point: sp.Expr
    status: str
    method: str
    certificate: object | None = None


def _split_relations(system):
    equations = []
    inequalities = []
    for rel in system if isinstance(system, (list, tuple, sp.Tuple)) else [system]:
        rel = sp.sympify(rel)
        if isinstance(rel, sp.Equality):
            equations.append(sp.expand(rel.lhs - rel.rhs))
        elif isinstance(
            rel, (sp.StrictGreaterThan, sp.GreaterThan, sp.StrictLessThan, sp.LessThan)
        ):
            inequalities.append(rel)
        else:
            equations.append(rel)
    return equations, inequalities


def _expand_at_infinity(expr, parameter, point, terms):
    expr = sp.sympify(expr)
    if parameter not in expr.free_symbols:
        return expr, False
    try:
        out = sp.series(expr, parameter, point, terms).removeO()
        return out, sp.simplify(out - expr) != 0
    except (ValueError, TypeError, NotImplementedError):
        return expr, False


def _assumption_query(predicate: sp.Expr, assumptions: sp.Expr) -> bool | None:
    return bounded_assumption_entails(predicate, assumptions)


def _domain_decision(value: sp.Expr, domain: sp.Set, assumptions: sp.Expr) -> bool | None:
    value = sp.sympify(value)
    domain = sp.sympify(domain)
    predicate = None
    if domain == sp.S.Reals:
        predicate = sp.Q.real(value)
    elif domain == sp.S.Complexes:
        predicate = sp.Q.complex(value)
    elif domain == sp.S.Integers:
        predicate = sp.Q.integer(value)
    if predicate is not None:
        decided = _assumption_query(predicate, assumptions)
        if decided is not None:
            return decided
    try:
        contains = sp.refine(sp.Contains(value, domain), assumptions)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if contains is sp.S.true:
        return True
    if contains is sp.S.false:
        return False
    return None


def _limit_decision(
    value: sp.Expr, target: sp.Expr, ctx: AsymptoticContext, assumptions: sp.Expr
) -> tuple[bool | None, sp.Expr]:
    value = sp.refine(sp.sympify(value), assumptions)
    target = sp.sympify(target)
    limit_value = ctx.limit(value)
    if limit_value == target:
        return True, limit_value
    if isinstance(limit_value, sp.Limit):
        return None, limit_value
    if target in (sp.oo, -sp.oo):
        if limit_value in (sp.oo, -sp.oo):
            return False, limit_value
        if limit_value.is_finite is True:
            return False, limit_value
    elif limit_value in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return False, limit_value
    try:
        equality = sp.refine(sp.Eq(limit_value, target), assumptions)
    except (TypeError, ValueError, NotImplementedError):
        equality = None
    if equality is sp.S.true:
        return True, limit_value
    if equality is sp.S.false:
        return False, limit_value
    try:
        if sp.simplify(limit_value - target) == 0:
            return True, limit_value
    except (TypeError, ValueError, NotImplementedError):
        pass
    return None, limit_value


def _inequality_decision(rel, sol, ctx, assumptions=sp.S.true):
    diff = sp.refine(sp.sympify(rel.lhs - rel.rhs).subs(sol), assumptions)
    if isinstance(rel, sp.StrictGreaterThan):
        direct = _assumption_query(sp.Q.positive(diff), assumptions)
        if direct is True:
            return True
        violating = _assumption_query(sp.Q.nonpositive(diff), assumptions)
        if violating is True:
            return False
    elif isinstance(rel, sp.GreaterThan):
        direct = _assumption_query(sp.Q.nonnegative(diff), assumptions)
        if direct is True:
            return True
        violating = _assumption_query(sp.Q.negative(diff), assumptions)
        if violating is True:
            return False
    elif isinstance(rel, sp.StrictLessThan):
        direct = _assumption_query(sp.Q.negative(diff), assumptions)
        if direct is True:
            return True
        violating = _assumption_query(sp.Q.nonnegative(diff), assumptions)
        if violating is True:
            return False
    elif isinstance(rel, sp.LessThan):
        direct = _assumption_query(sp.Q.nonpositive(diff), assumptions)
        if direct is True:
            return True
        violating = _assumption_query(sp.Q.positive(diff), assumptions)
        if violating is True:
            return False
    sign = ctx.eventual_sign(diff)
    if isinstance(rel, sp.StrictGreaterThan):
        return True if sign == 1 else False if sign in (0, -1) else None
    if isinstance(rel, sp.GreaterThan):
        return True if sign in (0, 1) else False if sign == -1 else None
    if isinstance(rel, sp.StrictLessThan):
        return True if sign == -1 else False if sign in (0, 1) else None
    if isinstance(rel, sp.LessThan):
        return True if sign in (-1, 0) else False if sign == 1 else None
    return None


def _prefer_mrv_hardy(polynomial: sp.Expr, dependent: sp.Symbol, parameter: sp.Symbol) -> bool:
    """Prefer Newton--MRV over exact radicals for transcendental coefficients."""

    try:
        coefficients = sp.Poly(polynomial, dependent).all_coeffs()
    except sp.PolynomialError:
        return False
    for coefficient in coefficients:
        if parameter not in coefficient.free_symbols:
            continue
        try:
            rational = coefficient.is_rational_function(parameter)
        except (TypeError, ValueError, NotImplementedError):
            rational = None
        if rational is not True:
            return True
    return False


def _mrv_hardy_solve_result(
    equation: sp.Expr,
    inequalities: list[sp.Expr],
    dependent: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr,
    terms: int,
    limits: dict[sp.Symbol, sp.Expr] | None,
    domain: sp.Set,
    assumptions: sp.Expr,
    context: AsymptoticContext,
):
    """Return a complete MRV-Hardy solve result when independently counted."""

    if domain not in (sp.S.Reals, sp.S.Complexes):
        return None
    from .hardy_solve import mrv_hardy_polynomial_solve

    try:
        mrv_result = mrv_hardy_polynomial_solve(
            equation,
            dependent,
            parameter,
            point=point,
            terms=terms,
            domain=domain,
            assumptions=assumptions,
            context=context,
        )
    except (ValueError, TypeError, NotImplementedError, sp.PolynomialError):
        return None
    if not mrv_result.complete:
        return None

    branches: list[AsymptoticSolutionBranch] = []
    for hardy_branch in mrv_result.branches:
        value = hardy_branch.expression
        solution = {dependent: value}
        conditions: list[sp.Expr] = []
        rejected = False

        domain_decision = _domain_decision(value, domain, assumptions)
        if domain_decision is False:
            continue
        if domain_decision is None:
            conditions.append(sp.Contains(value, domain, evaluate=False))

        if limits and dependent in limits:
            limit_decision, limit_value = _limit_decision(
                value, limits[dependent], context, assumptions
            )
            if limit_decision is False:
                continue
            if limit_decision is None:
                conditions.append(sp.Eq(limit_value, sp.sympify(limits[dependent]), evaluate=False))

        for relation in inequalities:
            decision = _inequality_decision(relation, solution, context, assumptions)
            if decision is False:
                rejected = True
                break
            if decision is None:
                conditions.append(relation.subs(solution))
        if rejected:
            continue

        branches.append(
            AsymptoticSolutionBranch(
                ((dependent, value),),
                tuple(conditions),
                "EXACT" if hardy_branch.exact else "FORMAL",
                hardy_branch.multiplicity,
                "mrv-hardy-newton",
                hardy_branch,
            )
        )

    status = "EXACT" if all(branch.status == "EXACT" for branch in branches) else "FORMAL"
    return AsymptoticSolveResult(
        tuple(branches),
        parameter,
        point,
        status,
        "mrv-hardy-newton+sturm" if domain == sp.S.Reals else "mrv-hardy-newton",
        mrv_result,
    )


def asymptotic_solve(
    system,
    variables,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    limits: dict[sp.Symbol, sp.Expr] | None = None,
    domain=sp.S.Complexes,
    assumptions: sp.Expr = sp.S.true,
) -> AsymptoticSolveResult:
    """Solve algebraic equations/inequalities asymptotically in ``parameter``.

    For a univariate polynomial with transcendental Hardy/log-exp
    coefficients, a Newton--MRV backend is tried before exact algebraic solving.
    It values coefficient scales, lifts smaller Newton corrections recursively,
    and uses an asymptotic Sturm sequence to certify completeness of real roots.
    Rational/algebraic systems retain the exact-solve route.  A supplied branch
    limit still enables the implicit/Puiseux backend when neither route settles
    the problem. Undecidable predicates remain symbolic branch conditions.
    """
    variables = tuple(variables) if isinstance(variables, (list, tuple, sp.Tuple)) else (variables,)
    equations, inequalities = _split_relations(system)
    ctx = AsymptoticContext(parameter, point=point)
    branches = []
    multiplicities: dict[tuple[tuple[sp.Symbol, sp.Expr], ...], int] = {}

    polynomial_case = (
        len(equations) == 1
        and len(variables) == 1
        and bool(equations[0].is_polynomial(variables[0]))
    )
    if polynomial_case and _prefer_mrv_hardy(equations[0], variables[0], parameter):
        mrv_solution = _mrv_hardy_solve_result(
            equations[0],
            inequalities,
            variables[0],
            parameter,
            point,
            terms,
            limits,
            domain,
            assumptions,
            ctx,
        )
        if mrv_solution is not None:
            return mrv_solution

    try:
        if len(equations) == 1 and len(variables) == 1 and equations[0].is_polynomial(variables[0]):
            y = variables[0]
            _, factors = sp.factor_list(equations[0], y)
            sols = []
            for factor, multiplicity in factors:
                values = bounded_solve_one(factor, y, allow_general=True) or ()
                for value in values:
                    sol = {y: value}
                    key = tuple(sol.items())
                    multiplicities[key] = multiplicities.get(key, 0) + multiplicity
                    sols.append(sol)
        else:
            sols = list(bounded_solve_system(equations, variables, allow_general=True) or ())
    except (ValueError, TypeError, NotImplementedError, sp.PolynomialError):
        sols = []
    if sols:
        seen = set()
        for sol in sols:
            key = tuple(sol.items())
            if key in seen:
                continue
            seen.add(key)
            cond = []
            rejected = False
            for v in variables:
                if v not in sol:
                    continue
                domain_decision = _domain_decision(sol[v], domain, assumptions)
                if domain_decision is False:
                    rejected = True
                    break
                if domain_decision is None:
                    cond.append(sp.Contains(sol[v], domain, evaluate=False))
                if limits and v in limits:
                    limit_decision, limit_value = _limit_decision(
                        sol[v], limits[v], ctx, assumptions
                    )
                    if limit_decision is False:
                        rejected = True
                        break
                    if limit_decision is None:
                        cond.append(sp.Eq(limit_value, sp.sympify(limits[v]), evaluate=False))
            if rejected:
                continue
            for rel in inequalities:
                d = _inequality_decision(rel, sol, ctx, assumptions)
                if d is False:
                    rejected = True
                    break
                if d is None:
                    cond.append(rel.subs(sol))
            if rejected:
                continue
            expanded = {}
            changed = False
            for v in variables:
                if v not in sol:
                    continue
                expanded[v], c = _expand_at_infinity(sol[v], parameter, point, terms)
                changed |= c
            branches.append(
                AsymptoticSolutionBranch(
                    tuple(expanded.items()),
                    tuple(cond),
                    "FORMAL" if changed else "EXACT",
                    multiplicities.get(key, 1),
                )
            )
        status = "EXACT" if not branches or all(b.status == "EXACT" for b in branches) else "FORMAL"
        return AsymptoticSolveResult(
            tuple(branches), parameter, point, status, "exact-algebraic+asymptotic-filter"
        )

    # Single-equation implicit fallback. Infinity is localized by u=1/p.
    if len(equations) == 1 and len(variables) == 1 and limits and variables[0] in limits:
        y = variables[0]
        b = limits[y]
        dep_inverse = b in (sp.oo, -sp.oo)
        local_y = sp.Dummy("z") if dep_inverse else y
        local_b = sp.S.Zero if dep_inverse else b
        y_sub = (1 / local_y if b is sp.oo else -1 / local_y) if dep_inverse else local_y
        if point in (sp.oo, -sp.oo):
            u = sp.Dummy("u", positive=True)
            p_sub = 1 / u if point is sp.oo else -1 / u
            eq = sp.together(equations[0].subs({parameter: p_sub, y: y_sub}))
            eq = sp.fraction(eq)[0]
            found = implicit_asymptotic(
                eq,
                local_y,
                u,
                point=0,
                dependent_limit=local_b,
                terms=terms,
                assumptions=assumptions,
            )
            seq = [] if hasattr(found, "strata") else found
            for br in seq:
                raw = br.series.truncate().subs(
                    u, 1 / parameter if point is sp.oo else -1 / parameter
                )
                value = (1 / raw if b is sp.oo else -1 / raw) if dep_inverse else raw
                sol = {y: value}
                cond = []
                rejected = False
                domain_decision = _domain_decision(value, domain, assumptions)
                if domain_decision is False:
                    rejected = True
                elif domain_decision is None:
                    cond.append(sp.Contains(value, domain, evaluate=False))
                for rel in inequalities:
                    if rejected:
                        break
                    d = _inequality_decision(rel, sol, ctx, assumptions)
                    if d is False:
                        rejected = True
                        break
                    if d is None:
                        cond.append(rel.subs(sol))
                if not rejected:
                    branches.append(
                        AsymptoticSolutionBranch(
                            ((y, value),), tuple(cond), "FORMAL", method="implicit-puiseux"
                        )
                    )
        else:
            eq = sp.together(equations[0].subs(y, y_sub))
            eq = sp.fraction(eq)[0]
            found = implicit_asymptotic(
                eq,
                local_y,
                parameter,
                point=point,
                dependent_limit=local_b,
                terms=terms,
                assumptions=assumptions,
            )
            seq = [] if hasattr(found, "strata") else found
            for br in seq:
                raw = br.series.truncate()
                value = (1 / raw if b is sp.oo else -1 / raw) if dep_inverse else raw
                sol = {y: value}
                cond = []
                rejected = False
                domain_decision = _domain_decision(value, domain, assumptions)
                if domain_decision is False:
                    rejected = True
                elif domain_decision is None:
                    cond.append(sp.Contains(value, domain, evaluate=False))
                for rel in inequalities:
                    if rejected:
                        break
                    d = _inequality_decision(rel, sol, ctx, assumptions)
                    if d is False:
                        rejected = True
                        break
                    if d is None:
                        cond.append(rel.subs(sol))
                if not rejected:
                    branches.append(
                        AsymptoticSolutionBranch(
                            ((y, value),), tuple(cond), "FORMAL", method="implicit-puiseux"
                        )
                    )
        return AsymptoticSolveResult(
            tuple(branches),
            parameter,
            point,
            "FORMAL" if branches else "UNKNOWN",
            "implicit-puiseux",
        )
    return AsymptoticSolveResult((), parameter, point, "UNKNOWN", "unsupported")
