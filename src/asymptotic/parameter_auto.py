"""Automatic parameter case discovery for asymptotic algorithms."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

import sympy as sp

from ._symbolic_policy import bounded_solve_one
from .canonical import canonical_key
from .function_properties.semantics import PropertyProvenance, entails
from .stratification import AsymptoticStratification, evaluate_parameter_strata

T = TypeVar("T")


def parameter_symbols(expr: sp.Expr, excluded: Iterable[sp.Symbol] = ()) -> tuple[sp.Symbol, ...]:
    excluded_set = set(excluded)
    return tuple(sorted(sp.sympify(expr).free_symbols - excluded_set, key=sp.default_sort_key))


def _nonconstant_parameter_factors(
    expr: sp.Expr, parameters: set[sp.Symbol]
) -> tuple[sp.Expr, ...]:
    expr = sp.factor(sp.sympify(expr))
    if not (expr.free_symbols & parameters):
        return ()
    try:
        _constant, factors = sp.factor_list(expr)
    except (sp.PolynomialError, TypeError, ValueError):
        factors = ()
    out = []
    if factors:
        for factor, _power in factors:
            factor = sp.factor(factor)
            if factor.free_symbols and factor.free_symbols <= parameters:
                out.append(factor)
    elif expr.free_symbols <= parameters:
        out.append(expr)
    return tuple(out)


def critical_parameter_expressions(
    expressions: Iterable[sp.Expr],
    *,
    parameters: Iterable[sp.Symbol],
    assumptions: sp.Expr | bool = sp.S.true,
    max_splits: int = 6,
) -> tuple[sp.Expr, ...]:
    """Return unresolved parameter factors whose vanishing changes structure."""

    params = set(parameters)
    base = sp.sympify(assumptions)
    found = []
    found_keys = set()
    for expression in expressions:
        expression = sp.sympify(expression)
        try:
            numerator, denominator = sp.fraction(sp.cancel(expression))
        except (sp.PolynomialError, TypeError, ValueError, AttributeError):
            numerator, denominator = expression, sp.S.One
        for component in (numerator, denominator):
            for factor in _nonconstant_parameter_factors(component, params):
                if entails(sp.Eq(factor, 0), base) is not None:
                    continue
                if entails(sp.Ne(factor, 0), base) is not None:
                    continue
                key = canonical_key(factor)
                if key in found_keys:
                    continue
                found.append(factor)
                found_keys.add(key)
                if len(found) >= max_splits:
                    return tuple(found)
    return tuple(found)


def _equality_substitutions(
    assumptions: sp.Expr | bool, parameters: Iterable[sp.Symbol]
) -> dict[sp.Symbol, sp.Expr]:
    params = tuple(parameters)
    substitutions = {}
    for clause in sp.And.make_args(sp.sympify(assumptions)):
        if not isinstance(clause, sp.Equality):
            continue
        lhs = sp.sympify(clause.lhs).subs(substitutions)
        rhs = sp.sympify(clause.rhs).subs(substitutions)
        if isinstance(lhs, sp.Symbol) and lhs in params and lhs not in rhs.free_symbols:
            substitutions[lhs] = rhs
            continue
        if isinstance(rhs, sp.Symbol) and rhs in params and rhs not in lhs.free_symbols:
            substitutions[rhs] = lhs
            continue
        equation = sp.expand(lhs - rhs)
        candidates = [p for p in params if p in equation.free_symbols and p not in substitutions]
        for parameter in candidates:
            # Most generated strata are affine equalities.  Solve those
            # directly instead of invoking SymPy's general equation solver.
            coefficient = sp.expand(equation).coeff(parameter)
            remainder = sp.expand(equation - coefficient * parameter)
            if (
                coefficient != 0
                and parameter not in coefficient.free_symbols
                and parameter not in remainder.free_symbols
            ):
                solution = sp.cancel(-remainder / coefficient)
                substitutions[parameter] = solution.subs(substitutions)
                break
            # Retain a conservative fallback only for small nonlinear clauses.
            if sp.count_ops(equation) > 20:
                continue
            solutions = bounded_solve_one(equation, parameter) or ()
            if len(solutions) == 1 and parameter not in sp.sympify(solutions[0]).free_symbols:
                substitutions[parameter] = sp.sympify(solutions[0]).subs(substitutions)
                break
    return substitutions


def specialize_expression(
    expr: sp.Expr,
    assumptions: sp.Expr | bool,
    *,
    parameters: Iterable[sp.Symbol] | None = None,
) -> sp.Expr:
    """Specialize an expression by exact equality assumptions when possible."""

    expr = sp.sympify(expr)
    params = (
        tuple(parameters)
        if parameters is not None
        else tuple(sorted(expr.free_symbols, key=sp.default_sort_key))
    )
    substitutions = _equality_substitutions(assumptions, params)
    specialized = expr.subs(substitutions, simultaneous=False)
    try:
        specialized = sp.refine(specialized, sp.sympify(assumptions).subs(substitutions))
    except (TypeError, ValueError):
        pass
    return sp.factor(sp.simplify(specialized))


def automatic_parameter_stratification(
    expressions: Iterable[sp.Expr],
    evaluator: Callable[[sp.Expr], T],
    *,
    parameters: Iterable[sp.Symbol],
    assumptions: sp.Expr | bool = sp.S.true,
    max_splits: int = 6,
    provenance_source: str = "asymptotic.automatic_parameter_stratification",
) -> AsymptoticStratification[T] | None:
    """Automatically split on unresolved zero/nonzero structural coefficients."""

    params = tuple(parameters)
    critical = critical_parameter_expressions(
        expressions,
        parameters=params,
        assumptions=assumptions,
        max_splits=max_splits,
    )
    if not critical:
        return None

    conditions: list[sp.Expr] = [sp.S.true]
    for expression in critical:
        next_conditions = []
        for condition in conditions:
            next_conditions.append(sp.And(condition, sp.Eq(expression, 0)))
            next_conditions.append(sp.And(condition, sp.Ne(expression, 0)))
        conditions = next_conditions

    return evaluate_parameter_strata(
        conditions,
        evaluator,
        assumptions=assumptions,
        parameters=params,
        require_exhaustive=True,
        provenance=(
            PropertyProvenance(
                provenance_source,
                note="automatic zero/nonzero split of structure-changing coefficients",
            ),
        ),
    )
