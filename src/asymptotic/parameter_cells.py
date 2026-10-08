"""Exact semialgebraic parameter-cell refinement utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp


@dataclass(frozen=True)
class ParameterCell:
    """One exact Boolean parameter cell with a fixed predicate signature."""

    condition: sp.Expr
    signature: tuple[bool, ...]


def _canonical_condition(condition, parameters):
    condition = sp.sympify(condition)
    if len(parameters) == 1:
        try:
            return condition.as_set().as_relational(parameters[0])
        except (AttributeError, NotImplementedError, ValueError):
            pass
    return sp.simplify_logic(condition)


def satisfiable_parameter_condition(condition, parameters):
    """Return True/False when exact real satisfiability can be certified."""
    condition = sp.sympify(condition)
    parameters = tuple(parameters)
    if condition is sp.S.false:
        return False
    if condition is sp.S.true:
        return True
    if len(parameters) == 1:
        p = parameters[0]
        try:
            solution = condition.as_set()
            if solution is sp.S.EmptySet:
                return False
            if isinstance(solution, sp.ConditionSet):
                return None
            if getattr(solution, "is_empty", None) is False:
                return True
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            try:
                parts = (
                    list(condition.args)
                    if isinstance(condition, sp.And)
                    else [condition]
                )
                reduced = sp.reduce_inequalities(parts, p)
                return reduced is not sp.S.false
            except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
                pass
    try:
        from semialg import is_satisfiable

        result = is_satisfiable(condition, parameters)
        return bool(getattr(result, "satisfiable", result))
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        return None


def parameter_truth_cells(predicates, parameters, base_condition=sp.S.true):
    """Partition parameter space by exact truth values of semialgebraic predicates.

    This is the shared cell engine for angular critical-point membership and
    projective pole/critical topology.  Unsatisfiable truth signatures are
    discarded only when exact real arithmetic proves them empty.
    """
    predicates = tuple(map(sp.sympify, predicates))
    parameters = tuple(parameters)
    base_condition = sp.sympify(base_condition)
    if not predicates:
        return (ParameterCell(_canonical_condition(base_condition, parameters), ()),)
    cells = []
    for signature in product((False, True), repeat=len(predicates)):
        literals = tuple(
            predicate if truth else sp.Not(predicate)
            for truth, predicate in zip(signature, predicates, strict=True)
        )
        condition = sp.And(base_condition, *literals)
        sat = satisfiable_parameter_condition(condition, parameters)
        if sat is False:
            continue
        if sat is None:
            # Completeness requires knowing whether every enumerated cell is
            # genuinely realizable; refuse rather than retaining ghost cells.
            return ()
        cells.append(
            ParameterCell(_canonical_condition(condition, parameters), signature)
        )
    return tuple(cells)


def root_interval_membership(root, coordinate, lower, upper, parameters):
    """Eliminate an already-solved root's membership in a closed interval."""
    root = sp.sympify(root)
    parameters = tuple(parameters)
    relation = sp.And(sp.Ge(root, lower), sp.Le(root, upper))
    if len(parameters) == 1:
        try:
            parts = list(relation.args) if isinstance(relation, sp.And) else [relation]
            reduced = sp.reduce_inequalities(parts, parameters[0])
            return _canonical_condition(reduced, parameters)
        except (NotImplementedError, TypeError, ValueError, sp.PolynomialError):
            pass
    # A solved algebraic root leaves only parameter inequalities.  semialg can
    # decide their satisfiability during truth-cell construction; no sampling.
    if not (relation.free_symbols - set(parameters)):
        return relation
    return None


__all__ = [
    "ParameterCell",
    "parameter_truth_cells",
    "root_interval_membership",
    "satisfiable_parameter_condition",
]
