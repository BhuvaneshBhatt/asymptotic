"""Bounded finite parameter partitions used by conditional asymptotics."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ParameterPartition:
    parameter: sp.Expr
    threshold: sp.Expr
    conditions: tuple[sp.Expr, ...]


def critical_threshold_partition(
    parameter: sp.Expr,
    threshold: sp.Expr = sp.S.Zero,
    *,
    branch_budget: int = 3,
    include_equality: bool = True,
) -> ParameterPartition | None:
    """Return the exhaustive real partition around one critical threshold.

    Refuses a partition rather than exceeding ``branch_budget``.
    The threshold may itself contain parameters (for example ``a=b``).
    """
    parameter, threshold = map(sp.sympify, (parameter, threshold))
    conditions = (
        (
            sp.Lt(parameter, threshold),
            sp.Eq(parameter, threshold),
            sp.Gt(parameter, threshold),
        )
        if include_equality
        else (sp.Ne(parameter, threshold), sp.Eq(parameter, threshold))
    )
    if len(conditions) > branch_budget:
        return None
    return ParameterPartition(parameter, threshold, conditions)


def comparison_threshold_partition(
    lhs: sp.Expr, rhs: sp.Expr, *, branch_budget: int = 3
) -> ParameterPartition | None:
    """Partition a comparison into ``lhs<rhs``, equality and ``lhs>rhs``."""
    return critical_threshold_partition(lhs, rhs, branch_budget=branch_budget)
