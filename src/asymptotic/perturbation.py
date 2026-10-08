"""Common perturbation hierarchies for asymptotic expansion methods.

This module separates construction of an order-by-order perturbation problem
from policies for solving those orders.  Regular perturbation, strained-time,
multiple-scale, and matched-expansion methods can therefore share coefficient
extraction, condition propagation, reconstruction, and residual checking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import lcm

import sympy as sp
from funcprops import normalize_assumptions
from sympy.core.function import AppliedUndef

from ._symbolic_policy import bounded_limit

EquationLike = sp.Expr | sp.Equality


def _as_residual(equation: EquationLike) -> sp.Expr:
    equation = sp.sympify(equation)
    if isinstance(equation, sp.Equality):
        return sp.expand(equation.lhs - equation.rhs)
    return sp.expand(equation)


def _as_tuple(value: object) -> tuple:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _coefficient_unknown(unknown: sp.Expr, index: int) -> sp.Expr:
    if isinstance(unknown, AppliedUndef):
        name = f"{unknown.func.__name__}_{index}"
        return sp.Function(name)(*unknown.args)
    if isinstance(unknown, sp.Symbol):
        return sp.Symbol(f"{unknown.name}_{index}", **unknown.assumptions0)
    raise TypeError("unknowns must be symbols or applied undefined functions")


def _power_exponent(gauge: sp.Expr, parameter: sp.Symbol) -> sp.Rational | None:
    gauge = sp.sympify(gauge)
    powers = gauge.as_powers_dict()
    exponent = sp.sympify(powers.get(parameter, 0))
    remainder = sp.cancel(gauge / parameter**exponent)
    if parameter in remainder.free_symbols or remainder != 1:
        return None
    if exponent.is_Rational is not True:
        return None
    return sp.Rational(exponent)


def _power_coefficients(
    expression: sp.Expr,
    parameter: sp.Symbol,
    gauges: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...] | None:
    exponents = tuple(_power_exponent(gauge, parameter) for gauge in gauges)
    if any(exponent is None for exponent in exponents):
        return None
    rational_exponents = tuple(sp.Rational(exponent) for exponent in exponents)
    denominator = 1
    for exponent in rational_exponents:
        denominator = lcm(denominator, int(exponent.q))
    eta = sp.Dummy("perturbation_scale", positive=True)
    transformed = sp.expand(expression.subs(parameter, eta**denominator))
    try:
        expanded = sp.series(
            transformed, eta, 0, int(max(rational_exponents) * denominator) + 1
        ).removeO()
    except (NotImplementedError, ValueError, TypeError, sp.PoleError):
        return None
    return tuple(
        sp.expand(expanded).coeff(eta, int(exponent * denominator))
        for exponent in rational_exponents
    )


def _gauge_coefficients(
    expression: sp.Expr,
    parameter: sp.Symbol,
    gauges: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...]:
    power_result = _power_coefficients(expression, parameter, gauges)
    if power_result is not None:
        return power_result

    remainder = sp.sympify(expression)
    coefficients: list[sp.Expr] = []
    for gauge in gauges:
        coefficient = bounded_limit(remainder / gauge, parameter, 0, direction="+")
        if coefficient is None:
            coefficient = sp.Limit(remainder / gauge, parameter, 0, dir="+")
        else:
            coefficient = sp.simplify(coefficient)
        coefficients.append(coefficient)
        remainder = sp.expand(remainder - coefficient * gauge)
    return tuple(coefficients)


def _substitute_unknowns(
    expression: sp.Expr,
    unknowns: tuple[sp.Expr, ...],
    replacements: tuple[sp.Expr, ...],
) -> sp.Expr:
    result = sp.sympify(expression)
    symbol_map: dict[sp.Expr, sp.Expr] = {}
    for unknown, replacement in zip(unknowns, replacements):
        if isinstance(unknown, sp.Symbol):
            symbol_map[unknown] = replacement
            continue
        if isinstance(unknown, AppliedUndef):
            variables = unknown.args

            function = unknown.func

            def matches(
                node: sp.Basic, *, function=function, variables=variables
            ) -> bool:
                return (
                    isinstance(node, AppliedUndef)
                    and node.func == function
                    and len(node.args) == len(variables)
                )

            def replace_node(
                node: AppliedUndef,
                *,
                replacement=replacement,
                variables=variables,
            ) -> sp.Expr:
                return replacement.xreplace(dict(zip(variables, node.args)))

            result = result.replace(matches, replace_node)
    if symbol_map:
        result = result.subs(symbol_map, simultaneous=True)
    return sp.expand(result.doit())


def _validate_gauges(parameter: sp.Symbol, gauges: tuple[sp.Expr, ...]) -> None:
    if not gauges:
        raise ValueError("at least one perturbation gauge is required")
    if any(gauge == 0 for gauge in gauges):
        raise ValueError("perturbation gauges must be nonzero")
    if sp.simplify(gauges[0] - 1) != 0:
        raise ValueError("the leading perturbation gauge must be 1")

    for earlier, later in pairwise(gauges):
        ratio = bounded_limit(later / earlier, parameter, 0, direction="+")
        if ratio is None:
            continue
        if ratio not in (0, sp.S.Zero):
            raise ValueError(
                "perturbation gauges must be ordered from dominant to smaller"
            )


@dataclass(frozen=True)
class SolvabilityCondition:
    """Condition required for consistency of a perturbation order."""

    expression: sp.Expr
    source_order: int
    reason: str = ""


@dataclass(frozen=True)
class PerturbationOrder:
    """One coefficient problem in a perturbation hierarchy."""

    index: int
    gauge: sp.Expr
    equations: tuple[sp.Expr, ...]
    conditions: tuple[sp.Expr, ...]
    unknowns: tuple[sp.Expr, ...]
    solution: tuple[tuple[sp.Expr, sp.Expr], ...] = ()
    solvability_conditions: tuple[SolvabilityCondition, ...] = ()

    @property
    def solved(self) -> bool:
        """Whether values have been supplied for all unknowns at this order."""
        solved = {unknown for unknown, _ in self.solution}
        return all(unknown in solved for unknown in self.unknowns)

    def solution_dict(self) -> dict[sp.Expr, sp.Expr]:
        """Return this order's supplied solution as a substitution mapping."""
        return dict(self.solution)


@dataclass(frozen=True)
class PerturbationHierarchy:
    """Order-by-order representation of a perturbation problem."""

    parameter: sp.Symbol
    unknowns: tuple[sp.Expr, ...]
    gauges: tuple[sp.Expr, ...]
    equations: tuple[sp.Expr, ...]
    conditions: tuple[sp.Expr, ...]
    assumptions: sp.Expr
    coefficient_unknowns: tuple[tuple[sp.Expr, ...], ...]
    orders: tuple[PerturbationOrder, ...]

    def order(self, index: int) -> PerturbationOrder:
        """Return one perturbation order by zero-based index."""
        if not 0 <= index < len(self.orders):
            raise IndexError("perturbation order is outside the hierarchy")
        return self.orders[index]

    @property
    def solved_through(self) -> int | None:
        """Largest consecutively solved order, or ``None`` if order zero is open."""
        solved = -1
        for order in self.orders:
            if not order.solved:
                break
            solved = order.index
        return None if solved < 0 else solved

    def substitutions(self, through: int | None = None) -> dict[sp.Expr, sp.Expr]:
        """Combine supplied order solutions into one substitution mapping."""
        limit = len(self.orders) - 1 if through is None else through
        result: dict[sp.Expr, sp.Expr] = {}
        for order in self.orders:
            if order.index > limit:
                break
            result.update(order.solution_dict())
        return result

    def approximation(self, through: int | None = None) -> tuple[sp.Expr, ...]:
        """Reconstruct the truncated dependent variables from supplied solutions."""
        limit = len(self.gauges) - 1 if through is None else through
        if not 0 <= limit < len(self.gauges):
            raise IndexError("perturbation order is outside the hierarchy")
        supplied = self.substitutions(limit)
        approximations = []
        for components in self.coefficient_unknowns:
            expression = sp.Add(
                *(self.gauges[index] * components[index] for index in range(limit + 1))
            )
            approximations.append(sp.expand(expression.subs(supplied)))
        return tuple(approximations)

    def residual(self, through: int | None = None) -> tuple[sp.Expr, ...]:
        """Substitute the reconstructed approximation into the original equations."""
        approximation = self.approximation(through)
        return tuple(
            _substitute_unknowns(equation, self.unknowns, approximation)
            for equation in self.equations
        )

    def condition_residual(self, through: int | None = None) -> tuple[sp.Expr, ...]:
        """Substitute the reconstructed approximation into the original conditions."""
        approximation = self.approximation(through)
        return tuple(
            _substitute_unknowns(condition, self.unknowns, approximation)
            for condition in self.conditions
        )

    def with_solvability_conditions(
        self,
        index: int,
        conditions: sp.Expr | Sequence[sp.Expr],
        *,
        reason: str = "",
    ) -> PerturbationHierarchy:
        """Return a hierarchy with consistency conditions attached to one order."""
        order = self.order(index)
        expressions = tuple(map(sp.sympify, _as_tuple(conditions)))
        added = tuple(
            SolvabilityCondition(expression, index, reason)
            for expression in expressions
        )
        updated_order = replace(
            order,
            solvability_conditions=order.solvability_conditions + added,
        )
        updated_orders = list(self.orders)
        updated_orders[index] = updated_order
        return replace(self, orders=tuple(updated_orders))

    def with_solution(
        self,
        index: int,
        solution: dict[sp.Expr, sp.Expr] | Sequence[sp.Expr] | sp.Expr,
    ) -> PerturbationHierarchy:
        """Return a hierarchy with a user-supplied solution recorded at one order."""
        order = self.order(index)
        if isinstance(solution, dict):
            extra = tuple(
                (sp.sympify(key), sp.sympify(value)) for key, value in solution.items()
            )
            unknown_set = set(order.unknowns)
            if any(key not in unknown_set for key, _ in extra):
                raise ValueError(
                    "solution contains an unknown from a different perturbation order"
                )
        else:
            values = _as_tuple(solution)
            if len(values) != len(order.unknowns):
                raise ValueError(
                    "solution must provide one value for each unknown at this order"
                )
            extra = tuple(zip(order.unknowns, map(sp.sympify, values)))
        merged = order.solution_dict()
        merged.update(extra)
        updated_order = replace(
            order,
            solution=tuple(
                (unknown, merged[unknown])
                for unknown in order.unknowns
                if unknown in merged
            ),
        )
        updated_orders = list(self.orders)
        updated_orders[index] = updated_order
        return replace(self, orders=tuple(updated_orders))


def perturbation_hierarchy(
    equations: EquationLike | Sequence[EquationLike],
    unknowns: sp.Expr | Sequence[sp.Expr],
    parameter: sp.Symbol,
    *,
    order: int | None = None,
    gauges: Sequence[sp.Expr] | None = None,
    conditions: EquationLike | Sequence[EquationLike] = (),
    assumptions: sp.Expr = sp.S.true,
) -> PerturbationHierarchy:
    """Construct a solver-independent perturbation hierarchy.

    Parameters
    ----------
    equations
        Scalar equation or system, written either as residual expressions equal
        to zero or as :class:`sympy.Equality` objects.
    unknowns
        Symbols or applied undefined functions to expand.
    parameter
        Small positive perturbation parameter tending to zero.
    order
        Highest integer power used by the default gauge sequence
        ``1, parameter, ..., parameter**order``.
    gauges
        Explicit ordered gauge sequence.  This may contain fractional powers or
        non-power scales; the first gauge must be one.
    conditions
        Initial, boundary, normalization, or other equations propagated through
        the same coefficient extraction mechanism.
    assumptions
        Symbolic assumptions retained with the hierarchy for method-specific
        solvers and solvability checks.

    The returned hierarchy contains equations and conditions at every requested
    order but does not solve them.  Solutions may be supplied with
    :meth:`PerturbationHierarchy.with_solution` and reconstructed with
    :meth:`PerturbationHierarchy.approximation`.
    """
    parameter = sp.sympify(parameter)
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a Symbol")
    if parameter.is_positive is False:
        raise ValueError(
            "the perturbation parameter must approach zero from the positive side"
        )

    unknown_tuple = tuple(map(sp.sympify, _as_tuple(unknowns)))
    if not unknown_tuple:
        raise ValueError("at least one perturbation unknown is required")
    if len(set(unknown_tuple)) != len(unknown_tuple):
        raise ValueError("perturbation unknowns must be distinct")
    if any(parameter in unknown.free_symbols for unknown in unknown_tuple):
        raise ValueError(
            "perturbation unknowns must not themselves depend on the small parameter"
        )

    if gauges is None:
        if order is None:
            order = 1
        if order < 0:
            raise ValueError("order must be nonnegative")
        gauge_tuple = tuple(parameter**index for index in range(order + 1))
    else:
        if order is not None:
            raise ValueError("specify either order or gauges, not both")
        gauge_tuple = tuple(map(sp.sympify, gauges))
    _validate_gauges(parameter, gauge_tuple)

    equation_tuple = tuple(_as_residual(equation) for equation in _as_tuple(equations))
    condition_items = () if conditions == () else _as_tuple(conditions)
    condition_tuple = tuple(_as_residual(condition) for condition in condition_items)

    coefficient_unknowns = tuple(
        tuple(_coefficient_unknown(unknown, index) for index in range(len(gauge_tuple)))
        for unknown in unknown_tuple
    )
    ansatz = tuple(
        sp.Add(
            *(
                gauge_tuple[index] * components[index]
                for index in range(len(gauge_tuple))
            )
        )
        for components in coefficient_unknowns
    )
    expanded_equations = tuple(
        _substitute_unknowns(equation, unknown_tuple, ansatz)
        for equation in equation_tuple
    )
    expanded_conditions = tuple(
        _substitute_unknowns(condition, unknown_tuple, ansatz)
        for condition in condition_tuple
    )
    equation_coefficients = tuple(
        _gauge_coefficients(equation, parameter, gauge_tuple)
        for equation in expanded_equations
    )
    condition_coefficients = tuple(
        _gauge_coefficients(condition, parameter, gauge_tuple)
        for condition in expanded_conditions
    )

    orders = tuple(
        PerturbationOrder(
            index=index,
            gauge=gauge,
            equations=tuple(
                coefficients[index] for coefficients in equation_coefficients
            ),
            conditions=tuple(
                coefficients[index] for coefficients in condition_coefficients
            ),
            unknowns=tuple(components[index] for components in coefficient_unknowns),
        )
        for index, gauge in enumerate(gauge_tuple)
    )
    return PerturbationHierarchy(
        parameter=parameter,
        unknowns=unknown_tuple,
        gauges=gauge_tuple,
        equations=equation_tuple,
        conditions=condition_tuple,
        assumptions=normalize_assumptions(assumptions),
        coefficient_unknowns=coefficient_unknowns,
        orders=orders,
    )


__all__ = [
    "PerturbationHierarchy",
    "PerturbationOrder",
    "SolvabilityCondition",
    "perturbation_hierarchy",
]
