"""Multiple-scale expansions for weakly perturbed oscillators.

The derivative calculus in this module is independent of the oscillator
solver.  It represents the total derivative as a weighted sum of partial
scale derivatives and expands arbitrary derivative orders algebraically.  The
high-level solver uses the common perturbation hierarchy and periodic Fredholm
projections to derive slow modulation equations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

from ._linear_ode_operator import linear_operator_coefficients
from ._symbolic_policy import bounded_simplify, bounded_solve_system
from .perturbation import EquationLike, PerturbationHierarchy, perturbation_hierarchy
from .resonance import periodic_solvability_conditions


def _as_tuple(value: object) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _residual(equation: EquationLike) -> sp.Expr:
    equation = sp.sympify(equation)
    if isinstance(equation, sp.Equality):
        return sp.expand(equation.lhs - equation.rhs)
    return sp.expand(equation)


def multiple_scale_derivative(
    expression: sp.Expr,
    scales: Sequence[sp.Symbol],
    parameter: sp.Symbol,
    *,
    derivative_order: int = 1,
    scale_orders: Sequence[int] | None = None,
) -> sp.Expr:
    """Expand a physical derivative in independent multiple-scale variables.

    With default scale orders ``(0, 1, ..., n-1)``, this applies
    ``D = D_0 + eps*D_1 + ...`` repeatedly.  Arbitrary nonnegative integer
    scale orders are accepted, so constructions such as ``T2 = eps**3*t`` do
    not require a separate derivative implementation.
    """
    parameter = sp.sympify(parameter)
    scale_tuple = tuple(scales)
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a Symbol")
    if not scale_tuple or any(
        not isinstance(scale, sp.Symbol) for scale in scale_tuple
    ):
        raise TypeError("scales must be a nonempty sequence of Symbols")
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    orders = (
        tuple(range(len(scale_tuple))) if scale_orders is None else tuple(scale_orders)
    )
    if len(orders) != len(scale_tuple) or any(
        not isinstance(item, int) or item < 0 for item in orders
    ):
        raise ValueError("scale_orders must contain one nonnegative integer per scale")
    if len(set(orders)) != len(orders):
        raise ValueError("scale orders must be distinct")

    result = sp.sympify(expression)
    for _ in range(derivative_order):
        result = sp.Add(
            *(
                parameter**power * sp.diff(result, scale)
                for scale, power in zip(scale_tuple, orders)
            )
        )
        result = sp.expand(result)
    return result


def _transform_equation(
    equation: sp.Expr,
    unknowns: tuple[AppliedUndef, ...],
    variable: sp.Symbol,
    profiles: tuple[AppliedUndef, ...],
    scales: tuple[sp.Symbol, ...],
    parameter: sp.Symbol,
    scale_orders: tuple[int, ...],
) -> sp.Expr:
    result = sp.sympify(equation)
    replacements: dict[sp.Expr, sp.Expr] = {}
    for unknown, profile in zip(unknowns, profiles):
        derivatives = sorted(
            (
                derivative
                for derivative in result.atoms(sp.Derivative)
                if derivative.expr == unknown
                and all(item == variable for item in derivative.variables)
            ),
            key=lambda item: item.derivative_count,
            reverse=True,
        )
        replacements.update(
            {
                derivative: multiple_scale_derivative(
                    profile,
                    scales,
                    parameter,
                    derivative_order=derivative.derivative_count,
                    scale_orders=scale_orders,
                )
                for derivative in derivatives
            }
        )
        replacements[unknown] = profile
    return sp.expand(result.xreplace(replacements))


def _transform_near_resonant_phases(
    expression: sp.Expr,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    fast: sp.Symbol,
    slow: sp.Symbol,
) -> sp.Expr:
    result = expression
    replacements: dict[sp.Expr, sp.Expr] = {}
    for trig in result.atoms(sp.sin, sp.cos):
        argument = sp.expand(trig.args[0])
        coefficient = bounded_simplify(sp.diff(argument, variable))
        if bounded_simplify(argument - coefficient * variable) != 0:
            continue
        if variable in coefficient.free_symbols:
            continue
        base = bounded_simplify(coefficient.subs(parameter, 0))
        detuning = bounded_simplify(sp.diff(coefficient, parameter).subs(parameter, 0))
        remainder = bounded_simplify(coefficient - base - parameter * detuning)
        if remainder != 0:
            continue
        replacements[trig] = trig.func(base * fast + detuning * slow)
    if replacements:
        result = result.xreplace(replacements)
    return sp.expand(result)


def _base_frequency(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    parameter: sp.Symbol,
) -> sp.Expr | None:
    base = sp.expand(equation.subs(parameter, 0))
    data = linear_operator_coefficients(base, unknown, variable)
    if data is None:
        return None
    coefficients, degree = data
    if degree != 2:
        return None
    a0, a1, a2 = coefficients
    if a2 == 0 or bounded_simplify(a1) != 0:
        return None
    if variable in a0.free_symbols | a2.free_symbols:
        return None
    ratio = bounded_simplify(a0 / a2)
    if ratio.is_zero is True or ratio.is_negative is True:
        return None
    return bounded_simplify(sp.sqrt(ratio))


def _zero_current(expression: sp.Expr, unknown: AppliedUndef) -> sp.Expr:
    replacements: dict[sp.Expr, sp.Expr] = {unknown: sp.S.Zero}
    replacements.update(
        {
            derivative: sp.S.Zero
            for derivative in expression.atoms(sp.Derivative)
            if derivative.expr == unknown
        }
    )
    return sp.expand(expression.xreplace(replacements))


def _leading_profile(
    coefficient: AppliedUndef,
    fast: sp.Symbol,
    slow: sp.Symbol,
    frequency: sp.Expr,
    index: int,
) -> tuple[sp.Expr, sp.FunctionClass, sp.FunctionClass]:
    amp_cos = sp.Function(f"A_{index}")
    amp_sin = sp.Function(f"B_{index}")
    expression = amp_cos(slow) * sp.cos(frequency * fast) + amp_sin(slow) * sp.sin(
        frequency * fast
    )
    return expression, amp_cos, amp_sin


@dataclass(frozen=True)
class SlowFlowEquation:
    """One first-order modulation equation obtained from solvability."""

    derivative: sp.Derivative
    expression: sp.Expr
    source_order: int

    @property
    def equation(self) -> sp.Equality:
        """Return the slow-flow relation as an equality."""
        return sp.Eq(self.derivative, self.expression)


@dataclass(frozen=True)
class MultipleScalesResult:
    """Multiple-scale hierarchy together with derived slow modulation flow."""

    hierarchy: PerturbationHierarchy
    original_unknowns: tuple[AppliedUndef, ...]
    original_variable: sp.Symbol
    scales: tuple[sp.Symbol, ...]
    scale_orders: tuple[int, ...]
    frequencies: tuple[sp.Expr, ...]
    slow_flow: tuple[SlowFlowEquation, ...]
    complete: bool
    unresolved_order: int | None = None
    limitation: str = ""

    @property
    def profiles(self) -> tuple[sp.Expr, ...]:
        """Return reconstructed multiple-scale profiles."""
        return self.hierarchy.approximation()

    @property
    def modulation_equations(self) -> tuple[sp.Equality, ...]:
        """Return the derived slow-flow equations."""
        return tuple(item.equation for item in self.slow_flow)


def multiple_scales(
    equations: EquationLike | Sequence[EquationLike],
    unknowns: sp.Expr | Sequence[sp.Expr],
    parameter: sp.Symbol,
    *,
    order: int = 1,
    scales: Sequence[sp.Symbol] | None = None,
    scale_orders: Sequence[int] | None = None,
    base_frequencies: Sequence[sp.Expr] | sp.Expr | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> MultipleScalesResult:
    """Derive a multiple-scale expansion and first slow modulation equations.

    The automatic solver presently targets autonomous weak perturbations of
    uncoupled linear oscillators at leading order.  Nonlinear, damping, and
    coupling terms may enter at higher perturbation orders.  The leading
    solution retains arbitrary cosine/sine amplitudes on the first slow scale,
    and Fredholm projections determine their slow derivatives.

    Problems outside that contract return an explicit solved prefix or raise a
    structural ``NotImplementedError`` rather than introducing secular terms.
    """
    parameter = sp.sympify(parameter)
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a Symbol")
    if order < 1:
        raise ValueError("multiple scales requires order >= 1")
    unknown_tuple = tuple(map(sp.sympify, _as_tuple(unknowns)))
    if not unknown_tuple or any(
        not isinstance(item, AppliedUndef) or len(item.args) != 1
        for item in unknown_tuple
    ):
        raise TypeError("unknowns must be applied scalar functions of one variable")
    variable = unknown_tuple[0].args[0]
    if any(item.args[0] != variable for item in unknown_tuple) or not isinstance(
        variable, sp.Symbol
    ):
        raise ValueError("all unknowns must share one symbolic independent variable")
    equation_tuple = tuple(_residual(item) for item in _as_tuple(equations))
    if len(equation_tuple) != len(unknown_tuple):
        raise ValueError("multiple scales requires one governing equation per unknown")

    if scales is None:
        scale_tuple = tuple(
            sp.Symbol(f"T{index}", real=True) for index in range(order + 1)
        )
    else:
        scale_tuple = tuple(scales)
    orders = (
        tuple(range(len(scale_tuple))) if scale_orders is None else tuple(scale_orders)
    )
    if len(scale_tuple) < 2:
        raise ValueError(
            "multiple scales requires at least one fast and one slow scale"
        )
    if len(orders) != len(scale_tuple) or orders[0] != 0 or 1 not in orders:
        raise ValueError(
            "automatic multiple scales requires scale orders containing 0 and 1"
        )
    fast = scale_tuple[orders.index(0)]
    slow = scale_tuple[orders.index(1)]

    profiles = tuple(
        sp.Function(f"{item.func.__name__}_ms")(*scale_tuple) for item in unknown_tuple
    )
    transformed = tuple(
        _transform_near_resonant_phases(
            _transform_equation(
                eq, unknown_tuple, variable, profiles, scale_tuple, parameter, orders
            ),
            variable,
            parameter,
            fast,
            slow,
        )
        for eq in equation_tuple
    )
    if any(variable in expression.free_symbols for expression in transformed):
        raise NotImplementedError(
            "automatic multiple scales requires autonomous equations; use the derivative calculus for explicit-time forcing"
        )
    hierarchy = perturbation_hierarchy(
        transformed,
        profiles,
        parameter,
        order=order,
        assumptions=assumptions,
    )

    if base_frequencies is None:
        frequencies = tuple(
            _base_frequency(eq, unknown, variable, parameter)
            for eq, unknown in zip(equation_tuple, unknown_tuple)
        )
    else:
        supplied = _as_tuple(base_frequencies)
        if len(supplied) == 1 and len(unknown_tuple) > 1:
            supplied = supplied * len(unknown_tuple)
        if len(supplied) != len(unknown_tuple):
            raise ValueError("base_frequencies must provide one value per unknown")
        frequencies = tuple(map(sp.sympify, supplied))
    if any(value is None for value in frequencies):
        raise NotImplementedError("could not infer every leading oscillator frequency")
    frequencies = tuple(sp.sympify(value) for value in frequencies)

    leading_values: list[sp.Expr] = []
    amplitude_functions: list[tuple[sp.FunctionClass, sp.FunctionClass]] = []
    for index, (components, frequency) in enumerate(
        zip(hierarchy.coefficient_unknowns, frequencies)
    ):
        expression, amp_cos, amp_sin = _leading_profile(
            components[0], fast, slow, frequency, index
        )
        leading_values.append(expression)
        amplitude_functions.append((amp_cos, amp_sin))
    hierarchy = hierarchy.with_solution(0, tuple(leading_values))

    slow_flow: list[SlowFlowEquation] = []
    if len(hierarchy.orders) > 1:
        order_data = hierarchy.order(1)
        previous = hierarchy.substitutions(0)
        projection_conditions: list[sp.Expr] = []
        derivative_unknowns: list[sp.Derivative] = []
        for index, (equation_i, component_i, frequency) in enumerate(
            zip(
                order_data.equations,
                (items[1] for items in hierarchy.coefficient_unknowns),
                frequencies,
            )
        ):
            reduced = sp.expand(equation_i.subs(previous).doit())
            forcing = _zero_current(reduced, component_i)
            period = bounded_simplify(2 * sp.pi / frequency)
            projection = periodic_solvability_conditions(
                forcing,
                fast,
                (sp.cos(frequency * fast), sp.sin(frequency * fast)),
                period=period,
                source_order=1,
            )
            projection_conditions.extend(projection.projections)
            amp_cos, amp_sin = amplitude_functions[index]
            derivative_unknowns.extend(
                (sp.diff(amp_cos(slow), slow), sp.diff(amp_sin(slow), slow))
            )

        hierarchy = hierarchy.with_solvability_conditions(
            1,
            tuple(projection_conditions),
            reason="multiple-scale periodic Fredholm solvability",
        )
        nonzero = tuple(
            bounded_simplify(item)
            for item in projection_conditions
            if bounded_simplify(item) != 0
        )
        if nonzero:
            solutions = bounded_solve_system(
                nonzero, tuple(derivative_unknowns), allow_general=True
            )
            if solutions is None or len(solutions) != 1:
                return MultipleScalesResult(
                    hierarchy,
                    unknown_tuple,
                    variable,
                    scale_tuple,
                    orders,
                    frequencies,
                    (),
                    False,
                    1,
                    "periodic solvability did not determine a unique first slow flow",
                )
            solution = solutions[0]
            for derivative in derivative_unknowns:
                if derivative not in solution:
                    return MultipleScalesResult(
                        hierarchy,
                        unknown_tuple,
                        variable,
                        scale_tuple,
                        orders,
                        frequencies,
                        (),
                        False,
                        1,
                        "periodic solvability left part of the first slow flow unresolved",
                    )
                slow_flow.append(
                    SlowFlowEquation(
                        derivative, bounded_simplify(solution[derivative]), 1
                    )
                )

    complete = order == 1
    limitation = (
        ""
        if complete
        else "higher profile corrections are retained symbolically after the first slow-flow derivation"
    )
    return MultipleScalesResult(
        hierarchy=hierarchy,
        original_unknowns=unknown_tuple,
        original_variable=variable,
        scales=scale_tuple,
        scale_orders=orders,
        frequencies=frequencies,
        slow_flow=tuple(slow_flow),
        complete=complete,
        unresolved_order=None if complete else 2,
        limitation=limitation,
    )


__all__ = [
    "MultipleScalesResult",
    "SlowFlowEquation",
    "multiple_scale_derivative",
    "multiple_scales",
]
