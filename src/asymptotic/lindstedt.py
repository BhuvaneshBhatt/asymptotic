"""Lindstedt–Poincaré expansions for weakly nonlinear oscillators.

The implementation is a transformation and solvability policy over the common
perturbation hierarchy.  Frequency corrections are determined by periodic
Fredholm conditions from :mod:`asymptotic.resonance`, while the coefficient
ODEs and initial data use the same order representation as regular
perturbation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

from ._linear_ode_operator import linear_operator_coefficients
from ._perturbation_ode import apply_ode_conditions, dsolve_order
from ._symbolic_policy import bounded_simplify, bounded_solve_system
from .perturbation import (
    EquationLike,
    PerturbationHierarchy,
    _substitute_unknowns,
    perturbation_hierarchy,
)
from .resonance import periodic_solvability_conditions


def _residual(equation: EquationLike) -> sp.Expr:
    equation = sp.sympify(equation)
    if isinstance(equation, sp.Equality):
        return sp.expand(equation.lhs - equation.rhs)
    return sp.expand(equation)


def _natural_frequency(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    parameter: sp.Symbol,
) -> sp.Expr | None:
    unperturbed = sp.expand(equation.subs(parameter, 0))
    data = linear_operator_coefficients(unperturbed, unknown, variable)
    if data is None:
        return None
    coefficients, order = data
    if order != 2:
        return None
    a0, a1, a2 = coefficients
    if bounded_simplify(a1) != 0 or a2 == 0:
        return None
    if variable in a0.free_symbols or variable in a2.free_symbols:
        return None
    ratio = bounded_simplify(a0 / a2)
    if ratio.is_negative is True or ratio.is_zero is True:
        return None
    return bounded_simplify(sp.sqrt(ratio))


def _transform_equation(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    profile: AppliedUndef,
    fast_variable: sp.Symbol,
    frequency: sp.Symbol,
) -> sp.Expr:
    result = sp.sympify(equation)
    derivatives = sorted(
        (
            derivative
            for derivative in result.atoms(sp.Derivative)
            if derivative.expr == unknown
            and all(item == variable for item in derivative.variables)
        ),
        key=lambda derivative: derivative.derivative_count,
        reverse=True,
    )
    replacements = {
        derivative: frequency**derivative.derivative_count
        * sp.diff(profile, fast_variable, derivative.derivative_count)
        for derivative in derivatives
    }
    if replacements:
        result = result.xreplace(replacements)
    result = result.xreplace({unknown: profile})
    if variable in result.free_symbols:
        raise NotImplementedError(
            "Lindstedt-Poincare accepts autonomous governing equations"
        )
    return sp.expand(result)


def _transform_condition(
    condition: EquationLike,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    profile: AppliedUndef,
    fast_variable: sp.Symbol,
    frequency: sp.Symbol,
) -> sp.Expr:
    result = _residual(condition)

    for node in tuple(result.atoms(sp.Subs)):
        expression = node.expr
        if not isinstance(expression, sp.Derivative) or expression.expr != unknown:
            continue
        if tuple(node.variables) != (variable,) or len(node.point) != 1:
            continue
        point = node.point[0]
        if bounded_simplify(point) != 0:
            raise NotImplementedError(
                "Lindstedt-Poincare conditions are restricted to the initial point"
            )
        count = expression.derivative_count
        replacement = frequency**count * sp.diff(profile, fast_variable, count).subs(
            fast_variable, 0
        )
        result = result.xreplace({node: replacement})

    def matches(node: sp.Basic) -> bool:
        return isinstance(node, AppliedUndef) and node.func == unknown.func

    def replace(node: AppliedUndef) -> sp.Expr:
        if len(node.args) != 1 or bounded_simplify(node.args[0]) != 0:
            if node == unknown:
                return profile
            raise NotImplementedError(
                "Lindstedt-Poincare conditions are restricted to the initial point"
            )
        return profile.subs(fast_variable, 0)

    result = result.replace(matches, replace)
    return sp.expand(result)


def _zero_profile(
    expression: sp.Expr, profile: AppliedUndef, variable: sp.Symbol
) -> sp.Expr:
    replacements: dict[sp.Expr, sp.Expr] = {profile: sp.S.Zero}
    replacements.update(
        {
            derivative: sp.S.Zero
            for derivative in expression.atoms(sp.Derivative)
            if derivative.expr == profile
            and all(item == variable for item in derivative.variables)
        }
    )
    return sp.expand(expression.xreplace(replacements))


def _known_symbols(
    expressions: tuple[sp.Expr, ...], variable: sp.Symbol
) -> set[sp.Symbol]:
    symbols: set[sp.Symbol] = {variable}
    for expression in expressions:
        symbols.update(expression.free_symbols)
    return symbols


@dataclass(frozen=True)
class LindstedtPoincareResult:
    """Periodic strained-time expansion and its solved perturbation hierarchy."""

    hierarchy: PerturbationHierarchy
    original_unknown: AppliedUndef
    original_variable: sp.Symbol
    fast_variable: sp.Symbol
    frequency_symbol: sp.Symbol
    complete: bool
    unresolved_order: int | None = None
    limitation: str = ""

    @property
    def frequency(self) -> sp.Expr:
        """Return the solved frequency expansion in the perturbation parameter."""
        components = self.hierarchy.coefficient_unknowns[1]
        substitutions = self.hierarchy.substitutions()
        return bounded_simplify(
            sp.Add(
                *(
                    gauge * substitutions.get(component, component)
                    for gauge, component in zip(self.hierarchy.gauges, components)
                )
            )
        )

    @property
    def profile(self) -> sp.Expr:
        """Return the periodic profile as a function of the fast variable."""
        components = self.hierarchy.coefficient_unknowns[0]
        substitutions = self.hierarchy.substitutions()
        return bounded_simplify(
            sp.Add(
                *(
                    gauge * substitutions.get(component, component)
                    for gauge, component in zip(self.hierarchy.gauges, components)
                )
            )
        )

    @property
    def approximation(self) -> sp.Expr:
        """Return the physical-time approximation, truncated to the requested order."""
        expression = self.profile.subs(
            self.fast_variable,
            self.frequency * self.original_variable,
        )
        if all(
            gauge == self.hierarchy.parameter**i
            for i, gauge in enumerate(self.hierarchy.gauges)
        ):
            try:
                return sp.expand(
                    sp.series(
                        expression,
                        self.hierarchy.parameter,
                        0,
                        len(self.hierarchy.gauges),
                    ).removeO()
                )
            except (NotImplementedError, TypeError, ValueError, sp.PoleError):
                pass
        return expression

    @property
    def verified(self) -> bool:
        """Whether solved order equations, conditions, and solvability projections vanish."""
        through = self.hierarchy.solved_through
        if through is None:
            return False
        substitutions = self.hierarchy.substitutions(through)
        supplied_unknowns = tuple(substitutions)
        supplied_values = tuple(substitutions.values())
        for order in self.hierarchy.orders[: through + 1]:
            for expression in order.equations + order.conditions:
                evaluated = _substitute_unknowns(
                    expression, supplied_unknowns, supplied_values
                )
                evaluated = bounded_simplify(sp.expand_trig(evaluated))
                if evaluated != 0:
                    return False
            for condition in order.solvability_conditions:
                evaluated = _substitute_unknowns(
                    condition.expression, supplied_unknowns, supplied_values
                )
                if bounded_simplify(sp.expand_trig(evaluated)) != 0:
                    return False
        return True


def _continue_lindstedt_hierarchy(
    hierarchy: PerturbationHierarchy,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    tau: sp.Symbol,
    omega: sp.Symbol,
    start_index: int,
) -> LindstedtPoincareResult:
    """Continue a prepared Lindstedt hierarchy from one perturbation order."""
    profile_components, omega_components = hierarchy.coefficient_unknowns
    for index in range(start_index, len(hierarchy.orders)):
        order_data = hierarchy.order(index)
        profile_i = profile_components[index]
        omega_i = omega_components[index]
        previous = hierarchy.substitutions(index - 1)
        equations = tuple(
            sp.expand(expr.subs(previous).doit()) for expr in order_data.equations
        )
        conditions_i = tuple(
            sp.expand(expr.subs(previous).doit()) for expr in order_data.conditions
        )
        if len(equations) != 1:
            return LindstedtPoincareResult(
                hierarchy,
                unknown,
                variable,
                tau,
                omega,
                False,
                index,
                "Lindstedt-Poincare requires one scalar order equation",
            )
        equation_i = equations[0]
        forcing = _zero_profile(equation_i, profile_i, tau)
        linear_part = sp.expand(equation_i - forcing)
        if linear_operator_coefficients(linear_part, profile_i, tau) is None:
            return LindstedtPoincareResult(
                hierarchy,
                unknown,
                variable,
                tau,
                omega,
                False,
                index,
                "the coefficient equation is not linear in the current profile correction",
            )

        projection = periodic_solvability_conditions(
            forcing,
            tau,
            (sp.cos(tau), sp.sin(tau)),
            source_order=index,
        )
        hierarchy = hierarchy.with_solvability_conditions(
            index,
            tuple(condition.expression for condition in projection.conditions),
            reason="periodic Fredholm solvability",
        )
        equations_for_omega = tuple(
            bounded_simplify(value)
            for value in projection.projections
            if bounded_simplify(value) != 0
        )
        if equations_for_omega:
            omega_solutions = bounded_solve_system(
                equations_for_omega,
                (omega_i,),
                allow_general=True,
            )
            if (
                omega_solutions is None
                or len(omega_solutions) != 1
                or omega_i not in omega_solutions[0]
            ):
                return LindstedtPoincareResult(
                    hierarchy,
                    unknown,
                    variable,
                    tau,
                    omega,
                    False,
                    index,
                    "periodic solvability did not determine a unique frequency correction",
                )
            omega_value = bounded_simplify(omega_solutions[0][omega_i])
        else:
            omega_value = sp.S.Zero

        equation_i = sp.expand(equation_i.subs(omega_i, omega_value))
        conditions_i = tuple(
            sp.expand(expr.subs(omega_i, omega_value)) for expr in conditions_i
        )
        solved_i = dsolve_order((equation_i,), (profile_i,))
        if solved_i is None:
            return LindstedtPoincareResult(
                hierarchy,
                unknown,
                variable,
                tau,
                omega,
                False,
                index,
                f"could not solve periodic profile correction at order {index}",
            )
        conditioned_i = apply_ode_conditions(
            solved_i,
            (profile_i,),
            conditions_i,
            tau,
            _known_symbols((equation_i,) + conditions_i, tau),
        )
        if conditioned_i is None:
            return LindstedtPoincareResult(
                hierarchy,
                unknown,
                variable,
                tau,
                omega,
                False,
                index,
                f"could not determine profile constants at order {index}",
            )
        hierarchy = hierarchy.with_solution(
            index,
            {profile_i: conditioned_i[0], omega_i: omega_value},
        )

    return LindstedtPoincareResult(
        hierarchy=hierarchy,
        original_unknown=unknown,
        original_variable=variable,
        fast_variable=tau,
        frequency_symbol=omega,
        complete=True,
    )


def lindstedt_poincare(
    equation: EquationLike,
    unknown: sp.Expr,
    parameter: sp.Symbol,
    *,
    order: int = 1,
    conditions: EquationLike | Sequence[EquationLike] = (),
    base_frequency: sp.Expr | None = None,
    fast_variable: sp.Symbol | None = None,
    assumptions: sp.Expr = sp.S.true,
) -> LindstedtPoincareResult:
    """Construct a Lindstedt–Poincaré expansion for a scalar oscillator.

    The unperturbed equation must be an autonomous, undamped, constant-
    coefficient second-order oscillator.  Its natural frequency is inferred
    unless ``base_frequency`` is supplied.  At each later order, periodic
    Fredholm projections against ``cos(tau)`` and ``sin(tau)`` determine the
    frequency correction before the bounded coefficient ODE is solved.

    Unsupported or unresolved orders are returned as a solved prefix with an
    explicit ``unresolved_order`` instead of accepting secular growth.
    """
    parameter = sp.sympify(parameter)
    unknown = sp.sympify(unknown)
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a Symbol")
    if not isinstance(unknown, AppliedUndef) or len(unknown.args) != 1:
        raise TypeError("unknown must be a scalar applied function of one variable")
    variable = unknown.args[0]
    if not isinstance(variable, sp.Symbol):
        raise TypeError("the oscillator independent variable must be a Symbol")
    if order < 0:
        raise ValueError("order must be nonnegative")

    original = _residual(equation)
    omega0 = (
        sp.sympify(base_frequency)
        if base_frequency is not None
        else _natural_frequency(original, unknown, variable, parameter)
    )
    if omega0 is None:
        raise NotImplementedError(
            "could not infer a nonzero undamped second-order base frequency"
        )

    tau = fast_variable or sp.Symbol("tau", real=True)
    if not isinstance(tau, sp.Symbol):
        raise TypeError("fast_variable must be a Symbol")
    profile = sp.Function(f"{unknown.func.__name__}_profile")(tau)
    omega = sp.Symbol("omega")
    transformed = _transform_equation(original, unknown, variable, profile, tau, omega)
    condition_items = (
        ()
        if conditions == ()
        else (
            tuple(conditions)
            if isinstance(conditions, (tuple, list))
            else (conditions,)
        )
    )
    transformed_conditions = tuple(
        _transform_condition(condition, unknown, variable, profile, tau, omega)
        for condition in condition_items
    )
    hierarchy = perturbation_hierarchy(
        transformed,
        (profile, omega),
        parameter,
        order=order,
        conditions=transformed_conditions,
        assumptions=assumptions,
    )

    profile_components, omega_components = hierarchy.coefficient_unknowns
    leading_profile = profile_components[0]
    leading_omega = omega_components[0]
    leading_order = hierarchy.order(0)
    leading_eqs = tuple(
        sp.expand(expr.subs(leading_omega, omega0)) for expr in leading_order.equations
    )
    leading_conditions = tuple(
        sp.expand(expr.subs(leading_omega, omega0)) for expr in leading_order.conditions
    )
    solved = dsolve_order(leading_eqs, (leading_profile,))
    if solved is None:
        return LindstedtPoincareResult(
            hierarchy=hierarchy,
            original_unknown=unknown,
            original_variable=variable,
            fast_variable=tau,
            frequency_symbol=omega,
            complete=False,
            unresolved_order=0,
            limitation="could not solve the leading periodic oscillator equation",
        )
    conditioned = apply_ode_conditions(
        solved,
        (leading_profile,),
        leading_conditions,
        tau,
        _known_symbols(leading_eqs + leading_conditions, tau),
    )
    if conditioned is None:
        return LindstedtPoincareResult(
            hierarchy=hierarchy,
            original_unknown=unknown,
            original_variable=variable,
            fast_variable=tau,
            frequency_symbol=omega,
            complete=False,
            unresolved_order=0,
            limitation="could not determine the leading solution from the supplied conditions",
        )
    hierarchy = hierarchy.with_solution(
        0,
        {leading_profile: conditioned[0], leading_omega: omega0},
    )

    return _continue_lindstedt_hierarchy(hierarchy, unknown, variable, tau, omega, 1)


__all__ = ["LindstedtPoincareResult", "lindstedt_poincare"]
