"""Matched asymptotic expansions for singularly perturbed boundary problems.

The automatic workflow targets scalar linear second-order boundary-value
problems whose reduced equation loses one differential order.  It discovers
endpoint layer scales from Newton-style coefficient valuations, solves outer
and stretched inner perturbation hierarchies, enforces overlap matching, and
constructs uniformly valid composite approximations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

from ._perturbation_ode import (
    dsolve_order,
    introduced_constants,
    replace_function_family,
)
from ._symbolic_policy import bounded_limit, bounded_simplify, bounded_solve_system
from .perturbation import (
    EquationLike,
    PerturbationHierarchy,
    _as_residual,
    _gauge_coefficients,
    perturbation_hierarchy,
)
from .regular_perturbation import RegularPerturbationBranch, regular_perturbation


@dataclass(frozen=True)
class BoundaryLayerScale:
    """Certified endpoint scaling inferred from a dominant derivative balance."""

    point: sp.Expr
    side: str
    exponent: sp.Rational
    parameter: sp.Symbol
    stretched_variable: sp.Symbol
    orientation: int
    balance_orders: tuple[tuple[int, sp.Rational, int], ...]

    @property
    def scale(self) -> sp.Expr:
        """Return the layer width as a power of the perturbation parameter placeholder."""
        return self.parameter**self.exponent


@dataclass(frozen=True)
class MatchedAsymptoticBranch:
    """One successfully matched endpoint-layer construction."""

    layer: BoundaryLayerScale
    outer: RegularPerturbationBranch
    inner_hierarchy: PerturbationHierarchy
    matching_terms: tuple[sp.Expr, ...]
    composite: sp.Expr
    equation: sp.Expr
    conditions: tuple[sp.Expr, ...]
    unknown: AppliedUndef
    variable: sp.Symbol
    parameter: sp.Symbol
    complete: bool
    unresolved_order: int | None = None
    limitation: str = ""

    @property
    def outer_approximation(self) -> sp.Expr:
        """Return the solved outer approximation."""
        return self.outer.approximation[0]

    @property
    def inner_approximation(self) -> sp.Expr:
        """Return the solved stretched inner approximation."""
        through = self.inner_hierarchy.solved_through
        if through is None:
            through = 0
        return self.inner_hierarchy.approximation(through)[0]

    @property
    def common_part(self) -> sp.Expr:
        """Return the overlap expansion expressed in the stretched coordinate."""
        return sp.expand(
            sum(
                gauge * term
                for gauge, term in zip(self.inner_hierarchy.gauges, self.matching_terms)
            )
        )

    @property
    def residual(self) -> sp.Expr:
        """Return the original differential-equation residual of the composite."""
        return _replace_solution(
            self.equation, self.unknown, self.composite, self.variable
        )

    @property
    def condition_residuals(self) -> tuple[sp.Expr, ...]:
        """Return original boundary-condition residuals of the composite."""
        return tuple(
            _replace_solution(condition, self.unknown, self.composite, self.variable)
            for condition in self.conditions
        )

    @property
    def verified(self) -> bool:
        """Whether the composite residuals vanish through the requested algebraic order."""
        order = self.inner_hierarchy.solved_through
        if order is None:
            return False
        denominator = self.inner_hierarchy.gauges[order]
        for residual in (self.residual,) + self.condition_residuals:
            value = bounded_limit(
                residual / denominator,
                self.parameter,
                0,
                direction="+",
            )
            if value != 0:
                return False
        return True


@dataclass(frozen=True)
class MatchedAsymptoticResult:
    """Endpoint-layer branches produced by matched asymptotic analysis."""

    branches: tuple[MatchedAsymptoticBranch, ...]
    parameter: sp.Symbol
    unknown: AppliedUndef
    variable: sp.Symbol
    interval: tuple[sp.Expr, sp.Expr]
    full_order: int
    reduced_order: int
    method: str = "matched"

    @property
    def complete(self) -> bool:
        """Whether at least one retained branch is solved through every requested order."""
        return bool(self.branches) and all(branch.complete for branch in self.branches)

    @property
    def composites(self) -> tuple[sp.Expr, ...]:
        """Return all retained uniformly valid composite approximations."""
        return tuple(branch.composite for branch in self.branches)


def _replace_solution(
    expression: sp.Expr,
    unknown: AppliedUndef,
    solution: sp.Expr,
    variable: sp.Symbol,
) -> sp.Expr:
    derivatives = sorted(
        (
            derivative
            for derivative in expression.atoms(sp.Derivative)
            if derivative.expr == unknown
            and all(item == variable for item in derivative.variables)
        ),
        key=lambda item: item.derivative_count,
        reverse=True,
    )
    replacements = {
        derivative: sp.diff(solution, variable, derivative.derivative_count)
        for derivative in derivatives
    }
    transformed = expression.xreplace(replacements)
    transformed = replace_function_family(transformed, unknown.func, solution, variable)
    return sp.expand(transformed.doit())


def _jet_data(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
) -> tuple[tuple[sp.Expr, ...], sp.Expr, int] | None:
    derivatives = [
        item for item in equation.atoms(sp.Derivative) if item.expr == unknown
    ]
    order = max((item.derivative_count for item in derivatives), default=0)
    jets = tuple(sp.Dummy(f"matched_D{k}") for k in range(order + 1))
    replacements: dict[sp.Expr, sp.Expr] = {unknown: jets[0]}
    replacements.update(
        {sp.diff(unknown, variable, k): jets[k] for k in range(1, order + 1)}
    )
    algebraic = sp.expand(equation.xreplace(replacements))
    try:
        poly = sp.Poly(algebraic, *jets)
    except sp.PolynomialError:
        return None
    if poly.total_degree() > 1:
        return None
    coefficients = tuple(sp.expand(poly.coeff_monomial(jet)) for jet in jets)
    constant = sp.expand(poly.coeff_monomial((0,) * len(jets)))
    reconstructed = sp.expand(
        constant + sum(coeff * jet for coeff, jet in zip(coefficients, jets))
    )
    if sp.expand(reconstructed - algebraic) != 0:
        return None
    return coefficients, constant, order


def _differential_order(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
) -> int:
    derivatives = [
        item.derivative_count
        for item in equation.atoms(sp.Derivative)
        if item.expr == unknown and all(arg == variable for arg in item.variables)
    ]
    return max(derivatives, default=0)


def _dirichlet_condition(
    condition: sp.Expr,
    unknown: AppliedUndef,
) -> tuple[sp.Expr, sp.Expr] | None:
    residual = sp.expand(condition)
    applications = tuple(
        item
        for item in residual.atoms(AppliedUndef)
        if item.func == unknown.func and len(item.args) == 1
    )
    if len(applications) != 1 or residual.has(sp.Derivative):
        return None
    application = applications[0]
    coefficient = sp.expand(residual).coeff(application)
    if coefficient == 0:
        return None
    remainder = sp.expand(residual - coefficient * application)
    if remainder.has(unknown.func):
        return None
    return application.args[0], sp.cancel(-remainder / coefficient)


def _coefficient_monomials(
    coefficient: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    parameter: sp.Symbol,
    derivative_order: int,
) -> tuple[tuple[sp.Rational, int, int], ...]:
    if coefficient == 0:
        return ()
    h = sp.Dummy("layer_offset")
    local = sp.expand(coefficient.subs(variable, point + h))
    try:
        poly = sp.Poly(local, parameter, h)
    except sp.PolynomialError:
        return ()
    result = []
    for (eps_power, offset_power), coeff in poly.terms():
        if coeff == 0:
            continue
        result.append((sp.Rational(eps_power), int(offset_power), derivative_order))
    return tuple(result)


def _scale_exponents(
    coefficients: tuple[sp.Expr, ...],
    variable: sp.Symbol,
    point: sp.Expr,
    parameter: sp.Symbol,
    full_order: int,
) -> tuple[tuple[sp.Rational, tuple[tuple[int, sp.Rational, int], ...]], ...]:
    terms: list[tuple[sp.Rational, int, int]] = []
    for derivative_order, coefficient in enumerate(coefficients):
        terms.extend(
            _coefficient_monomials(
                coefficient,
                variable,
                point,
                parameter,
                derivative_order,
            )
        )
    candidates: dict[sp.Rational, tuple[tuple[int, sp.Rational, int], ...]] = {}
    for i, first in enumerate(terms):
        q1, r1, k1 = first
        slope1 = r1 - k1
        for second in terms[i + 1 :]:
            q2, r2, k2 = second
            if full_order not in (k1, k2):
                continue
            slope2 = r2 - k2
            if slope1 == slope2:
                continue
            exponent = sp.Rational(q2 - q1, slope1 - slope2)
            if exponent <= 0:
                continue
            valuations = [q + exponent * (r - k) for q, r, k in terms]
            minimum = min(valuations)
            tied = tuple(
                (k, q, r)
                for (q, r, k), valuation in zip(terms, valuations)
                if valuation == minimum
            )
            if len(tied) >= 2 and any(k == full_order for k, _, _ in tied):
                candidates[exponent] = tied
    return tuple(sorted(candidates.items(), key=lambda item: item[0]))


def _leading_inner_power(
    balance: tuple[tuple[int, sp.Rational, int], ...],
    exponent: sp.Rational,
) -> sp.Rational:
    return min(q + exponent * (r - k) for k, q, r in balance)


def _transform_inner(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    profile: AppliedUndef,
    stretched: sp.Symbol,
    parameter: sp.Symbol,
    point: sp.Expr,
    orientation: int,
    exponent: sp.Rational,
) -> sp.Expr:
    replacements: dict[sp.Expr, sp.Expr] = {}
    for derivative in equation.atoms(sp.Derivative):
        if derivative.expr != unknown or any(
            item != variable for item in derivative.variables
        ):
            continue
        count = derivative.derivative_count
        replacements[derivative] = (
            orientation / parameter**exponent
        ) ** count * sp.diff(profile, stretched, count)
    replacements[unknown] = profile
    transformed = equation.xreplace(replacements)
    transformed = transformed.subs(
        variable,
        point + orientation * parameter**exponent * stretched,
    )
    return sp.expand(transformed.doit())


def _matching_terms(
    outer: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    orientation: int,
    exponent: sp.Rational,
    stretched: sp.Symbol,
    parameter: sp.Symbol,
    gauges: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...]:
    local = sp.expand(
        outer.subs(
            variable,
            point + orientation * parameter**exponent * stretched,
        )
    )
    return _gauge_coefficients(local, parameter, gauges)


def _solve_inner_order(
    hierarchy: PerturbationHierarchy,
    index: int,
    matching: sp.Expr,
    stretched: sp.Symbol,
) -> PerturbationHierarchy | None:
    order = hierarchy.order(index)
    previous = hierarchy.substitutions(index - 1) if index else {}
    equations = tuple(sp.expand(item.subs(previous).doit()) for item in order.equations)
    conditions = tuple(
        sp.expand(item.subs(previous).doit()) for item in order.conditions
    )
    unknown = order.unknowns[0]
    if not isinstance(unknown, AppliedUndef):
        return None
    solved = dsolve_order(equations, (unknown,))
    if solved is None:
        return None
    solution = solved[0]
    residuals = [
        replace_function_family(condition, unknown.func, solution, stretched)
        for condition in conditions
    ]
    match_residual = bounded_limit(
        sp.expand(solution - matching),
        stretched,
        sp.oo,
        direction="-",
    )
    if match_residual is None:
        return None
    residuals.append(match_residual)
    known_symbols = set().union(*(item.free_symbols for item in equations + conditions))
    known_symbols.update(matching.free_symbols)
    known_symbols.add(stretched)
    constants = introduced_constants((solution,), known_symbols)
    if constants:
        values = bounded_solve_system(residuals, constants, allow_general=True)
        if values is None or len(values) != 1:
            return None
        solution = bounded_simplify(solution.subs(values[0]))
    elif any(bounded_simplify(item) != 0 for item in residuals):
        return None
    return hierarchy.with_solution(index, {unknown: solution})


def _inner_hierarchy(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    layer: BoundaryLayerScale,
    layer_value: sp.Expr,
    outer: RegularPerturbationBranch,
    gauges: tuple[sp.Expr, ...],
    minimum_power: sp.Rational,
) -> tuple[PerturbationHierarchy, tuple[sp.Expr, ...]] | None:
    stretched = layer.stretched_variable
    profile = sp.Function(f"{unknown.func.__name__}_inner")(stretched)
    transformed = _transform_inner(
        equation,
        unknown,
        variable,
        profile,
        stretched,
        parameter,
        layer.point,
        layer.orientation,
        layer.exponent,
    )
    normalized = sp.expand(transformed * parameter ** (-minimum_power))
    hierarchy = perturbation_hierarchy(
        normalized,
        profile,
        parameter,
        gauges=gauges,
        conditions=(profile.subs(stretched, 0) - layer_value,),
    )
    outer_order = len(outer.hierarchy.gauges) - 1
    outer_approx = outer.hierarchy.approximation(outer_order)[0]
    matching = _matching_terms(
        outer_approx,
        variable,
        layer.point,
        layer.orientation,
        layer.exponent,
        stretched,
        parameter,
        hierarchy.gauges,
    )
    current = hierarchy
    for index in range(len(gauges)):
        advanced = _solve_inner_order(current, index, matching[index], stretched)
        if advanced is None:
            return None
        current = advanced
    return current, matching


def _composite(
    outer: sp.Expr,
    inner: sp.Expr,
    matching: tuple[sp.Expr, ...],
    gauges: tuple[sp.Expr, ...],
    layer: BoundaryLayerScale,
    variable: sp.Symbol,
    parameter: sp.Symbol,
) -> sp.Expr:
    common = sp.expand(sum(gauge * term for gauge, term in zip(gauges, matching)))
    stretched_physical = (
        layer.orientation * (variable - layer.point) / parameter**layer.exponent
    )
    correction = sp.expand(
        (inner - common).subs(layer.stretched_variable, stretched_physical)
    )
    return bounded_simplify(sp.expand(outer + correction))


def matched_asymptotic_expansion(
    equation: EquationLike,
    unknown: sp.Expr,
    parameter: sp.Symbol,
    *,
    conditions: Sequence[EquationLike],
    order: int = 0,
    assumptions: sp.Expr = sp.S.true,
) -> MatchedAsymptoticResult:
    """Construct endpoint-layer matched expansions for a singular scalar BVP.

    The automatic solver requires a scalar linear second-order ODE whose
    ``parameter -> 0`` reduction is first order, together with two endpoint
    Dirichlet conditions.  Candidate layer widths are inferred from local
    Newton-style balances, not assumed to equal ``parameter``.  For every
    viable endpoint, the omitted boundary condition is restored by a stretched
    inner hierarchy and matched to the outer hierarchy in the overlap region.

    Higher-order matching is supported when the inferred scaling and local
    expansions lie in the integer perturbation gauges requested by ``order``.
    Unsupported structures are rejected explicitly rather than assigned a
    heuristic layer.
    """
    parameter = sp.sympify(parameter)
    unknown = sp.sympify(unknown)
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a Symbol")
    if not isinstance(unknown, AppliedUndef) or len(unknown.args) != 1:
        raise TypeError("unknown must be an applied scalar function of one variable")
    variable = unknown.args[0]
    if not isinstance(variable, sp.Symbol):
        raise TypeError("the independent variable must be a Symbol")
    if order < 0:
        raise ValueError("order must be nonnegative")
    residual = _as_residual(equation)
    data = _jet_data(residual, unknown, variable)
    if data is None:
        raise NotImplementedError(
            "automatic matched expansion requires a scalar linear ODE"
        )
    coefficients, _, full_order = data
    reduced = sp.expand(residual.subs(parameter, 0))
    reduced_order = _differential_order(reduced, unknown, variable)
    if full_order != 2 or reduced_order != 1:
        raise NotImplementedError(
            "automatic matched expansion requires second-to-first-order singular reduction"
        )

    condition_tuple = tuple(_as_residual(item) for item in conditions)
    parsed = tuple(_dirichlet_condition(item, unknown) for item in condition_tuple)
    if len(parsed) != 2 or any(item is None for item in parsed):
        raise NotImplementedError(
            "automatic matched expansion requires two endpoint Dirichlet conditions"
        )
    boundary_data = tuple(item for item in parsed if item is not None)
    boundary_data = tuple(
        sorted(boundary_data, key=lambda item: sp.default_sort_key(item[0]))
    )
    left, right = boundary_data
    if left[0] == right[0]:
        raise ValueError("boundary conditions must be imposed at distinct endpoints")

    branches: list[MatchedAsymptoticBranch] = []
    endpoint_specs = (
        ("left", left, right, 1),
        ("right", right, left, -1),
    )
    for side, layer_data, outer_data, orientation in endpoint_specs:
        point, layer_value = layer_data
        candidates = _scale_exponents(
            coefficients,
            variable,
            point,
            parameter,
            full_order,
        )
        for exponent, balance in candidates:
            denominator = int(exponent.q)
            gauges = tuple(
                parameter ** sp.Rational(index, denominator)
                for index in range(order * denominator + 1)
            )
            stretched = sp.Symbol("X" if side == "left" else "R", nonnegative=True)
            layer = BoundaryLayerScale(
                point=point,
                side=side,
                exponent=exponent,
                parameter=parameter,
                stretched_variable=stretched,
                orientation=orientation,
                balance_orders=balance,
            )
            outer_condition = unknown.func(outer_data[0]) - outer_data[1]
            outer_result = regular_perturbation(
                residual,
                unknown,
                parameter,
                order=order,
                conditions=(outer_condition,),
                assumptions=assumptions,
            )
            for outer_branch in outer_result.branches:
                if not outer_branch.complete:
                    continue
                minimum_power = _leading_inner_power(balance, exponent)
                inner_data = _inner_hierarchy(
                    residual,
                    unknown,
                    variable,
                    parameter,
                    layer,
                    layer_value,
                    outer_branch,
                    gauges,
                    minimum_power,
                )
                if inner_data is None:
                    continue
                inner_hierarchy, matching = inner_data
                outer_approx = outer_branch.hierarchy.approximation(order)[0]
                inner_approx = inner_hierarchy.approximation(order)[0]
                composite = _composite(
                    outer_approx,
                    inner_approx,
                    matching,
                    inner_hierarchy.gauges,
                    layer,
                    variable,
                    parameter,
                )
                branches.append(
                    MatchedAsymptoticBranch(
                        layer=layer,
                        outer=outer_branch,
                        inner_hierarchy=inner_hierarchy,
                        matching_terms=matching,
                        composite=composite,
                        equation=residual,
                        conditions=condition_tuple,
                        unknown=unknown,
                        variable=variable,
                        parameter=parameter,
                        complete=True,
                    )
                )
            if branches and any(branch.layer.side == side for branch in branches):
                break

    return MatchedAsymptoticResult(
        branches=tuple(branches),
        parameter=parameter,
        unknown=unknown,
        variable=variable,
        interval=(left[0], right[0]),
        full_order=full_order,
        reduced_order=reduced_order,
    )


__all__ = [
    "BoundaryLayerScale",
    "MatchedAsymptoticBranch",
    "MatchedAsymptoticResult",
    "matched_asymptotic_expansion",
]
