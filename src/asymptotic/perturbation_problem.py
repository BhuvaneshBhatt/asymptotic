"""High-level structured interface for perturbation problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import sympy as sp
from sympy.core.function import AppliedUndef

from .evidence import Evidence, EvidenceStatus
from .lindstedt import LindstedtPoincareResult
from .matched import MatchedAsymptoticResult
from .multiple_scales import MultipleScalesResult
from .perturbation import _as_residual, _substitute_unknowns
from .perturbation_dispatch import (
    PerturbationDiagnostics,
    PerturbationExpansionResult,
    PerturbationMethod,
    perturbation_diagnostics,
    perturbation_expansion,
)
from .regular_perturbation import RegularPerturbationResult

ConditionKind = Literal["value", "derivative", "limit", "periodic", "matching"]
ProblemKind = Literal["algebraic", "algebraic-system", "ode", "ode-system"]


@dataclass(frozen=True)
class PerturbationCondition:
    """Canonical value, derivative, limit, periodic, or matching condition."""

    unknown: sp.Expr
    value: sp.Expr
    point: sp.Expr
    derivative_order: int = 0
    kind: ConditionKind = "value"
    direction: str = "+"
    reference_point: sp.Expr | None = None
    reference_unknown: sp.Expr | None = None

    def __post_init__(self) -> None:
        if self.derivative_order < 0:
            raise ValueError("derivative_order must be nonnegative")
        if self.kind == "value" and self.derivative_order:
            object.__setattr__(self, "kind", "derivative")
        if self.kind == "periodic" and self.reference_point is None:
            raise ValueError("periodic conditions require reference_point")
        if self.kind == "matching" and self.reference_unknown is None:
            raise ValueError("matching conditions require reference_unknown")

    def residual(self) -> sp.Expr:
        """Return the condition as a zero-valued SymPy residual."""
        unknown = sp.sympify(self.unknown)
        target = sp.sympify(self.value)
        if not isinstance(unknown, AppliedUndef) or len(unknown.args) != 1:
            raise ValueError("conditions require a one-variable applied function")
        variable = unknown.args[0]
        expression = sp.diff(unknown, variable, self.derivative_order)
        if self.kind == "limit":
            return (
                sp.Limit(expression, variable, self.point, dir=self.direction) - target
            )
        if self.kind == "periodic":
            left = expression.subs(variable, self.point)
            right = expression.subs(variable, sp.sympify(self.reference_point))
            return left - right - target
        if self.kind == "matching":
            reference = sp.sympify(self.reference_unknown)
            if not isinstance(reference, AppliedUndef) or len(reference.args) != 1:
                raise ValueError(
                    "reference_unknown must be a one-variable applied function"
                )
            reference_expression = sp.diff(
                reference, reference.args[0], self.derivative_order
            )
            reference_point = (
                self.point if self.reference_point is None else self.reference_point
            )
            left = _condition_value(expression, variable, self.point, self.direction)
            right = _condition_value(
                reference_expression,
                reference.args[0],
                reference_point,
                self.direction,
            )
            return left - right - target
        return expression.subs(variable, self.point) - target


def _condition_value(expression, variable, point, direction):
    if point in (sp.oo, -sp.oo):
        return sp.Limit(expression, variable, point, dir=direction)
    return expression.subs(variable, point)


@dataclass(frozen=True)
class ValidationReport:
    """Symbolic residual and condition validation for a perturbation approximation."""

    residuals: tuple[sp.Expr, ...]
    residual_orders: tuple[sp.Expr | None, ...]
    condition_residuals: tuple[sp.Expr, ...]
    evidence: Evidence


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """Numerical errors and observed versus expected power-law convergence."""

    epsilon_values: tuple[float, ...]
    errors: tuple[float, ...]
    observed_order: float | None
    expected_order: float | None
    order_difference: float | None
    evidence: Evidence


@dataclass(frozen=True)
class MethodRecommendation:
    """Explainable perturbation-method recommendation."""

    method: PerturbationMethod | None
    diagnostics: PerturbationDiagnostics
    reasons: tuple[str, ...]
    ambiguous: bool


@dataclass(frozen=True)
class PerturbationResult:
    """Uniform view over a method-specific perturbation result."""

    problem: PerturbationProblem
    expansion: PerturbationExpansionResult
    branch: int = 0

    @property
    def method(self) -> PerturbationMethod | None:
        """Return the selected perturbation method."""
        return self.expansion.selected_method

    @property
    def native(self):
        """Return the method-specific result for advanced inspection."""
        return self.expansion.result

    @property
    def orders(self):
        """Return the underlying perturbation hierarchy orders when available."""
        subject = self.native
        if isinstance(subject, RegularPerturbationResult):
            return subject.branches[self.branch].hierarchy.orders
        hierarchy = getattr(subject, "hierarchy", None)
        return () if hierarchy is None else hierarchy.orders

    @property
    def approximation(self) -> tuple[sp.Expr, ...]:
        """Return the selected approximation as a tuple matching the unknowns."""
        subject = self.native
        if subject is None:
            return ()
        if isinstance(subject, RegularPerturbationResult):
            return subject.branches[self.branch].approximation
        if isinstance(subject, LindstedtPoincareResult):
            return (subject.approximation,)
        if isinstance(subject, MultipleScalesResult):
            scale_substitutions = {
                scale: self.problem.parameter**power * subject.original_variable
                for scale, power in zip(
                    subject.scales, subject.scale_orders, strict=True
                )
            }
            return tuple(
                sp.expand(profile.subs(scale_substitutions))
                for profile in subject.profiles
            )
        if isinstance(subject, MatchedAsymptoticResult):
            return (subject.branches[self.branch].composite,)
        raise TypeError("unsupported perturbation result type")

    def evaluate(self, parameter_value, substitutions: Mapping | None = None):
        """Evaluate the approximation after substituting the small parameter and data."""
        values = {self.problem.parameter: sp.sympify(parameter_value)}
        if substitutions:
            values.update(substitutions)
        return tuple(sp.N(expr.subs(values)) for expr in self.approximation)

    def validate(self) -> ValidationReport:
        """Check symbolic equation residual orders and supplied conditions."""
        approximations = self.approximation
        residuals = tuple(
            sp.expand(
                _substitute_unknowns(eq, self.problem.unknowns, approximations).doit()
            )
            for eq in self.problem.residuals
        )
        orders = tuple(
            _leading_parameter_order(r, self.problem.parameter) for r in residuals
        )
        condition_residuals = tuple(
            sp.simplify(
                _condition_on_approximation(c, approximations, self.problem.unknowns)
            )
            for c in self.problem.conditions
        )
        unresolved = []
        if any(order is None for order in orders):
            unresolved.append("residual order could not be determined")
        if any(value != 0 for value in condition_residuals):
            unresolved.append("one or more supplied conditions do not vanish exactly")
        status = EvidenceStatus.PROVED if not unresolved else EvidenceStatus.FORMAL
        evidence = Evidence(
            status,
            "symbolic perturbation residual validation",
            obligations=tuple(unresolved),
        )
        return ValidationReport(residuals, orders, condition_residuals, evidence)

    def convergence_diagnostics(
        self,
        epsilon_values: Sequence[float],
        reference: Callable[[float], float | Sequence[float]],
        *,
        substitutions: Mapping | None = None,
    ) -> ConvergenceDiagnostics:
        """Estimate observed convergence order from errors at several epsilon values."""
        eps = tuple(float(value) for value in epsilon_values)
        if len(eps) < 2 or any(value <= 0 for value in eps):
            raise ValueError("at least two positive epsilon values are required")
        errors = []
        for value in eps:
            approx = self.evaluate(value, substitutions)
            exact = reference(value)
            exact_tuple = tuple(exact) if isinstance(exact, (tuple, list)) else (exact,)
            if len(exact_tuple) != len(approx):
                raise ValueError(
                    "reference result does not match the number of unknowns"
                )
            errors.append(
                max(
                    abs(float(sp.N(a - b)))
                    for a, b in zip(approx, exact_tuple, strict=True)
                )
            )
        usable = [(h, error) for h, error in zip(eps, errors, strict=True) if error > 0]
        observed = _log_log_slope(usable)
        validation = self.validate()
        finite_orders = [
            float(order)
            for order in validation.residual_orders
            if order is not None and order.is_finite is True and order.is_number
        ]
        expected = min(finite_orders) if finite_orders else None
        difference = (
            observed - expected
            if observed is not None and expected is not None
            else None
        )
        obligations = []
        if observed is None:
            obligations.append("nonzero errors at distinct epsilon values")
        if expected is None:
            obligations.append("a finite symbolic residual order")
        status = (
            EvidenceStatus.NUMERICALLY_SUPPORTED
            if observed is not None
            else EvidenceStatus.UNKNOWN
        )
        evidence = Evidence(
            status,
            "observed perturbation convergence order",
            obligations=tuple(obligations),
            details=(("expected_order", expected), ("order_difference", difference)),
        )
        return ConvergenceDiagnostics(
            eps, tuple(errors), observed, expected, difference, evidence
        )


def _log_log_slope(samples: Sequence[tuple[float, float]]) -> float | None:
    """Return the least-squares log-log slope for distinct positive samples."""
    if len(samples) < 2:
        return None
    points = [(math.log(h), math.log(error)) for h, error in samples]
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def _leading_parameter_order(
    expression: sp.Expr, parameter: sp.Symbol
) -> sp.Expr | None:
    expression = sp.cancel(expression)
    if expression == 0:
        return sp.oo
    try:
        leading = expression.as_leading_term(parameter)
    except (NotImplementedError, TypeError, ValueError, sp.PoleError):
        return None
    order = sp.sympify(leading.as_powers_dict().get(parameter, 0))
    return order if order.is_real is True else None


def _condition_on_approximation(condition, approximations, unknowns):
    residual = condition.residual()
    return _substitute_unknowns(residual, unknowns, approximations).doit()


class PerturbationProblem:
    """Structured perturbation problem with classification and method dispatch."""

    def __init__(
        self, equations, unknowns, parameter, *, conditions=(), assumptions=sp.S.true
    ):
        self.equations = (
            tuple(equations) if isinstance(equations, (tuple, list)) else (equations,)
        )
        self.unknowns = (
            tuple(unknowns) if isinstance(unknowns, (tuple, list)) else (unknowns,)
        )
        self.parameter = sp.sympify(parameter)
        if not isinstance(self.parameter, sp.Symbol):
            raise TypeError("parameter must be a Symbol")
        self.conditions = tuple(conditions)
        if any(not isinstance(item, PerturbationCondition) for item in self.conditions):
            raise TypeError("conditions must contain PerturbationCondition objects")
        self.assumptions = assumptions
        self.residuals = tuple(_as_residual(eq) for eq in self.equations)

    @property
    def kind(self) -> ProblemKind:
        """Classify the problem as algebraic/ODE and scalar/system."""
        differential = any(eq.atoms(sp.Derivative) for eq in self.residuals)
        system = len(self.unknowns) > 1 or len(self.equations) > 1
        if differential:
            return "ode-system" if system else "ode"
        return "algebraic-system" if system else "algebraic"

    def recommend_method(self) -> MethodRecommendation:
        """Return the best structural method recommendation and its reasons."""
        diagnostics = perturbation_diagnostics(
            self.equations,
            self.unknowns,
            self.parameter,
            conditions=self.condition_residuals,
            assumptions=self.assumptions,
        )
        preferred = diagnostics.preferred
        method = preferred[0].method if len(preferred) == 1 else None
        reasons = tuple(item.reason for item in preferred) or (
            "no implemented method is structurally applicable",
        )
        return MethodRecommendation(method, diagnostics, reasons, len(preferred) > 1)

    @property
    def condition_residuals(self) -> tuple[sp.Expr, ...]:
        """Return canonical conditions as zero-valued symbolic residuals."""
        return tuple(condition.residual() for condition in self.conditions)

    def solve(
        self,
        *,
        method: PerturbationMethod | Literal["auto"] = "auto",
        order: int = 1,
        **kwargs,
    ) -> PerturbationResult:
        """Solve with an explicit method or the unique recommended automatic method."""
        expansion = perturbation_expansion(
            self.equations,
            self.unknowns,
            self.parameter,
            method=method,
            order=order,
            conditions=self.condition_residuals,
            assumptions=self.assumptions,
            **kwargs,
        )
        if expansion.result is None:
            raise ValueError(
                expansion.limitation or "perturbation method could not be selected"
            )
        return PerturbationResult(self, expansion)
