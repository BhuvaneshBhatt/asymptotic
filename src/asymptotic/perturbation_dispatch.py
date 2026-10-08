"""Unified perturbation dispatch, continuation, and certification.

The dispatcher classifies a problem structurally before invoking one of the
method-specific solvers.  Automatic dispatch executes only when the
highest-ranked applicable method is unique; genuinely different asymptotic
descriptions are reported as an ambiguity instead of being chosen silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import sympy as sp
from funcprops import normalize_assumptions
from sympy.core.function import AppliedUndef

from ._symbolic_policy import bounded_simplify
from .evidence import Evidence, EvidenceStatus
from .lindstedt import (
    LindstedtPoincareResult,
    _continue_lindstedt_hierarchy,
    _natural_frequency,
    lindstedt_poincare,
)
from .matched import MatchedAsymptoticResult, matched_asymptotic_expansion
from .multiple_scales import MultipleScalesResult, _base_frequency, multiple_scales
from .perturbation import (
    EquationLike,
    PerturbationHierarchy,
    _as_residual,
    _substitute_unknowns,
)
from .regular_perturbation import (
    RegularPerturbationBranch,
    RegularPerturbationResult,
    _advance_branch,
    regular_perturbation,
)
from .remainder import Remainder

PerturbationMethod = Literal["regular", "lindstedt", "multiple-scales", "matched"]
PerturbationMethodResult = (
    RegularPerturbationResult
    | LindstedtPoincareResult
    | MultipleScalesResult
    | MatchedAsymptoticResult
)


def _as_tuple(value: object) -> tuple:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _differential_order(
    equations: tuple[sp.Expr, ...],
    unknowns: tuple[sp.Expr, ...],
) -> int:
    order = 0
    functions = {
        unknown.func for unknown in unknowns if isinstance(unknown, AppliedUndef)
    }
    for equation in equations:
        for derivative in equation.atoms(sp.Derivative):
            if (
                isinstance(derivative.expr, AppliedUndef)
                and derivative.expr.func in functions
            ):
                order = max(order, derivative.derivative_count)
    return order


def _has_explicit_variable(
    equation: sp.Expr,
    unknown: AppliedUndef,
    variable: sp.Symbol,
) -> bool:
    replacements: dict[sp.Expr, sp.Expr] = {unknown: sp.Dummy("state")}
    for derivative in equation.atoms(sp.Derivative):
        if derivative.expr == unknown and all(
            item == variable for item in derivative.variables
        ):
            replacements[derivative] = sp.Dummy(f"state_D{derivative.derivative_count}")
    reduced = equation.xreplace(replacements)
    return variable in reduced.free_symbols


def _shared_variable(unknowns: tuple[sp.Expr, ...]) -> sp.Symbol | None:
    if not unknowns or any(
        not isinstance(item, AppliedUndef) or len(item.args) != 1 for item in unknowns
    ):
        return None
    variable = unknowns[0].args[0]
    if not isinstance(variable, sp.Symbol):
        return None
    if any(item.args[0] != variable for item in unknowns):
        return None
    return variable


@dataclass(frozen=True)
class PerturbationMethodAssessment:
    """Structural applicability assessment for one perturbation method."""

    method: PerturbationMethod
    applicable: bool
    rank: int
    reason: str


@dataclass(frozen=True)
class PerturbationDiagnostics:
    """Ranked method assessments for a perturbation problem."""

    assessments: tuple[PerturbationMethodAssessment, ...]

    @property
    def applicable(self) -> tuple[PerturbationMethodAssessment, ...]:
        """Return applicable methods ordered from strongest to weakest preference."""
        return tuple(
            sorted(
                (item for item in self.assessments if item.applicable),
                key=lambda item: (item.rank, item.method),
            )
        )

    @property
    def preferred(self) -> tuple[PerturbationMethodAssessment, ...]:
        """Return all methods tied at the best applicable rank."""
        applicable = self.applicable
        if not applicable:
            return ()
        best = applicable[0].rank
        return tuple(item for item in applicable if item.rank == best)


@dataclass(frozen=True)
class FormalResidualCertificate:
    """Replayable formal statement that solved perturbation orders vanish."""

    method: str
    through_order: int | None
    verified: bool
    equation_residuals: tuple[sp.Expr, ...]
    condition_residuals: tuple[sp.Expr, ...]
    statement: str = "formal coefficient equations and supplied conditions vanish through the solved hierarchy"


@dataclass(frozen=True)
class PerturbationCertification:
    """Formal perturbation evidence plus an optional rigorous remainder bound.

    ``formal`` certifies only coefficient/residual cancellation.  It is not an
    error theorem.  ``rigorous_remainder`` is populated only from an existing
    certified :class:`Remainder`; the dispatcher never upgrades a
    formal residual calculation into a rigorous asymptotic bound.
    """

    formal: FormalResidualCertificate
    rigorous_remainder: Remainder | None = None

    @property
    def has_rigorous_bound(self) -> bool:
        """Whether an independently certified asymptotic remainder is attached."""
        return bool(
            self.rigorous_remainder is not None and self.rigorous_remainder.is_certified
        )

    @property
    def evidence(self) -> Evidence:
        """Return this perturbation certificate in the package-wide evidence model."""
        obligations = () if self.formal.verified else ("formal residual replay failed",)
        if self.has_rigorous_bound:
            return Evidence(
                EvidenceStatus.PROVED,
                "perturbation expansion with certified remainder",
                obligations=obligations,
            )
        return Evidence(
            EvidenceStatus.FORMAL if self.formal.verified else EvidenceStatus.UNKNOWN,
            "formal perturbation hierarchy",
            obligations=obligations + ("no certified asymptotic remainder",),
        )


@dataclass(frozen=True)
class PerturbationExpansionResult:
    """Unified result from :func:`perturbation_expansion`.

    ``result`` is ``None`` when automatic dispatch finds no applicable method
    or when several inequivalent methods tie for the best rank.  The complete
    ranking remains available through ``diagnostics`` so callers can make the
    method choice explicitly.
    """

    diagnostics: PerturbationDiagnostics
    selected_method: PerturbationMethod | None
    result: PerturbationMethodResult | None
    ambiguity: bool = False
    limitation: str = ""

    @property
    def complete(self) -> bool:
        """Whether a method was selected and its method-specific result is complete."""
        return self.result is not None and bool(getattr(self.result, "complete", False))


def perturbation_diagnostics(
    equations: EquationLike | Sequence[EquationLike],
    unknowns: sp.Expr | Sequence[sp.Expr],
    parameter: sp.Symbol,
    *,
    conditions: EquationLike | Sequence[EquationLike] = (),
    assumptions: sp.Expr = sp.S.true,
) -> PerturbationDiagnostics:
    """Classify and rank the perturbation methods structurally applicable to a problem.

    Rankings express method suitability, not mathematical equivalence.  A tie
    at the best rank means automatic dispatch must not select a method.
    """
    assumptions = normalize_assumptions(assumptions)
    parameter = sp.sympify(parameter)
    equation_tuple = tuple(
        sp.refine(_as_residual(item), assumptions) for item in _as_tuple(equations)
    )
    unknown_tuple = tuple(map(sp.sympify, _as_tuple(unknowns)))
    condition_tuple = (
        tuple(_as_residual(item) for item in _as_tuple(conditions))
        if conditions != ()
        else ()
    )
    variable = _shared_variable(unknown_tuple)
    full_order = _differential_order(equation_tuple, unknown_tuple)
    reduced = tuple(sp.expand(item.subs(parameter, 0)) for item in equation_tuple)
    reduced_order = _differential_order(reduced, unknown_tuple)

    assessments: list[PerturbationMethodAssessment] = []

    if all(isinstance(item, sp.Symbol) for item in unknown_tuple):
        regular_ok = True
        regular_reason = (
            "algebraic perturbation hierarchy has no differential order loss"
        )
        regular_rank = 1
    elif variable is not None and full_order == reduced_order:
        regular_ok = True
        regular_reason = "differential order is preserved in the reduced problem"
        regular_rank = 3 if full_order == 2 else 1
    elif variable is not None and full_order > reduced_order:
        regular_ok = False
        regular_reason = "the reduced equation loses differential order"
        regular_rank = 9
    else:
        regular_ok = False
        regular_reason = (
            "unknowns are not a supported algebraic or common-variable ODE family"
        )
        regular_rank = 9
    assessments.append(
        PerturbationMethodAssessment(
            "regular", regular_ok, regular_rank, regular_reason
        )
    )

    lindstedt_ok = False
    lindstedt_reason = "requires one autonomous scalar second-order oscillator"
    if (
        variable is not None
        and len(unknown_tuple) == len(equation_tuple) == 1
        and full_order == reduced_order == 2
    ):
        unknown = unknown_tuple[0]
        unperturbed = reduced[0]
        autonomous = not _has_explicit_variable(unperturbed, unknown, variable)
        frequency = _natural_frequency(equation_tuple[0], unknown, variable, parameter)
        lindstedt_ok = autonomous and frequency is not None
        if lindstedt_ok:
            lindstedt_reason = "undamped autonomous base oscillator admits strained-time frequency correction"
        elif not autonomous:
            lindstedt_reason = (
                "the unperturbed equation has explicit fast-time dependence"
            )
        else:
            lindstedt_reason = "no nonzero undamped base frequency could be inferred"
    assessments.append(
        PerturbationMethodAssessment(
            "lindstedt", lindstedt_ok, 1 if lindstedt_ok else 9, lindstedt_reason
        )
    )

    multiple_ok = False
    multiple_reason = "requires a common-variable oscillator system with inferable leading frequencies"
    if (
        variable is not None
        and len(equation_tuple) == len(unknown_tuple)
        and full_order == reduced_order == 2
    ):
        frequencies = tuple(
            _base_frequency(equation, unknown, variable, parameter)
            for equation, unknown in zip(equation_tuple, unknown_tuple)
        )
        multiple_ok = all(value is not None for value in frequencies)
        if multiple_ok:
            multiple_reason = (
                "leading oscillator frequencies support slow modulation analysis"
            )
    assessments.append(
        PerturbationMethodAssessment(
            "multiple-scales", multiple_ok, 1 if multiple_ok else 9, multiple_reason
        )
    )

    matched_ok = (
        variable is not None
        and len(unknown_tuple) == len(equation_tuple) == 1
        and full_order == 2
        and reduced_order == 1
        and len(condition_tuple) == 2
    )
    matched_reason = (
        "second-order scalar BVP loses one differential order and has two boundary conditions"
        if matched_ok
        else "requires a scalar second-order BVP whose reduced equation is first order"
    )
    assessments.append(
        PerturbationMethodAssessment(
            "matched", matched_ok, 1 if matched_ok else 9, matched_reason
        )
    )
    return PerturbationDiagnostics(tuple(assessments))


def _dispatch(
    method: PerturbationMethod,
    equations: EquationLike | Sequence[EquationLike],
    unknowns: sp.Expr | Sequence[sp.Expr],
    parameter: sp.Symbol,
    *,
    order: int,
    gauges: Sequence[sp.Expr] | None,
    conditions: EquationLike | Sequence[EquationLike],
    assumptions: sp.Expr,
    base_frequency: sp.Expr | None,
    base_frequencies: Sequence[sp.Expr] | sp.Expr | None,
) -> PerturbationMethodResult:
    if method == "regular":
        return regular_perturbation(
            equations,
            unknowns,
            parameter,
            order=None if gauges is not None else order,
            gauges=gauges,
            conditions=conditions,
            assumptions=assumptions,
        )
    if gauges is not None:
        raise ValueError(f"custom gauges are not accepted by the {method} dispatcher")
    if method == "lindstedt":
        equation_tuple = _as_tuple(equations)
        unknown_tuple = _as_tuple(unknowns)
        if len(equation_tuple) != 1 or len(unknown_tuple) != 1:
            raise ValueError("Lindstedt-Poincare requires one equation and one unknown")
        return lindstedt_poincare(
            equation_tuple[0],
            unknown_tuple[0],
            parameter,
            order=order,
            conditions=conditions,
            base_frequency=base_frequency,
            assumptions=assumptions,
        )
    if method == "multiple-scales":
        if order < 1:
            raise ValueError("multiple scales requires order >= 1")
        frequency_data = (
            base_frequencies if base_frequencies is not None else base_frequency
        )
        return multiple_scales(
            equations,
            unknowns,
            parameter,
            order=order,
            base_frequencies=frequency_data,
            assumptions=assumptions,
        )
    if method == "matched":
        equation_tuple = _as_tuple(equations)
        unknown_tuple = _as_tuple(unknowns)
        if len(equation_tuple) != 1 or len(unknown_tuple) != 1:
            raise ValueError(
                "matched asymptotics requires one equation and one unknown"
            )
        return matched_asymptotic_expansion(
            equation_tuple[0],
            unknown_tuple[0],
            parameter,
            conditions=_as_tuple(conditions),
            order=order,
            assumptions=assumptions,
        )
    raise ValueError(f"unknown perturbation method: {method}")


def perturbation_expansion(
    equations: EquationLike | Sequence[EquationLike],
    unknowns: sp.Expr | Sequence[sp.Expr],
    parameter: sp.Symbol,
    *,
    method: PerturbationMethod | Literal["auto"] = "auto",
    order: int = 1,
    gauges: Sequence[sp.Expr] | None = None,
    conditions: EquationLike | Sequence[EquationLike] = (),
    assumptions: sp.Expr = sp.S.true,
    base_frequency: sp.Expr | None = None,
    base_frequencies: Sequence[sp.Expr] | sp.Expr | None = None,
) -> PerturbationExpansionResult:
    """Dispatch a perturbation problem without choosing inequivalent methods.

    With ``method="auto"``, structural applicability is ranked first.  The
    solver runs only when one method has the unique best rank.  Tied preferred
    methods produce an ambiguity result with ``result=None``.  Supplying a
    method name always invokes that method directly after diagnostics are
    recorded.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    diagnostics = perturbation_diagnostics(
        equations, unknowns, parameter, conditions=conditions, assumptions=assumptions
    )
    if method == "auto":
        preferred = diagnostics.preferred
        if not preferred:
            return PerturbationExpansionResult(
                diagnostics,
                None,
                None,
                False,
                "no implemented perturbation method is structurally applicable",
            )
        if len(preferred) != 1:
            names = ", ".join(item.method for item in preferred)
            return PerturbationExpansionResult(
                diagnostics,
                None,
                None,
                True,
                f"multiple inequivalent methods share the best rank: {names}",
            )
        selected = preferred[0].method
    else:
        selected = method
        known = {item.method for item in diagnostics.assessments}
        if selected not in known:
            raise ValueError(f"unknown perturbation method: {selected}")
    result = _dispatch(
        selected,
        equations,
        unknowns,
        parameter,
        order=order,
        gauges=gauges,
        conditions=conditions,
        assumptions=assumptions,
        base_frequency=base_frequency,
        base_frequencies=base_frequencies,
    )
    return PerturbationExpansionResult(diagnostics, selected, result)


def _clear_from(hierarchy: PerturbationHierarchy, index: int) -> PerturbationHierarchy:
    orders = tuple(
        replace(
            order,
            solution=() if order.index >= index else order.solution,
            solvability_conditions=(
                () if order.index > index else order.solvability_conditions
            ),
        )
        for order in hierarchy.orders
    )
    return replace(hierarchy, orders=orders)


def _resume_regular(
    branch: RegularPerturbationBranch,
    index: int,
    solution: dict[sp.Expr, sp.Expr] | Sequence[sp.Expr] | sp.Expr,
) -> RegularPerturbationResult:
    hierarchy = _clear_from(branch.hierarchy, index).with_solution(index, solution)
    active = (hierarchy,)
    partial: list[RegularPerturbationBranch] = []
    for next_index in range(index + 1, len(hierarchy.orders)):
        next_active: list[PerturbationHierarchy] = []
        for current in active:
            advanced = _advance_branch(current, next_index)
            if advanced is None:
                partial.append(
                    RegularPerturbationBranch(
                        current,
                        False,
                        next_index,
                        f"could not solve perturbation order {next_index}",
                    )
                )
            else:
                next_active.extend(advanced)
        active = tuple(next_active)
        if not active:
            break
    complete = tuple(RegularPerturbationBranch(item, True) for item in active)
    return RegularPerturbationResult(
        tuple(partial) + complete,
        hierarchy.parameter,
        hierarchy.unknowns,
        hierarchy.gauges,
    )


def resume_perturbation(
    result: (
        PerturbationExpansionResult
        | RegularPerturbationResult
        | RegularPerturbationBranch
        | LindstedtPoincareResult
        | PerturbationHierarchy
    ),
    order: int,
    solution: dict[sp.Expr, sp.Expr] | Sequence[sp.Expr] | sp.Expr,
    *,
    branch: int = 0,
) -> RegularPerturbationResult | LindstedtPoincareResult:
    """Supply or replace one perturbation order and continue its solving policy.

    Regular hierarchies recursively recompute all later coefficient problems.
    Lindstedt hierarchies likewise discard later stored data and re-derive the
    subsequent Fredholm solvability conditions and frequency corrections.  Raw
    :class:`PerturbationHierarchy` objects are interpreted as regular problems.
    """
    subject = result
    if isinstance(subject, PerturbationExpansionResult):
        if subject.result is None:
            raise ValueError("there is no selected perturbation result to resume")
        subject = subject.result
    if isinstance(subject, LindstedtPoincareResult):
        hierarchy = _clear_from(subject.hierarchy, order).with_solution(order, solution)
        return _continue_lindstedt_hierarchy(
            hierarchy,
            subject.original_unknown,
            subject.original_variable,
            subject.fast_variable,
            subject.frequency_symbol,
            order + 1,
        )
    if isinstance(subject, RegularPerturbationResult):
        if not 0 <= branch < len(subject.branches):
            raise IndexError("regular perturbation branch is outside the result")
        subject = subject.branches[branch]
    if isinstance(subject, PerturbationHierarchy):
        subject = RegularPerturbationBranch(subject, False)
    if not isinstance(subject, RegularPerturbationBranch):
        raise TypeError("result is not a resumable perturbation hierarchy")
    subject.hierarchy.order(order)
    return _resume_regular(subject, order, solution)


def _formal_from_regular(
    branch: RegularPerturbationBranch,
) -> FormalResidualCertificate:
    through = branch.hierarchy.solved_through
    if through is None:
        return FormalResidualCertificate("regular", None, False, (), ())
    return FormalResidualCertificate(
        "regular",
        through,
        branch.verified,
        branch.residual,
        branch.condition_residual,
    )


def certify_perturbation(
    result: PerturbationExpansionResult
    | PerturbationMethodResult
    | RegularPerturbationBranch,
    *,
    branch: int = 0,
    rigorous_remainder: Remainder | None = None,
) -> PerturbationCertification:
    """Certify formal perturbation residuals without conflating them with error bounds.

    A formal certificate replays the method-specific solved hierarchy or
    composite residual checks.  A rigorous remainder is attached only when an
    already certified :class:`Remainder` is supplied for the same
    small parameter at zero; no rigorous error theorem is inferred from formal
    coefficient cancellation alone.
    """
    subject = (
        result.result if isinstance(result, PerturbationExpansionResult) else result
    )
    if subject is None:
        raise ValueError("there is no selected perturbation result to certify")
    if isinstance(subject, RegularPerturbationResult):
        if not 0 <= branch < len(subject.branches):
            raise IndexError("regular perturbation branch is outside the result")
        formal = _formal_from_regular(subject.branches[branch])
        parameter = subject.parameter
    elif isinstance(subject, RegularPerturbationBranch):
        formal = _formal_from_regular(subject)
        parameter = subject.hierarchy.parameter
    elif isinstance(subject, LindstedtPoincareResult):
        through = subject.hierarchy.solved_through
        formal = FormalResidualCertificate(
            "lindstedt", through, subject.verified, (), ()
        )
        parameter = subject.hierarchy.parameter
    elif isinstance(subject, MultipleScalesResult):
        through = subject.hierarchy.solved_through
        flow_subs = {item.derivative: item.expression for item in subject.slow_flow}
        hierarchy_subs = (
            subject.hierarchy.substitutions(through) if through is not None else {}
        )
        replay: list[sp.Expr] = []
        if through is not None:
            supplied_unknowns = tuple(hierarchy_subs)
            supplied_values = tuple(hierarchy_subs.values())
            for order_data in subject.hierarchy.orders[: through + 1]:
                for expression in order_data.equations + order_data.conditions:
                    value = _substitute_unknowns(
                        expression, supplied_unknowns, supplied_values
                    )
                    replay.append(bounded_simplify(sp.expand_trig(value)))
        for order_data in subject.hierarchy.orders:
            for condition in order_data.solvability_conditions:
                value = _substitute_unknowns(
                    condition.expression,
                    tuple(hierarchy_subs),
                    tuple(hierarchy_subs.values()),
                )
                replay.append(
                    bounded_simplify(sp.expand_trig(value.subs(flow_subs).doit()))
                )
        formal = FormalResidualCertificate(
            "multiple-scales",
            through,
            bool(replay) and all(item == 0 for item in replay),
            tuple(replay),
            (),
        )
        parameter = subject.hierarchy.parameter
    elif isinstance(subject, MatchedAsymptoticResult):
        if not 0 <= branch < len(subject.branches):
            raise IndexError("matched-asymptotic branch is outside the result")
        selected = subject.branches[branch]
        formal = FormalResidualCertificate(
            "matched",
            selected.inner_hierarchy.solved_through,
            selected.verified,
            (selected.residual,),
            selected.condition_residuals,
        )
        parameter = subject.parameter
    else:
        raise TypeError("unsupported perturbation result type")

    if rigorous_remainder is not None:
        if rigorous_remainder.variable != parameter or rigorous_remainder.point != 0:
            raise ValueError(
                "rigorous remainder must use the perturbation parameter at zero"
            )
        if not rigorous_remainder.is_certified:
            raise ValueError(
                "an unknown remainder is not a rigorous perturbation bound"
            )
    return PerturbationCertification(formal, rigorous_remainder)


__all__ = [
    "FormalResidualCertificate",
    "PerturbationCertification",
    "PerturbationDiagnostics",
    "PerturbationExpansionResult",
    "PerturbationMethodAssessment",
    "certify_perturbation",
    "perturbation_diagnostics",
    "perturbation_expansion",
    "resume_perturbation",
]
