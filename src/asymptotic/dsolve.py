"""High-level asymptotic solving for ordinary differential equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .nonlinear_ode import nonlinear_differential_transseries
from .ode_adapter import from_formal_ode_data


@dataclass(frozen=True)
class AsymptoticDSolveResult:
    """Structured result returned by :func:`asymptotic_dsolve`.

    ``solutions`` contains ordinary SymPy expressions for the finite prefixes.
    ``branches`` retains the richer transseries or nonlinear-lifting objects so
    callers can inspect certificates, residuals, monodromy metadata, and other
    structural information without forcing everything into one expression.
    """

    solutions: tuple[sp.Expr, ...]
    function: sp.FunctionClass
    variable: sp.Symbol
    point: sp.Expr
    status: str
    method: str
    branches: tuple[Any, ...]
    complete: bool
    limitation: str | None = None

    def residuals(self, equation: sp.Expr | sp.Equality) -> tuple[sp.Expr, ...]:
        """Return equation residuals after substituting each reported prefix."""

        expression = _equation_expression(equation)
        dependent = self.function(self.variable)
        return tuple(
            sp.expand(expression.subs(dependent, solution).doit()) for solution in self.solutions
        )


def _equation_expression(equation: sp.Expr | sp.Equality) -> sp.Expr:
    equation = sp.sympify(equation)
    if isinstance(equation, sp.Equality):
        return sp.expand(equation.lhs - equation.rhs)
    return sp.expand(equation)


def _formal_linear_route(
    equation: sp.Expr,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    *,
    point: sp.Expr,
    terms: int,
) -> AsymptoticDSolveResult | None:
    """Try the optional odeanalysis formal-data interface."""

    try:
        from odeanalysis import formal_ode_data
    except ImportError:
        return None

    try:
        data = formal_ode_data(
            equation,
            function,
            variable,
            point=point,
            terms=max(terms, 2),
            include_stokes=True,
        )
        converted = from_formal_ode_data(data, variable)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        return None

    branches = converted.solutions
    solutions = tuple(sp.sympify(branch.truncate()) for branch in branches)
    if not solutions:
        return None
    return AsymptoticDSolveResult(
        solutions=solutions,
        function=function,
        variable=variable,
        point=point,
        status="FORMAL",
        method="odeanalysis-formal-linear",
        branches=branches,
        complete=converted.complete,
        limitation=converted.limitation,
    )


def asymptotic_dsolve(
    equation: sp.Expr | sp.Equality,
    function: sp.FunctionClass,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    terms: int = 6,
    assumptions: sp.Expr | bool = sp.S.true,
    method: str = "auto",
) -> AsymptoticDSolveResult:
    """Solve an ODE asymptotically near a finite point or infinity.

    In ``auto`` mode linear equations are first sent through the stable
    :mod:`odeanalysis` formal-data interface, which can expose Frobenius,
    ramification, exponential blocks, monodromy, and Stokes metadata.  If that
    route does not apply, differential-polynomial nonlinear equations are
    handled by recursive Newton/transseries lifting.  The function is
    conservative: unsupported equations raise ``NotImplementedError`` rather
    than being mislabeled as complete asymptotic solutions.
    """

    if not isinstance(variable, sp.Symbol):
        raise TypeError("variable must be a Symbol")
    if not isinstance(function, sp.FunctionClass):
        raise TypeError("function must be an undefined SymPy Function")
    if terms < 1:
        raise ValueError("terms must be positive")
    if method not in {"auto", "linear", "nonlinear"}:
        raise ValueError("method must be 'auto', 'linear', or 'nonlinear'")

    expression = _equation_expression(equation)
    point = sp.sympify(point)

    if method in {"auto", "linear"}:
        linear = _formal_linear_route(expression, function, variable, point=point, terms=terms)
        if linear is not None:
            return linear
        if method == "linear":
            raise NotImplementedError("odeanalysis could not construct formal linear ODE data")

    try:
        lifted = nonlinear_differential_transseries(
            expression,
            function,
            variable,
            point=point,
            terms=terms,
            assumptions=assumptions,
            stratify_parameters=False,
        )
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError) as exc:
        raise NotImplementedError(
            "no supported asymptotic ODE route could solve this equation"
        ) from exc

    if not isinstance(lifted, tuple) or not lifted:
        raise NotImplementedError("nonlinear asymptotic lifting produced no branches")
    complete = all(branch.complete for branch in lifted)
    limitation = (
        None
        if complete
        else "; ".join(sorted({branch.limitation for branch in lifted if branch.limitation}))
        or "one or more nonlinear branches are incomplete"
    )
    return AsymptoticDSolveResult(
        solutions=tuple(sp.sympify(branch.series) for branch in lifted),
        function=function,
        variable=variable,
        point=point,
        status="FORMAL",
        method="nonlinear-differential-transseries",
        branches=lifted,
        complete=complete,
        limitation=limitation,
    )
