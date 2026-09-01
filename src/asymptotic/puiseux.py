from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import sympy as sp

from ._integer_utils import integer_lcm as _lcm
from ._symbolic_errors import SYMBOLIC_ERRORS
from ._symbolic_policy import bounded_solve_one
from .dominant import dominant_balance_candidates, lift_dominant_balance_branches


@dataclass(frozen=True)
class BranchChoice:
    index: int
    label: str | None = None
    condition: sp.Expr | None = None


@dataclass(frozen=True)
class PuiseuxTerm:
    exponent: sp.Rational
    coefficient: sp.Expr

    def as_expr(self, variable: sp.Symbol) -> sp.Expr:
        return self.coefficient * variable**self.exponent


@dataclass(frozen=True)
class NewtonCandidate:
    exponent: sp.Rational
    coefficient_equation: sp.Expr
    coefficients: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class PuiseuxSeries:
    expr: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    terms: tuple[PuiseuxTerm, ...]
    ramification_index: int
    branch: BranchChoice | None = None

    @property
    def leading_term(self) -> PuiseuxTerm | None:
        return self.terms[0] if self.terms else None

    def truncate(self, n: int | None = None) -> sp.Expr:
        selected = self.terms if n is None else self.terms[:n]
        return sp.Add(*(t.as_expr(self.variable) for t in selected))

    def differentiate(self, order: int = 1) -> PuiseuxSeries:
        expr = sp.diff(self.truncate(), self.variable, order)
        return puiseux_series(expr, self.variable, point=self.point, terms=max(1, len(self.terms)))

    def asymptotic_element(self):
        """View this Puiseux series through the common asymptotic algebra."""
        from .algebra import asymptotic_element

        return asymptotic_element(self, self.variable, point=self.point)


@dataclass(frozen=True)
class AlgebraicBranch:
    polynomial: sp.Expr
    dependent: sp.Symbol
    variable: sp.Symbol
    exact_root: sp.Expr | None
    series: PuiseuxSeries
    newton_exponent: sp.Rational | None
    newton_coefficient: sp.Expr | None
    choice: BranchChoice


def _extract_puiseux_terms(
    expr: sp.Expr,
    x: sp.Symbol,
    *,
    error_context: str = "Puiseux",
    constant_coefficients: bool = False,
) -> tuple[PuiseuxTerm, ...]:
    """Extract rational-power terms with one shared grouping implementation."""

    expanded = sp.expand(expr.removeO() if hasattr(expr, "removeO") else expr)
    grouped: dict[sp.Rational, sp.Expr] = {}
    for term in sp.Add.make_args(expanded):
        exponent = sp.sympify(term.as_powers_dict().get(x, 0))
        if not exponent.is_Rational:
            raise NotImplementedError(f"non-rational {error_context} exponent {exponent}")
        exponent = sp.Rational(exponent)
        coefficient = sp.simplify(term / x**exponent)
        if constant_coefficients and x in coefficient.free_symbols:
            raise NotImplementedError(f"non-Puiseux {error_context} term {term}")
        grouped[exponent] = sp.simplify(grouped.get(exponent, 0) + coefficient)
    return tuple(
        PuiseuxTerm(exponent, grouped[exponent])
        for exponent in sorted(grouped)
        if grouped[exponent] != 0
    )


def puiseux_series(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    terms: int = 6,
    branch: BranchChoice | None = None,
) -> PuiseuxSeries:
    """Construct a rational-exponent local series with explicit ramification."""

    expr = sp.sympify(expr)
    if point in (sp.oo, -sp.oo):
        z = sp.Dummy("z", positive=True)
        sign = 1 if point is sp.oo else -1
        local = expr.xreplace({variable: sign / z})
        raw = sp.series(local, z, 0, terms).removeO().xreplace({z: sign / variable})
    else:
        raw = sp.series(expr, variable, point, terms).removeO()
        if point != 0:
            # Represent in powers of (x-point) using a local dummy, then map
            # back. PuiseuxTerm is intentionally for zero/infinity coordinates;
            # nonzero centers return the shifted expression as coefficients.
            u = sp.Dummy("u")
            local = sp.series(expr.xreplace({variable: u + point}), u, 0, terms).removeO()
            local_terms = _extract_puiseux_terms(local, u)
            den = reduce(_lcm, (int(t.exponent.q) for t in local_terms), 1)
            shifted = tuple(PuiseuxTerm(t.exponent, t.coefficient) for t in local_terms)
            return PuiseuxSeries(expr, variable - point, point, shifted, den, branch)
    extracted = _extract_puiseux_terms(raw, variable)
    ramification = reduce(_lcm, (int(t.exponent.q) for t in extracted), 1)
    return PuiseuxSeries(expr, variable, point, extracted, ramification, branch)


def _valuation_and_lead(expr: sp.Expr, x: sp.Symbol) -> tuple[sp.Rational, sp.Expr] | None:
    if expr == 0:
        return None
    try:
        lead = sp.expand(expr).as_leading_term(x)
        powers = lead.as_powers_dict()
        exponent = sp.sympify(powers.get(x, 0))
        if not exponent.is_Rational:
            return None
        coeff = sp.simplify(lead / x**exponent)
        return sp.Rational(exponent), coeff
    except SYMBOLIC_ERRORS:
        return None


def newton_polygon_candidates(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
) -> tuple[NewtonCandidate, ...]:
    """Compute leading Puiseux balances using the shared dominant-balance engine."""

    shared = dominant_balance_candidates(polynomial, dependent, variable)
    return tuple(
        NewtonCandidate(
            exponent=item.exponent,
            coefficient_equation=item.coefficient_equation,
            coefficients=item.coefficients,
        )
        for item in shared
    )


def algebraic_branches(
    polynomial: sp.Expr,
    dependent: sp.Symbol,
    variable: sp.Symbol,
    *,
    point: sp.Expr = 0,
    terms: int = 6,
) -> tuple[AlgebraicBranch, ...]:
    """Construct Puiseux branches with Newton-polygon leading balances.

    Explicit radicals are used when available because they give exact branch
    identities.  Otherwise, at zero, simple nonzero Newton-edge roots are
    lifted coefficient-by-coefficient in a ramified coordinate, so arbitrary
    polynomial degree does not require an explicit radical formula.
    """

    polynomial = sp.expand(sp.sympify(polynomial))
    candidates = newton_polygon_candidates(polynomial, dependent, variable) if point == 0 else ()
    explicit_roots = bounded_solve_one(polynomial, dependent, allow_general=True) or ()

    if explicit_roots:
        branches = []
        for index, root in enumerate(explicit_roots):
            choice = BranchChoice(index=index, label=f"branch-{index}")
            series = puiseux_series(root, variable, point=point, terms=terms, branch=choice)
            nr = None
            nc = None
            lead = series.leading_term
            if lead is not None:
                for candidate in candidates:
                    if candidate.exponent == lead.exponent and any(
                        sp.simplify(lead.coefficient - c) == 0 for c in candidate.coefficients
                    ):
                        nr = candidate.exponent
                        nc = lead.coefficient
                        break
            branches.append(
                AlgebraicBranch(polynomial, dependent, variable, root, series, nr, nc, choice)
            )
        return tuple(branches)

    if point != 0 or not candidates:
        raise NotImplementedError("Could not isolate an algebraic branch at this expansion point")

    lifted = lift_dominant_balance_branches(polynomial, dependent, variable, terms=terms)
    branches = []
    for index, item in enumerate(lifted):
        if not item.path or not item.leading_coefficients:
            continue
        formal_terms = _extract_puiseux_terms(item.series, variable)[:terms]
        ramification = reduce(_lcm, (int(t.exponent.q) for t in formal_terms), 1)
        choice = BranchChoice(index=index, label=f"branch-{index}")
        series = PuiseuxSeries(item.series, variable, sp.S.Zero, formal_terms, ramification, choice)
        branches.append(
            AlgebraicBranch(
                polynomial,
                dependent,
                variable,
                None,
                series,
                item.path[0].exponent,
                item.leading_coefficients[0],
                choice,
            )
        )
    if not branches:
        raise NotImplementedError("Newton polygon produced no liftable nonzero branches")
    return tuple(branches)
