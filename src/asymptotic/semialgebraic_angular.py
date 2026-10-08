"""Exact semialgebraic images on exceptional divisors."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .coverage import CoverageCertificate


@dataclass(frozen=True)
class SemialgebraicAngularImage:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    domain: sp.Expr
    image: sp.Set | None
    coverage: CoverageCertificate
    provider: str
    statement: str

    @property
    def certified(self):
        return self.image is not None and self.coverage.certified

    @property
    def minimum(self):
        return self.image.inf if self.certified else None

    @property
    def maximum(self):
        return self.image.sup if self.certified else None


def _rational_graph(expression, variables, value):
    num, den = map(sp.expand, sp.fraction(sp.cancel(sp.together(expression))))
    try:
        sp.Poly(num, *variables)
        sp.Poly(den, *variables)
    except sp.PolynomialError:
        return None
    clauses = [sp.Eq(value * den - num, 0)]
    if den != 1:
        clauses.append(sp.Ne(den, 0))
    return sp.And(*clauses)


def _formula_to_real_set(formula, value):
    formula = sp.simplify(formula)
    if formula is sp.S.true:
        return sp.S.Reals
    if formula is sp.S.false:
        return sp.S.EmptySet
    try:
        return formula.as_set()
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        try:
            return sp.solve_univariate_inequality(formula, value, relational=False)
        except (NotImplementedError, TypeError, ValueError):
            return None


def semialgebraic_angular_image(expression, variables, domain=sp.S.true):
    """Compute the exact real image of an angular semialgebraic map."""
    expression = sp.sympify(expression)
    variables = tuple(variables)
    domain = sp.sympify(domain)
    try:
        from semialg import function_range

        result = function_range(
            expression, domain, variables, method="auto", return_result=True
        )
        formula = result.formula
        z = result.value_symbol
        try:
            image = formula.as_set()
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            image = _formula_to_real_set(formula, z)
        if image is not None:
            return SemialgebraicAngularImage(
                expression,
                variables,
                domain,
                image,
                CoverageCertificate.complete(
                    "semialgebraic_function_range",
                    "exact semialgebraic function-range computation covers the exceptional divisor",
                    (domain,),
                ),
                getattr(result, "method", "semialgebraic_function_range"),
                "complete semialgebraic exceptional-divisor image",
            )
    except (
        ImportError,
        AttributeError,
        ArithmeticError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
        sp.PolynomialError,
    ):
        pass
    return SemialgebraicAngularImage(
        expression,
        variables,
        domain,
        None,
        CoverageCertificate.unknown(
            "semialgebraic_angular_image",
            "semialg could not certify the complete angular image",
        ),
        "none",
        "angular image remains unresolved",
    )


__all__ = ["SemialgebraicAngularImage", "semialgebraic_angular_image"]
