"""Asymptotic products reduced to certified asymptotic sums of logarithms."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask
from .sums import SumResult, sum


@dataclass(frozen=True)
class ProductResult:
    """Asymptotic description of a parameter-dependent discrete product."""

    expression: sp.Expr
    method: str
    certified: bool
    logarithmic_sum: SumResult | None = None


def product(
    factor: sp.Expr,
    variable: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    *,
    parameter: sp.Symbol,
    point: sp.Expr = sp.oo,
    terms: int = 4,
    assumptions: sp.Expr = sp.S.true,
) -> ProductResult:
    """Expand a positive real product through ``exp(sum(log(factor)))``.

    Certification is inherited from the asymptotic-sum proof and additionally
    requires positivity of the factor under the supplied assumptions. Products
    with unresolved sign or branch behavior are returned as uncertified.
    """
    factor = sp.sympify(factor)
    positive = bounded_ask(sp.Q.positive(factor), assumptions)
    if positive is not True:
        return ProductResult(
            sp.Product(factor, (variable, lower, upper)), "unknown", False
        )
    summed = sum(
        sp.log(factor),
        variable,
        lower,
        upper,
        parameter=parameter,
        point=point,
        terms=terms,
        assumptions=assumptions,
    )
    expression = sp.exp(summed.truncate(terms))
    return ProductResult(
        expression,
        "logarithmic-sum",
        summed.certified,
        summed,
    )
