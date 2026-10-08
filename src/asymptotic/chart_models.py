"""Shared local-coordinate chart protocol and data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import sympy as sp


@runtime_checkable
class LocalChart(Protocol):
    """Structural protocol implemented by local coordinate charts."""

    @property
    def substitution(self) -> dict[sp.Symbol, sp.Expr]: ...

    def transform(self, expression): ...

    def pullback_domain(self, domain): ...

    @property
    def exceptional_domain(self): ...


@dataclass(frozen=True)
class CoordinateChart:
    """General local coordinate chart used by limit and geometry providers."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    domain: sp.Expr
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...]
    provider: str = "coordinate_chart"

    @property
    def substitution(self):
        return dict(self.substitutions)

    def transform(self, expression):
        return sp.cancel(
            sp.together(
                sp.sympify(expression).subs(self.substitution, simultaneous=True)
            )
        )

    def pullback_domain(self, domain):
        return sp.simplify(
            sp.sympify(domain).subs(self.substitution, simultaneous=True)
        )

    @property
    def exceptional_domain(self):
        return self.domain


def chart_transform(chart: LocalChart, expression):
    return chart.transform(expression)


def chart_pullback_domain(chart: LocalChart, domain):
    return chart.pullback_domain(domain)


def chart_exceptional_domain(chart: LocalChart):
    return chart.exceptional_domain
