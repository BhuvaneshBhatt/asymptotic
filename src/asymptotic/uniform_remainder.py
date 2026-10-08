"""Uniform multivariate remainder certificates and conservative calculus."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .coverage import CoverageCertificate
from .proof_obligations import ObligationKind, ProofObligation


class UniformRemainderKind(Enum):
    EXACT = "exact"
    LITTLE_O = "little_o"
    BIG_O = "big_o"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UniformRemainder:
    """A uniform asymptotic bound on a local chart/domain.

    For BIG_O this represents ``|R| <= C |scale|`` uniformly for all
    ``uniform_variables`` satisfying ``domain`` and ``0 < radius <= radius_bound``.
    LITTLE_O additionally records a certified uniform little-o statement.
    """

    radius: sp.Symbol
    kind: UniformRemainderKind
    scale: sp.Expr | None
    constant: sp.Expr | None
    radius_bound: sp.Expr | None
    uniform_variables: tuple[sp.Symbol, ...] = ()
    domain: sp.Expr = sp.S.true
    exact_expression: sp.Expr | None = None
    provider: str = "uniform_remainder"
    statement: str = ""
    coverage: CoverageCertificate | None = None
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def certified(self) -> bool:
        return (
            self.kind is not UniformRemainderKind.UNKNOWN
            and not self.obligations
            and (self.coverage is None or self.coverage.certified)
        )

    @classmethod
    def exact_zero(
        cls, radius, *, uniform_variables=(), domain=sp.S.true, provider="exact"
    ):
        return cls(
            radius,
            UniformRemainderKind.EXACT,
            sp.S.Zero,
            sp.S.Zero,
            None,
            tuple(uniform_variables),
            domain,
            sp.S.Zero,
            provider,
            "exact zero remainder",
        )

    @classmethod
    def big_o(
        cls,
        scale,
        radius,
        *,
        constant,
        radius_bound,
        uniform_variables=(),
        domain=sp.S.true,
        exact_expression=None,
        provider="uniform_big_o",
        statement="uniform big-O bound",
        coverage=None,
    ):
        return cls(
            radius,
            UniformRemainderKind.BIG_O,
            sp.sympify(scale),
            sp.sympify(constant),
            sp.sympify(radius_bound),
            tuple(uniform_variables),
            sp.sympify(domain),
            None if exact_expression is None else sp.sympify(exact_expression),
            provider,
            statement,
            coverage,
        )

    @classmethod
    def unknown(
        cls,
        radius,
        *,
        uniform_variables=(),
        domain=sp.S.true,
        provider="uniform_remainder",
        statement="uniform bound not certified",
    ):
        obligation = ProofObligation(
            ObligationKind.THEOREM_PREREQUISITE, statement, provider=provider
        )
        return cls(
            radius,
            UniformRemainderKind.UNKNOWN,
            None,
            None,
            None,
            tuple(uniform_variables),
            sp.sympify(domain),
            None,
            provider,
            statement,
            obligations=(obligation,),
        )

    def _compatible(self, other):
        return (
            self.radius == other.radius
            and self.uniform_variables == other.uniform_variables
            and sp.simplify(self.domain ^ other.domain) is sp.S.false
        )

    def scale_by(self, factor):
        factor = sp.sympify(factor)
        if self.kind is UniformRemainderKind.EXACT or factor == 0:
            return self.exact_zero(
                self.radius,
                uniform_variables=self.uniform_variables,
                domain=self.domain,
            )
        if not self.certified or factor.free_symbols.intersection(
            self.uniform_variables
        ):
            return self.unknown(
                self.radius,
                uniform_variables=self.uniform_variables,
                domain=self.domain,
                statement="uniform scaling factor requires a certified chartwise bound",
            )
        return UniformRemainder(
            self.radius,
            self.kind,
            sp.simplify(factor * self.scale),
            sp.simplify(sp.Abs(factor) * self.constant),
            self.radius_bound,
            self.uniform_variables,
            self.domain,
            None
            if self.exact_expression is None
            else sp.simplify(factor * self.exact_expression),
            "uniform_scaling",
            "uniform remainder scaled by a parameter-independent exact factor",
            self.coverage,
        )

    def add(self, other):
        if not self._compatible(other) or not self.certified or not other.certified:
            return self.unknown(
                self.radius,
                uniform_variables=self.uniform_variables,
                domain=self.domain,
                statement="sum requires compatible certified uniform domains",
            )
        if self.kind is UniformRemainderKind.EXACT:
            return other
        if other.kind is UniformRemainderKind.EXACT:
            return self
        # No scale-comparison theorem is assumed: |R1+R2| <= C1|s1|+C2|s2|.
        scale = sp.simplify(
            self.constant * sp.Abs(self.scale) + other.constant * sp.Abs(other.scale)
        )
        return UniformRemainder.big_o(
            scale,
            self.radius,
            constant=1,
            radius_bound=sp.Min(self.radius_bound, other.radius_bound),
            uniform_variables=self.uniform_variables,
            domain=self.domain,
            provider="uniform_sum",
            statement="triangle-inequality uniform sum bound",
        )

    def product(self, other):
        if not self._compatible(other) or not self.certified or not other.certified:
            return self.unknown(
                self.radius,
                uniform_variables=self.uniform_variables,
                domain=self.domain,
                statement="product requires compatible certified uniform domains",
            )
        if (
            self.kind is UniformRemainderKind.EXACT
            or other.kind is UniformRemainderKind.EXACT
        ):
            return self.exact_zero(
                self.radius,
                uniform_variables=self.uniform_variables,
                domain=self.domain,
            )
        kind = (
            UniformRemainderKind.LITTLE_O
            if UniformRemainderKind.LITTLE_O in (self.kind, other.kind)
            else UniformRemainderKind.BIG_O
        )
        return UniformRemainder(
            self.radius,
            kind,
            sp.simplify(self.scale * other.scale),
            sp.simplify(self.constant * other.constant),
            sp.Min(self.radius_bound, other.radius_bound),
            self.uniform_variables,
            self.domain,
            provider="uniform_product",
            statement="product of uniform remainder bounds",
        )

    def substitute(self, mapping, new_radius):
        mapping = sp.sympify(mapping)
        if mapping.free_symbols.intersection(self.uniform_variables):
            return self.unknown(
                new_radius,
                uniform_variables=self.uniform_variables,
                domain=self.domain,
                statement="chart substitution mixes radial and uniform variables",
            )
        if self.kind is UniformRemainderKind.UNKNOWN:
            return self.unknown(
                new_radius, uniform_variables=self.uniform_variables, domain=self.domain
            )
        return UniformRemainder(
            new_radius,
            self.kind,
            None if self.scale is None else self.scale.subs(self.radius, mapping),
            self.constant,
            self.radius_bound,
            self.uniform_variables,
            self.domain,
            None
            if self.exact_expression is None
            else self.exact_expression.subs(self.radius, mapping),
            "uniform_substitution",
            "uniformity preserved by parameter-independent radial substitution",
            self.coverage,
        )


def from_weighted_certificate(
    certificate, radius, *, uniform_variables=(), domain=sp.S.true, coverage=None
):
    """Adapt a weighted-chart certificate to the common calculus."""
    if certificate is None:
        return UniformRemainder.unknown(
            radius,
            uniform_variables=uniform_variables,
            domain=domain,
            statement="weighted expansion has no uniform remainder certificate",
        )
    if certificate.constant == 0:
        return UniformRemainder.exact_zero(
            radius,
            uniform_variables=uniform_variables,
            domain=domain,
            provider=certificate.provider,
        )
    return UniformRemainder.big_o(
        radius**certificate.order,
        radius,
        constant=certificate.constant,
        radius_bound=certificate.radius,
        uniform_variables=uniform_variables,
        domain=domain,
        provider=certificate.provider,
        statement=certificate.statement,
        coverage=coverage,
    )
