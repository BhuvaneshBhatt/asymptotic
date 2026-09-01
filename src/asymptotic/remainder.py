from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, GrowthComparison, context_for


class RemainderKind(Enum):
    """Semantic strength of an asymptotic remainder statement."""

    EXACT = "exact"
    LITTLE_O = "little_o"
    BIG_O = "big_o"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RemainderProvenance:
    """Why a remainder statement is believed/certified."""

    source: str
    note: str | None = None


@dataclass(frozen=True)
class AsymptoticRemainder:
    """Certified or explicitly unknown remainder attached to a finite prefix.

    ``kind`` describes the mathematical statement about the error ``R``:

    * ``EXACT``: ``R == 0``;
    * ``LITTLE_O``: ``R = o(scale)``;
    * ``BIG_O``: ``R = O(scale)``;
    * ``UNKNOWN``: no asymptotic bound has been certified.

    ``exact_expression`` may additionally store the exact represented error.
    This is useful when truncating a finite exact expression: the exact omitted
    tail is known even though its compact asymptotic description is normally
    only ``O(first_omitted_monomial)``.
    """

    variable: sp.Symbol
    point: sp.Expr
    kind: RemainderKind
    scale: sp.Expr | None = None
    exact_expression: sp.Expr | None = None
    provenance: tuple[RemainderProvenance, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", sp.sympify(self.point))
        if self.scale is not None:
            object.__setattr__(self, "scale", analytic_powsimp(sp.sympify(self.scale)))
        if self.exact_expression is not None:
            object.__setattr__(
                self,
                "exact_expression",
                analytic_powsimp(sp.expand(sp.sympify(self.exact_expression))),
            )
        if self.kind is RemainderKind.EXACT:
            if self.scale not in (None, 0, sp.S.Zero):
                raise ValueError("an exact-zero remainder cannot have a nonzero scale")
            if self.exact_expression not in (None, 0, sp.S.Zero):
                raise ValueError("an exact-zero remainder cannot have a nonzero exact expression")
        elif self.kind in (RemainderKind.BIG_O, RemainderKind.LITTLE_O):
            if self.scale is None or sp.sympify(self.scale).is_zero is True:
                raise ValueError("O/o remainders require a nonzero scale")

    @classmethod
    def exact_zero(
        cls, variable: sp.Symbol, point: sp.Expr, *, source: str = "exact finite representation"
    ) -> AsymptoticRemainder:
        return cls(
            variable,
            point,
            RemainderKind.EXACT,
            exact_expression=sp.S.Zero,
            provenance=(RemainderProvenance(source),),
        )

    @classmethod
    def unknown(
        cls,
        variable: sp.Symbol,
        point: sp.Expr,
        *,
        exact_expression: sp.Expr | None = None,
        source: str = "no certified asymptotic bound",
    ) -> AsymptoticRemainder:
        return cls(
            variable,
            point,
            RemainderKind.UNKNOWN,
            exact_expression=exact_expression,
            provenance=(RemainderProvenance(source),),
        )

    @classmethod
    def big_o(
        cls,
        scale: sp.Expr,
        variable: sp.Symbol,
        point: sp.Expr,
        *,
        exact_expression: sp.Expr | None = None,
        source: str = "certified big-O remainder",
    ) -> AsymptoticRemainder:
        return cls(
            variable,
            point,
            RemainderKind.BIG_O,
            scale=scale,
            exact_expression=exact_expression,
            provenance=(RemainderProvenance(source),),
        )

    @classmethod
    def little_o(
        cls,
        scale: sp.Expr,
        variable: sp.Symbol,
        point: sp.Expr,
        *,
        exact_expression: sp.Expr | None = None,
        source: str = "certified little-o remainder",
    ) -> AsymptoticRemainder:
        return cls(
            variable,
            point,
            RemainderKind.LITTLE_O,
            scale=scale,
            exact_expression=exact_expression,
            provenance=(RemainderProvenance(source),),
        )

    @property
    def is_exact(self) -> bool:
        return self.kind is RemainderKind.EXACT

    @property
    def is_certified(self) -> bool:
        return self.kind is not RemainderKind.UNKNOWN

    @property
    def notation(self) -> str:
        if self.kind is RemainderKind.EXACT:
            return "0"
        if self.kind is RemainderKind.UNKNOWN:
            return "unknown remainder"
        op = "o" if self.kind is RemainderKind.LITTLE_O else "O"
        return f"{op}({sp.sstr(self.scale)})"

    def with_provenance(self, source: str, note: str | None = None) -> AsymptoticRemainder:
        return AsymptoticRemainder(
            self.variable,
            self.point,
            self.kind,
            self.scale,
            self.exact_expression,
            self.provenance + (RemainderProvenance(source, note),),
        )

    def _check_compatible(self, other: AsymptoticRemainder) -> None:
        if self.variable != other.variable or self.point != other.point:
            raise ValueError("remainders use different variables or asymptotic points")

    def check(self, *, context: AsymptoticContext | None = None) -> bool | None:
        """Replay a remainder certificate when the exact error is available.

        This deliberately returns ``None`` when boundedness/limits cannot be
        certified symbolically rather than upgrading uncertainty to success.
        """

        if context is not None:
            context_for(self.variable, self.point, context)
        if self.kind is RemainderKind.EXACT:
            return True
        if self.kind is RemainderKind.UNKNOWN or self.exact_expression is None:
            return None
        ctx = context_for(self.variable, self.point, context)
        ratio = sp.simplify(self.exact_expression / sp.sympify(self.scale))
        try:
            lim = ctx.limit(sp.Abs(ratio))
        except SYMBOLIC_ERRORS:
            return None
        if self.kind is RemainderKind.LITTLE_O:
            if lim == 0:
                return True
            if lim in (sp.oo, -sp.oo, sp.zoo) or getattr(lim, "is_zero", None) is False:
                return False
            return None
        if lim in (sp.oo, -sp.oo, sp.zoo):
            return False
        if getattr(lim, "is_finite", None) is True:
            return True
        return None

    def scale_by(self, factor: sp.Expr) -> AsymptoticRemainder:
        """Multiply a remainder by a known exact factor."""

        factor = analytic_powsimp(sp.sympify(factor))
        exact = (
            None
            if self.exact_expression is None
            else analytic_powsimp(factor * self.exact_expression)
        )
        if factor.is_zero is True or self.kind is RemainderKind.EXACT:
            return AsymptoticRemainder.exact_zero(
                self.variable, self.point, source="scaled exact remainder"
            )
        if self.kind is RemainderKind.UNKNOWN:
            return AsymptoticRemainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="scaling preserves an unknown bound",
            )
        return AsymptoticRemainder(
            self.variable,
            self.point,
            self.kind,
            analytic_powsimp(factor * sp.sympify(self.scale)),
            exact,
            self.provenance + (RemainderProvenance("exact scaling"),),
        )

    def product(self, other: AsymptoticRemainder) -> AsymptoticRemainder:
        """Remainder product rule, e.g. ``o(a) O(b) = o(ab)``."""

        self._check_compatible(other)
        exact = (
            analytic_powsimp(self.exact_expression * other.exact_expression)
            if self.exact_expression is not None and other.exact_expression is not None
            else None
        )
        if self.is_exact or other.is_exact:
            return AsymptoticRemainder.exact_zero(
                self.variable, self.point, source="product with exact-zero remainder"
            )
        if self.kind is RemainderKind.UNKNOWN or other.kind is RemainderKind.UNKNOWN:
            return AsymptoticRemainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="product contains unknown remainder",
            )
        kind = (
            RemainderKind.LITTLE_O
            if RemainderKind.LITTLE_O in (self.kind, other.kind)
            else RemainderKind.BIG_O
        )
        return AsymptoticRemainder(
            self.variable,
            self.point,
            kind,
            analytic_powsimp(sp.sympify(self.scale) * sp.sympify(other.scale)),
            exact,
            self.provenance + other.provenance + (RemainderProvenance("remainder product rule"),),
        )

    def add(
        self,
        other: AsymptoticRemainder,
        *,
        context: AsymptoticContext | None = None,
    ) -> AsymptoticRemainder:
        """Add remainder statements, retaining the strongest safe common bound."""

        self._check_compatible(other)
        exact = (
            analytic_powsimp(sp.expand(self.exact_expression + other.exact_expression))
            if self.exact_expression is not None and other.exact_expression is not None
            else None
        )
        if self.is_exact:
            if exact is None:
                return other
            return AsymptoticRemainder(
                other.variable,
                other.point,
                other.kind,
                other.scale,
                exact,
                self.provenance + other.provenance,
            )
        if other.is_exact:
            if exact is None:
                return self
            return AsymptoticRemainder(
                self.variable,
                self.point,
                self.kind,
                self.scale,
                exact,
                self.provenance + other.provenance,
            )
        if self.kind is RemainderKind.UNKNOWN or other.kind is RemainderKind.UNKNOWN:
            return AsymptoticRemainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="sum contains unknown remainder",
            )

        ctx = context_for(self.variable, self.point, context)
        relation, _ = ctx.compare_growth(
            sp.Abs(sp.sympify(self.scale)), sp.Abs(sp.sympify(other.scale))
        )
        if relation is GrowthComparison.LARGER:
            dominant = self
        elif relation is GrowthComparison.SMALLER:
            dominant = other
        elif relation is GrowthComparison.SAME_ORDER:
            dominant = self
        else:
            return AsymptoticRemainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="remainder scales are not comparably certified",
            )

        if relation is GrowthComparison.SAME_ORDER:
            kind = (
                RemainderKind.LITTLE_O
                if self.kind is other.kind is RemainderKind.LITTLE_O
                else RemainderKind.BIG_O
            )
        else:
            # A smaller O/o term is o(dominant scale). Hence the dominant
            # statement controls the sum without loss of strength.
            kind = dominant.kind
        return AsymptoticRemainder(
            self.variable,
            self.point,
            kind,
            dominant.scale,
            exact,
            self.provenance + other.provenance + (RemainderProvenance("remainder sum rule"),),
        )


@dataclass(frozen=True)
class AsymptoticTruncation:
    """A finite prefix together with its explicit remainder semantics."""

    prefix: sp.Expr
    remainder: AsymptoticRemainder
    terms_kept: int
    total_known_terms: int

    @property
    def statement(self) -> str:
        return f"{sp.sstr(self.prefix)} + {self.remainder.notation}"

    def reconstruct(self) -> sp.Expr | None:
        """Return the exact represented value when the exact error is known."""

        if self.remainder.exact_expression is None:
            return None
        return analytic_powsimp(sp.expand(self.prefix + self.remainder.exact_expression))
