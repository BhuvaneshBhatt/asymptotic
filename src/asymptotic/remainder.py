from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, AsymptoticGrowthComparison, context_for


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
class Remainder:
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
                raise ValueError(
                    "an exact-zero remainder cannot have a nonzero exact expression"
                )
        elif self.kind in (RemainderKind.BIG_O, RemainderKind.LITTLE_O):
            if self.scale is None or sp.sympify(self.scale).is_zero is True:
                raise ValueError("O/o remainders require a nonzero scale")

    @classmethod
    def exact_zero(
        cls,
        variable: sp.Symbol,
        point: sp.Expr,
        *,
        source: str = "exact finite representation",
    ) -> Remainder:
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
    ) -> Remainder:
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
    ) -> Remainder:
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
    ) -> Remainder:
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

    def with_provenance(self, source: str, note: str | None = None) -> Remainder:
        return Remainder(
            self.variable,
            self.point,
            self.kind,
            self.scale,
            self.exact_expression,
            self.provenance + (RemainderProvenance(source, note),),
        )

    def _check_compatible(self, other: Remainder) -> None:
        if self.variable != other.variable or self.point != other.point:
            raise ValueError("remainders use different variables or asymptotic points")

    def check(self, *, context: AsymptoticContext | None = None) -> bool | None:
        """Replay a remainder certificate when the exact error is available.

        This returns ``None`` when boundedness/limits cannot be
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

    def scale_by(self, factor: sp.Expr) -> Remainder:
        """Multiply a remainder by a known exact factor."""

        factor = analytic_powsimp(sp.sympify(factor))
        exact = (
            None
            if self.exact_expression is None
            else analytic_powsimp(factor * self.exact_expression)
        )
        if factor.is_zero is True or self.kind is RemainderKind.EXACT:
            return Remainder.exact_zero(
                self.variable, self.point, source="scaled exact remainder"
            )
        if self.kind is RemainderKind.UNKNOWN:
            return Remainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="scaling preserves an unknown bound",
            )
        return Remainder(
            self.variable,
            self.point,
            self.kind,
            analytic_powsimp(factor * sp.sympify(self.scale)),
            exact,
            self.provenance + (RemainderProvenance("exact scaling"),),
        )

    def product(self, other: Remainder) -> Remainder:
        """Remainder product rule, e.g. ``o(a) O(b) = o(ab)``."""

        self._check_compatible(other)
        exact = (
            analytic_powsimp(self.exact_expression * other.exact_expression)
            if self.exact_expression is not None and other.exact_expression is not None
            else None
        )
        if self.is_exact or other.is_exact:
            return Remainder.exact_zero(
                self.variable, self.point, source="product with exact-zero remainder"
            )
        if self.kind is RemainderKind.UNKNOWN or other.kind is RemainderKind.UNKNOWN:
            return Remainder.unknown(
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
        return Remainder(
            self.variable,
            self.point,
            kind,
            analytic_powsimp(sp.sympify(self.scale) * sp.sympify(other.scale)),
            exact,
            self.provenance
            + other.provenance
            + (RemainderProvenance("remainder product rule"),),
        )

    def add(
        self,
        other: Remainder,
        *,
        context: AsymptoticContext | None = None,
    ) -> Remainder:
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
            return Remainder(
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
            return Remainder(
                self.variable,
                self.point,
                self.kind,
                self.scale,
                exact,
                self.provenance + other.provenance,
            )
        if self.kind is RemainderKind.UNKNOWN or other.kind is RemainderKind.UNKNOWN:
            return Remainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="sum contains unknown remainder",
            )

        ctx = context_for(self.variable, self.point, context)
        relation, _ = ctx.compare_growth(
            sp.Abs(sp.sympify(self.scale)), sp.Abs(sp.sympify(other.scale))
        )
        if relation is AsymptoticGrowthComparison.LARGER:
            dominant = self
        elif relation is AsymptoticGrowthComparison.SMALLER:
            dominant = other
        elif relation is AsymptoticGrowthComparison.SAME_ORDER:
            dominant = self
        else:
            return Remainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="remainder scales are not comparably certified",
            )

        if relation is AsymptoticGrowthComparison.SAME_ORDER:
            kind = (
                RemainderKind.LITTLE_O
                if self.kind is other.kind is RemainderKind.LITTLE_O
                else RemainderKind.BIG_O
            )
        else:
            # A smaller O/o term is o(dominant scale). Hence the dominant
            # statement controls the sum without loss of strength.
            kind = dominant.kind
        return Remainder(
            self.variable,
            self.point,
            kind,
            dominant.scale,
            exact,
            self.provenance
            + other.provenance
            + (RemainderProvenance("remainder sum rule"),),
        )

    def _verified_transform(
        self,
        *,
        scale: sp.Expr | None,
        exact_expression: sp.Expr | None,
        source: str,
        context: AsymptoticContext | None = None,
        prefer_little_o: bool = False,
    ) -> Remainder:
        """Build a transformed certificate only when its bound replays."""
        if self.is_exact:
            return Remainder.exact_zero(self.variable, self.point, source=source)
        if scale is None or exact_expression is None:
            return Remainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact_expression,
                source=source,
            )
        kind = RemainderKind.LITTLE_O if prefer_little_o else RemainderKind.BIG_O
        candidate = Remainder(
            self.variable,
            self.point,
            kind,
            scale,
            exact_expression,
            self.provenance + (RemainderProvenance(source),),
        )
        if candidate.check(context=context) is True:
            return candidate
        return Remainder.unknown(
            self.variable,
            self.point,
            exact_expression=exact_expression,
            source=f"{source}: transformed bound could not be certified",
        )

    def compose_analytic(
        self,
        function: sp.FunctionClass | Callable[[sp.Expr], sp.Expr],
        prefix: sp.Expr,
        *,
        context: AsymptoticContext | None = None,
    ) -> Remainder:
        """Propagate through analytic composition, verifying the local bound."""
        if self.exact_expression is None:
            return Remainder.unknown(
                self.variable,
                self.point,
                source="analytic composition lacks exact represented error",
            )
        prefix = sp.sympify(prefix)
        exact = analytic_powsimp(
            function(prefix + self.exact_expression) - function(prefix)
        )
        if self.is_exact:
            return Remainder.exact_zero(
                self.variable,
                self.point,
                source="analytic composition of exact remainder",
            )
        if self.scale is None:
            return Remainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="analytic composition lacks input scale",
            )
        derivative = sp.diff(function(sp.Dummy("_z")), sp.Dummy("_z"))
        # Rebuild with one shared dummy: FunctionClass calls above can create expressions.
        z = sp.Dummy("_z")
        derivative = sp.diff(function(z), z).subs(z, prefix)
        scale = analytic_powsimp(derivative * self.scale)
        return self._verified_transform(
            scale=scale,
            exact_expression=exact,
            source="verified analytic composition",
            context=context,
            prefer_little_o=self.kind is RemainderKind.LITTLE_O,
        )

    def differentiate(self, *, context: AsymptoticContext | None = None) -> Remainder:
        """Differentiate a represented error; certify only after replay."""
        if self.exact_expression is None:
            return Remainder.unknown(
                self.variable,
                self.point,
                source="differentiation needs an exact represented error",
            )
        exact = analytic_powsimp(sp.diff(self.exact_expression, self.variable))
        if exact == 0:
            return Remainder.exact_zero(
                self.variable, self.point, source="exact differentiated remainder"
            )
        scale = (
            None
            if self.scale is None
            else analytic_powsimp(sp.diff(self.scale, self.variable))
        )
        if scale == 0:
            scale = None
        return self._verified_transform(
            scale=scale,
            exact_expression=exact,
            source="verified differentiated remainder",
            context=context,
            prefer_little_o=self.kind is RemainderKind.LITTLE_O,
        )

    def integrate(
        self,
        lower: sp.Expr,
        upper: sp.Expr,
        *,
        context: AsymptoticContext | None = None,
    ) -> Remainder:
        """Integrate a represented error over explicit bounds and replay its bound."""
        if self.exact_expression is None:
            return Remainder.unknown(
                self.variable,
                self.point,
                source="integration needs an exact represented error",
            )
        exact = sp.integrate(self.exact_expression, (self.variable, lower, upper))
        if isinstance(exact, sp.Integral):
            return Remainder.unknown(
                self.variable, self.point, source="remainder integral was not evaluated"
            )
        # A definite integral eliminates the asymptotic variable, so it is exact evidence.
        return (
            Remainder.exact_zero(
                self.variable,
                self.point,
                source="exact definite integration of represented error",
            )
            if exact == 0
            else Remainder.unknown(
                self.variable,
                self.point,
                exact_expression=exact,
                source="definite integration changes the asymptotic variable; no generic O-rule applied",
            )
        )

    def substitute(
        self,
        mapping: sp.Expr,
        new_variable: sp.Symbol,
        new_point: sp.Expr,
        *,
        context: AsymptoticContext | None = None,
    ) -> Remainder:
        """Pull a remainder back along a map known to approach the original germ."""
        mapping = sp.sympify(mapping)
        ctx = context_for(new_variable, new_point, context)
        if ctx.limit(mapping) != self.point:
            return Remainder.unknown(
                new_variable,
                new_point,
                source="substitution does not approach the certified germ",
            )
        exact = (
            None
            if self.exact_expression is None
            else self.exact_expression.subs(self.variable, mapping)
        )
        if self.is_exact:
            return Remainder.exact_zero(
                new_variable, new_point, source="exact remainder substitution"
            )
        if self.kind is RemainderKind.UNKNOWN or self.scale is None:
            return Remainder.unknown(
                new_variable,
                new_point,
                exact_expression=exact,
                source="substitution of unknown remainder",
            )
        return Remainder(
            new_variable,
            new_point,
            self.kind,
            self.scale.subs(self.variable, mapping),
            exact,
            self.provenance + (RemainderProvenance("germ-preserving substitution"),),
        )


@dataclass(frozen=True)
class Truncation:
    """A finite prefix together with its explicit remainder semantics."""

    prefix: sp.Expr
    remainder: Remainder
    terms_kept: int
    total_known_terms: int

    @property
    def statement(self) -> str:
        return f"{sp.sstr(self.prefix)} + {self.remainder.notation}"

    def reconstruct(self) -> sp.Expr | None:
        """Return the exact represented value when the exact error is known."""

        if self.remainder.exact_expression is None:
            return None
        return analytic_powsimp(
            sp.expand(self.prefix + self.remainder.exact_expression)
        )
