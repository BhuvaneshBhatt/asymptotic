"""Canonical asymptotic monomials on a ramified local uniformizer.

The core representation is deliberately small: on a local cover ``h=t**r`` a
monomial is stored as

    exp(Q(t)) * t**alpha * log(h)**beta,   h=t**r.

This covers the formal monomials produced by ``odeanalysis`` and gives the
transseries layer a structural object to compare and multiply instead of
repeatedly reverse engineering arbitrary SymPy products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce

import sympy as sp

from ._integer_utils import integer_lcm as _lcm
from ._power_simplify import analytic_powsimp, formal_powsimp
from .context import AsymptoticContext, GrowthComparison


def ramification_index(*indices: int) -> int:
    """Return the least common cover containing all positive ramification indices."""

    values = tuple(int(value) for value in indices)
    if any(value < 1 for value in values):
        raise ValueError("ramification indices must be positive")
    return reduce(_lcm, values, 1)


@dataclass(frozen=True)
class RamificationModel:
    """A positive local uniformizer with ``h = t**index``.

    ``h`` means ``x-point`` at a finite point, ``1/x`` at ``+oo``, and
    ``-1/x`` at ``-oo``.  Consequently ``t -> 0+`` in every supported case.
    This uniform convention is what makes finite, infinite, and ramified ODE
    expansions use the same monomial algebra.
    """

    variable: sp.Symbol
    point: sp.Expr = sp.oo
    index: int = 1
    parameter: sp.Symbol = field(default_factory=lambda: sp.Dummy("t", positive=True))

    def __post_init__(self) -> None:
        if not isinstance(self.variable, sp.Symbol):
            raise TypeError("variable must be a Symbol")
        if int(self.index) != self.index or int(self.index) < 1:
            raise ValueError("ramification index must be a positive integer")
        object.__setattr__(self, "point", sp.sympify(self.point))
        object.__setattr__(self, "index", int(self.index))
        if not isinstance(self.parameter, sp.Symbol):
            raise TypeError("parameter must be a Symbol")

    @property
    def local_coordinate(self) -> sp.Expr:
        if self.point is sp.oo:
            return 1 / self.variable
        if self.point is -sp.oo:
            return -1 / self.variable
        return self.variable - self.point

    @property
    def variable_in_parameter(self) -> sp.Expr:
        t = self.parameter
        if self.point is sp.oo:
            return t ** (-self.index)
        if self.point is -sp.oo:
            return -(t ** (-self.index))
        return self.point + t**self.index

    def to_parameter(self, expr: sp.Expr) -> sp.Expr:
        """Rewrite an expression in the original variable on the ``t`` cover."""

        expr = sp.sympify(expr)
        mapped = expr.xreplace({self.variable: self.variable_in_parameter})
        mapped = sp.expand_log(mapped, force=False)
        return sp.powsimp(sp.expand_power_base(mapped, force=False), force=False)

    def from_parameter(self, expr: sp.Expr) -> sp.Expr:
        """Rewrite a cover expression back using the principal rational power of ``h``."""

        root = self.local_coordinate ** sp.Rational(1, self.index)
        return sp.powsimp(sp.sympify(expr).xreplace({self.parameter: root}), force=False)

    def refine(self, new_index: int, *, parameter: sp.Symbol | None = None) -> RamificationModel:
        """Return a common finer cover, preserving the asymptotic point."""

        new_index = int(new_index)
        if new_index < 1 or new_index % self.index:
            raise ValueError("new ramification index must be a positive multiple of the old index")
        return RamificationModel(
            self.variable,
            self.point,
            new_index,
            parameter or sp.Dummy("t", positive=True),
        )


@dataclass(frozen=True)
class AsymptoticMonomial:
    """Canonical power/log/exponential monomial on a local ramified cover.

    ``log_power`` is the power of ``log(h)`` with ``h=t**r``.  Using the
    unramified local coordinate for logarithms makes refinement to a finer
    cover exact rather than changing a monomial by an unnoticed constant.
    """

    ramification: RamificationModel
    exponential: sp.Expr = sp.S.Zero
    power: sp.Expr = sp.S.Zero
    log_power: sp.Expr = sp.S.Zero

    def __post_init__(self) -> None:
        t = self.ramification.parameter
        exponential = analytic_powsimp(sp.expand(sp.sympify(self.exponential)))
        power = sp.simplify(sp.sympify(self.power))
        log_power = sp.simplify(sp.sympify(self.log_power))
        # All structural pieces must live on this one cover.  Constants in Q
        # are intentionally retained: they are harmless for ordering and keep
        # exact reconstruction possible when the object was built explicitly.
        foreign = (exponential.free_symbols | power.free_symbols | log_power.free_symbols) - {t}
        allowed_parameters = foreign - {self.ramification.variable}
        if self.ramification.variable in foreign:
            raise ValueError("monomial data must be written in the local parameter")
        # Other symbols are allowed as true symbolic parameters.
        _ = allowed_parameters
        object.__setattr__(self, "exponential", exponential)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "log_power", log_power)

    @property
    def parameter(self) -> sp.Symbol:
        return self.ramification.parameter

    @property
    def parameter_expression(self) -> sp.Expr:
        t = self.parameter
        log_h = self.ramification.index * sp.log(t)
        return formal_powsimp(sp.exp(self.exponential) * t**self.power * log_h**self.log_power)

    @property
    def expression(self) -> sp.Expr:
        return self.ramification.from_parameter(self.parameter_expression)

    def on_ramification(self, target: RamificationModel) -> AsymptoticMonomial:
        """Lift this monomial to a compatible finer cover."""

        if self.ramification.variable != target.variable or self.ramification.point != target.point:
            raise ValueError("ramifications refer to different asymptotic variables/points")
        if target.index % self.ramification.index:
            raise ValueError("target is not a refinement of this ramification")
        factor = target.index // self.ramification.index
        old_t = self.parameter
        new_t = target.parameter
        subst = {old_t: new_t**factor}
        return AsymptoticMonomial(
            target,
            sp.expand(self.exponential.xreplace(subst)),
            sp.simplify(self.power * factor),
            self.log_power,
        )

    def _common_cover(
        self, other: AsymptoticMonomial
    ) -> tuple[AsymptoticMonomial, AsymptoticMonomial]:
        if (
            self.ramification.variable != other.ramification.variable
            or self.ramification.point != other.ramification.point
        ):
            raise ValueError("monomials refer to different asymptotic variables/points")
        r = ramification_index(self.ramification.index, other.ramification.index)
        t = sp.Dummy("t", positive=True)
        target = RamificationModel(self.ramification.variable, self.ramification.point, r, t)
        return self.on_ramification(target), other.on_ramification(target)

    def __mul__(self, other: AsymptoticMonomial) -> AsymptoticMonomial:
        if not isinstance(other, AsymptoticMonomial):
            return NotImplemented
        left, right = self._common_cover(other)
        return AsymptoticMonomial(
            left.ramification,
            sp.expand(left.exponential + right.exponential),
            sp.simplify(left.power + right.power),
            sp.simplify(left.log_power + right.log_power),
        )

    def __truediv__(self, other: AsymptoticMonomial) -> AsymptoticMonomial:
        if not isinstance(other, AsymptoticMonomial):
            return NotImplemented
        left, right = self._common_cover(other)
        return AsymptoticMonomial(
            left.ramification,
            sp.expand(left.exponential - right.exponential),
            sp.simplify(left.power - right.power),
            sp.simplify(left.log_power - right.log_power),
        )

    def __pow__(self, exponent: sp.Expr) -> AsymptoticMonomial:
        exponent = sp.sympify(exponent)
        return AsymptoticMonomial(
            self.ramification,
            sp.expand(exponent * self.exponential),
            sp.simplify(exponent * self.power),
            sp.simplify(exponent * self.log_power),
        )


def _extract_parameter_monomial(
    expr: sp.Expr,
    ramification: RamificationModel,
) -> tuple[sp.Expr, AsymptoticMonomial]:
    """Split ``expr(t)`` into a t-independent coefficient and canonical monomial.

    The accepted structural group is exactly products of exponentials, powers
    of ``t``, and powers of ``log(t)``.  Refusing an unrecognized t-dependent
    factor is preferable to silently pretending it is a coefficient.
    """

    t = ramification.parameter
    expr = sp.powsimp(sp.expand_power_base(sp.sympify(expr), force=False), force=False)
    coefficient = sp.S.One
    q = sp.S.Zero
    alpha = sp.S.Zero
    beta = sp.S.Zero

    for factor in sp.Mul.make_args(expr):
        base, exponent = factor.as_base_exp()
        if factor.func is sp.exp:
            q += factor.args[0]
            continue
        if base.func is sp.exp:
            q += exponent * base.args[0]
            continue
        if base == t:
            alpha += exponent
            continue
        if base == sp.log(t):
            beta += exponent
            coefficient *= sp.Integer(ramification.index) ** (-exponent)
            continue
        if t not in factor.free_symbols:
            coefficient *= factor
            continue
        raise ValueError(f"unsupported t-dependent monomial factor: {factor}")

    return (
        sp.simplify(coefficient),
        AsymptoticMonomial(ramification, sp.expand(q), sp.simplify(alpha), sp.simplify(beta)),
    )


def canonical_parameter_monomial(
    expr: sp.Expr,
    ramification: RamificationModel,
) -> tuple[sp.Expr, AsymptoticMonomial]:
    """Canonicalize a multiplicative term already written in the uniformizer."""

    return _extract_parameter_monomial(expr, ramification)


def canonical_asymptotic_monomial(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    ramification_index: int = 1,
    parameter: sp.Symbol | None = None,
) -> tuple[sp.Expr, AsymptoticMonomial]:
    """Canonicalize a single multiplicative asymptotic term.

    Returns ``(coefficient, monomial)``.  At infinity the canonical parameter
    tends to zero, so e.g. ``x**3*exp(-x)*log(x)**2`` is represented on
    ``t=1/x`` rather than mixing conventions for large and small variables.
    """

    ram = RamificationModel(
        variable,
        point,
        ramification_index,
        parameter or sp.Dummy("t", positive=True),
    )
    return _extract_parameter_monomial(ram.to_parameter(expr), ram)


def _sign_of_real_expression(expr: sp.Expr, ctx: AsymptoticContext) -> int | None:
    expr = sp.simplify(expr)
    if expr.is_positive is True:
        return 1
    if expr.is_negative is True:
        return -1
    if expr.is_zero is True or ctx.is_zero(expr) is True:
        return 0
    return ctx.eventual_sign(expr)


def compare_asymptotic_monomials(
    left: AsymptoticMonomial,
    right: AsymptoticMonomial,
) -> GrowthComparison:
    """Compare monomial magnitudes by exponential, power, then logarithmic level.

    The comparison is lexicographic only after cancellation on a common cover.
    In particular ``exp(t**-2+t**-1)`` versus ``exp(t**-2-t**-1)`` first
    cancels the shared ``t**-2`` contribution and orders the surviving
    exponential difference, rather than classifying the two as merely the same
    top exponential scale.
    """

    left, right = left._common_cover(right)
    t = left.parameter
    ctx = AsymptoticContext(t, point=0)

    dq = sp.simplify(sp.expand(left.exponential - right.exponential))
    if ctx.is_zero(dq) is not True:
        real_dq = dq if dq.is_real is True else sp.re(dq)
        lim = ctx.limit(real_dq)
        if lim is sp.oo:
            return GrowthComparison.LARGER
        if lim is -sp.oo:
            return GrowthComparison.SMALLER
        # A finite Q-difference contributes only a bounded nonzero factor.
        if getattr(lim, "is_finite", False) is not True:
            # Sometimes the sign is decidable even when SymPy declines a limit.
            sign = _sign_of_real_expression(real_dq, ctx)
            magnitude = ctx.limit(sp.Abs(real_dq)) if sign else None
            if sign and magnitude is sp.oo:
                return GrowthComparison.LARGER if sign > 0 else GrowthComparison.SMALLER
            return GrowthComparison.UNKNOWN

    dp = sp.simplify(left.power - right.power)
    sign = _sign_of_real_expression(dp, ctx)
    if sign is None:
        return GrowthComparison.UNKNOWN
    if sign:
        # t -> 0+: a smaller power exponent means a larger magnitude.
        return GrowthComparison.SMALLER if sign > 0 else GrowthComparison.LARGER

    dl = sp.simplify(left.log_power - right.log_power)
    sign = _sign_of_real_expression(dl, ctx)
    if sign is None:
        return GrowthComparison.UNKNOWN
    if sign:
        # |log t| -> infinity.
        return GrowthComparison.LARGER if sign > 0 else GrowthComparison.SMALLER
    return GrowthComparison.SAME_ORDER
