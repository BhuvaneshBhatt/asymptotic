from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, context_for
from .decomposition import StructuralDecomposition, decompose_expression
from .mrv import MRVDecomposition, mrv_decomposition
from .tower import ExpLogTower


def _iter_log(expr: sp.Expr, n: int) -> sp.Expr:
    out = expr
    for _ in range(n):
        out = sp.log(out)
    return out


def _iter_exp(expr: sp.Expr, n: int) -> sp.Expr:
    out = expr
    for _ in range(n):
        out = sp.exp(out)
    return out


@dataclass(frozen=True)
class NestedLevel:
    epsilon: int
    exp_depth: int
    log_depth: int
    power: sp.Expr
    remainder: sp.Expr

    def __post_init__(self) -> None:
        if self.epsilon not in (-1, 1):
            raise ValueError("epsilon must be ±1")
        if self.exp_depth < 0 or self.log_depth < 0:
            raise ValueError("depths must be nonnegative")
        if sp.sympify(self.power).is_positive is False:
            raise ValueError("power must be positive")

    def reconstruct(self, x: sp.Symbol) -> sp.Expr:
        base = _iter_log(x, self.log_depth) ** self.power * self.remainder
        value = _iter_exp(base, self.exp_depth)
        return value if self.epsilon == 1 else 1 / value


@dataclass(frozen=True)
class NestedForm:
    variable: sp.Symbol
    exact_expr: sp.Expr
    constant: sp.Expr | None
    outer_sign: int
    levels: tuple[NestedLevel, ...]
    structural: StructuralDecomposition | None = field(default=None, compare=False, repr=False)
    mrv: MRVDecomposition | None = field(default=None, compare=False, repr=False)

    @property
    def terminal_remainder(self) -> sp.Expr:
        if not self.levels:
            return self.exact_expr - self.constant if self.constant is not None else self.exact_expr
        return self.levels[-1].remainder

    def reconstruct(self) -> sp.Expr:
        if self.constant is not None:
            if not self.levels:
                return self.constant
            return self.constant + self.outer_sign * self.levels[0].reconstruct(self.variable)
        if not self.levels:
            return self.exact_expr
        return self.outer_sign * self.levels[0].reconstruct(self.variable)

    def as_expansion(self, *, point: sp.Expr = sp.oo) -> NestedExpansion:
        return NestedExpansion(self.exact_expr, self.variable, point=point, seed=(self,))

    def __add__(self, other):
        return self.as_expansion() + other

    def __mul__(self, other):
        return self.as_expansion() * other

    def __pow__(self, power):
        return self.as_expansion() ** power


class NestedExpansion:
    """Resumable exact nested expansion.

    Successive forms are generated from the exact terminal residual only when
    requested.  The object therefore has no fixed expansion depth. Arithmetic
    creates another lazy ``NestedExpansion`` from exact operands; cancellation
    is checked by the shared ``AsymptoticContext``/``exprtest`` zero oracle
    before another residual form is requested.
    """

    def __init__(
        self,
        expr: sp.Expr,
        variable: sp.Symbol,
        *,
        point: sp.Expr = sp.oo,
        context: AsymptoticContext | None = None,
        seed: tuple[NestedForm, ...] = (),
    ) -> None:
        self.expr = sp.sympify(expr)
        self.variable = variable
        self.point = point
        self.context = context_for(variable, point, context)
        self._forms: list[NestedForm] = list(seed)
        self._current = self._forms[-1].terminal_remainder if self._forms else self.expr
        self.exhausted = False
        if self.context.is_zero(self._current) is True or not self._current.has(variable):
            self.exhausted = True

    @property
    def forms(self) -> tuple[NestedForm, ...]:
        return tuple(self._forms)

    @property
    def exact_remainder(self) -> sp.Expr:
        return self._current

    def next_form(self) -> NestedForm | None:
        if self.exhausted:
            return None
        form = nested_form(
            self._current, self.variable, point=self.point, max_levels=1, context=self.context
        )
        self._forms.append(form)
        remainder = analytic_powsimp(sp.simplify(form.terminal_remainder))
        self._current = remainder
        zero = self.context.is_zero(remainder)
        if zero is True or not remainder.has(self.variable):
            self.exhausted = True
        return form

    def refine(self, depth: int) -> NestedExpansion:
        while len(self._forms) < depth and not self.exhausted:
            if self.next_form() is None:
                break
        return self

    def _coerce_expr(self, other) -> sp.Expr:
        if isinstance(other, NestedExpansion):
            return other.expr
        if isinstance(other, NestedForm):
            return other.exact_expr
        return sp.sympify(other)

    def _binary(self, other, op: Callable[[sp.Expr, sp.Expr], sp.Expr]) -> NestedExpansion:
        expr = analytic_powsimp(sp.simplify(op(self.expr, self._coerce_expr(other))))
        return NestedExpansion(expr, self.variable, point=self.point, context=self.context)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return NestedExpansion(
            self._coerce_expr(other) - self.expr,
            self.variable,
            point=self.point,
            context=self.context,
        )

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __pow__(self, power):
        return NestedExpansion(
            self.expr ** sp.sympify(power), self.variable, point=self.point, context=self.context
        )

    def exp(self) -> NestedExpansion:
        return NestedExpansion(
            sp.exp(self.expr), self.variable, point=self.point, context=self.context
        )

    def log(self) -> NestedExpansion:
        return NestedExpansion(
            sp.log(self.expr), self.variable, point=self.point, context=self.context
        )

    def asymptotic_element(self):
        """View this nested expansion through the common asymptotic-field protocol."""
        from .algebra import asymptotic_element

        return asymptotic_element(self)

    def differentiate(self, order: int = 1) -> NestedExpansion:
        if order < 0:
            raise ValueError("order must be nonnegative")
        return NestedExpansion(
            sp.diff(self.expr, self.variable, order),
            self.variable,
            point=self.point,
            context=self.context,
        )

    def integrate(self, *, constant: sp.Expr = 0, terms: int | None = None) -> NestedExpansion:
        primitive = sp.integrate(self.expr, self.variable)
        if primitive.has(sp.Integral):
            raise NotImplementedError(f"Could not integrate {self.expr}")
        return NestedExpansion(
            analytic_powsimp(sp.simplify(primitive + sp.sympify(constant))),
            self.variable,
            point=self.point,
            context=self.context,
        )

    def inverse_asymptotic(
        self,
        inverse_variable: sp.Symbol | None = None,
        *,
        terms: int = 6,
        branch: int | None = 0,
    ):
        """Asymptotically invert the exact expression represented here."""
        from .reversion import inverse_asymptotic

        return inverse_asymptotic(
            self.expr,
            self.variable,
            inverse_variable,
            point=self.point,
            terms=terms,
            branch=branch,
            context=self.context,
        )


def _finite_nonzero(value: sp.Expr) -> bool:
    return value.is_finite is True and value.is_zero is False


def _depth_budget(expr: sp.Expr, variable: sp.Symbol) -> tuple[int, int]:
    tower = ExpLogTower.from_expr(expr, variable)
    exp_depth = sum(ext.kind == "exp" for ext in tower.extensions) + 1
    log_depth = sum(ext.kind == "log" for ext in tower.extensions) + 2
    return max(1, exp_depth), max(1, log_depth)


def _choose_level(
    magnitude: sp.Expr,
    x: sp.Symbol,
    ctx: AsymptoticContext,
    *,
    max_exp_depth: int | None,
    max_log_depth: int | None,
) -> NestedLevel:
    """Find one Shackell-style exp/log/power level for ``magnitude``.

    The search takes the number of logarithms needed to expose a power of an
    iterated logarithm from the expression's exp-log tower. A candidate is
    accepted only when the remaining factor has a finite nonzero limit or is
    logarithmically smaller than the selected base.
    """

    mag_lim = ctx.limit(magnitude)
    if mag_lim == 0:
        epsilon = -1
        target = sp.simplify(1 / magnitude)
    elif mag_lim is sp.oo:
        epsilon = 1
        target = magnitude
    else:
        raise ValueError("_choose_level expects a function tending to 0 or +oo")

    auto_exp, auto_log = _depth_budget(target, x)
    emax = auto_exp if max_exp_depth is None else max_exp_depth
    lmax = auto_log if max_log_depth is None else max_log_depth

    for s in range(emax + 1):
        y = target
        valid = True
        for _ in range(s):
            if ctx.limit(y) is not sp.oo:
                valid = False
                break
            y = sp.log(y)
        if not valid:
            continue
        for m in range(lmax + 1):
            base = _iter_log(x, m)
            try:
                d = ctx.limit(sp.log(sp.Abs(y)) / sp.log(base))
            except SYMBOLIC_ERRORS:
                continue
            if not _finite_nonzero(d) or d.is_positive is not True:
                continue
            remainder = analytic_powsimp(sp.simplify(y / base**d))
            rlim = ctx.limit(remainder)
            if _finite_nonzero(rlim):
                return NestedLevel(epsilon, s, m, sp.simplify(d), remainder)
            try:
                lower = ctx.limit(sp.log(sp.Abs(remainder)) / sp.log(base))
            except SYMBOLIC_ERRORS:
                lower = None
            if lower == 0:
                return NestedLevel(epsilon, s, m, sp.simplify(d), remainder)
    raise NotImplementedError(f"Could not identify a nested level for {magnitude}")


def nested_form(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    max_levels: int = 8,
    max_exp_depth: int | None = None,
    max_log_depth: int | None = None,
    context: AsymptoticContext | None = None,
) -> NestedForm:
    """Decompose an expression into a canonical nested asymptotic form."""
    expr = sp.sympify(expr)
    ctx = context_for(variable, point, context)
    structural = decompose_expression(expr, variable)
    mrv = mrv_decomposition(expr, variable, point, context=ctx, structural=structural)
    lim = ctx.limit(expr)

    constant: sp.Expr | None = None
    residual = expr
    if _finite_nonzero(lim):
        constant = lim
        residual = sp.simplify(expr - lim)
        if ctx.is_zero(residual) is True:
            return NestedForm(variable, expr, constant, 1, (), structural, mrv)

    sign = ctx.eventual_sign(residual)
    if sign is None or sign == 0:
        raise NotImplementedError(f"Could not determine eventual sign of {residual}")
    magnitude = sp.simplify(sign * residual)

    levels = []
    current = magnitude
    for _ in range(max_levels):
        if _finite_nonzero(ctx.limit(current)):
            break
        level = _choose_level(
            current,
            variable,
            ctx,
            max_exp_depth=max_exp_depth,
            max_log_depth=max_log_depth,
        )
        levels.append(level)
        current = level.remainder
        if _finite_nonzero(ctx.limit(current)):
            break
    return NestedForm(variable, expr, constant, sign, tuple(levels), structural, mrv)


def nested_expansion(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    depth: int = 4,
    point: sp.Expr = sp.oo,
    max_exp_depth: int | None = None,
    max_log_depth: int | None = None,
) -> NestedExpansion:
    """Build a resumable nested expansion, eagerly refining up to *depth* levels."""
    expansion = NestedExpansion(expr, variable, point=point)
    # ``depth`` controls only eager initial refinement; the returned expansion
    # remains resumable and can be refined further.
    while len(expansion.forms) < depth and not expansion.exhausted:
        form = nested_form(
            expansion.exact_remainder,
            variable,
            point=point,
            max_levels=1,
            max_exp_depth=max_exp_depth,
            max_log_depth=max_log_depth,
            context=expansion.context,
        )
        expansion._forms.append(form)
        expansion._current = analytic_powsimp(sp.simplify(form.terminal_remainder))
        if expansion.context.is_zero(expansion._current) is True or not expansion._current.has(
            variable
        ):
            expansion.exhausted = True
    return expansion
