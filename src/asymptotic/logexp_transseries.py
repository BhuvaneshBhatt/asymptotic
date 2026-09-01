"""Recursive finite-height logarithmico-exponential transseries algebra.

The flat :mod:`asymptotic.monomial` representation is optimized for
``exp(Q(t))*t**a*log(h)**b``.  This module extends the multiplicative monomial
algebra to arbitrary *finite-height* logarithmico-exponential generators such
as ``exp(exp(x))``, ``log(log(x))`` and products/powers of them.

A recursive monomial is represented canonically as

    exp(E(x)) * prod(g_i(x)**a_i),

where ``E`` may itself contain finite-height exp/log expressions and every
``g_i`` is a non-exponential generator (typically a local coordinate or a
nested logarithm).  Exponential factors are always folded into ``E``.  This
normal form makes multiplication, division and constant powers exact while
comparison delegates to the finite-height LE hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp, formal_powsimp
from .context import GrowthComparison
from .exp_log_scale import LogExpScale, compare_log_exp_scales


def _height(expr: sp.Expr, variable: sp.Symbol) -> int:
    if variable not in expr.free_symbols:
        return 0
    child = max((_height(arg, variable) for arg in expr.args), default=0)
    if expr.func in (sp.exp, sp.log):
        return child + 1
    return child


def _generator_key(item: tuple[sp.Expr, sp.Expr]) -> tuple:
    return sp.default_sort_key(item[0])


@dataclass(frozen=True)
class RecursiveLogExpMonomial:
    """Canonical finite-height logarithmico-exponential monomial.

    ``powers`` is a sorted tuple of ``(generator, exponent)`` pairs.  The
    exponents must be independent of the asymptotic variable; symbolic
    parameters are allowed.  Exponential generators never occur in
    ``powers``: ``exp(A)**c`` is normalized into ``exponential += c*A``.
    """

    variable: sp.Symbol
    point: sp.Expr = sp.oo
    exponential: sp.Expr = sp.S.Zero
    powers: tuple[tuple[sp.Expr, sp.Expr], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.variable, sp.Symbol):
            raise TypeError("variable must be a Symbol")
        point = sp.sympify(self.point)
        exponential = analytic_powsimp(sp.expand(sp.sympify(self.exponential)))
        grouped = {}
        for base, exponent in self.powers:
            base = analytic_powsimp(sp.sympify(base))
            exponent = sp.simplify(sp.sympify(exponent))
            if self.variable in exponent.free_symbols:
                raise ValueError(
                    "generator exponents must be independent of the asymptotic variable"
                )
            if exponent == 0:
                continue
            if base.func is sp.exp:
                exponential += exponent * base.args[0]
                continue
            grouped[base] = sp.simplify(grouped.get(base, 0) + exponent)
        powers = tuple(sorted(((b, e) for b, e in grouped.items() if e != 0), key=_generator_key))
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "exponential", analytic_powsimp(sp.expand(exponential)))
        object.__setattr__(self, "powers", powers)

    @property
    def expression(self) -> sp.Expr:
        product = sp.exp(self.exponential)
        for base, exponent in self.powers:
            product *= base**exponent
        return formal_powsimp(product)

    @property
    def height(self) -> int:
        return _height(self.expression, self.variable)

    @property
    def logarithm_expression(self) -> sp.Expr:
        result = self.exponential
        for base, exponent in self.powers:
            result += exponent * sp.log(base)
        return sp.expand(result)

    @property
    def logarithmic_derivative(self) -> sp.Expr:
        return sp.simplify(sp.diff(self.logarithm_expression, self.variable))

    def _check(self, other: RecursiveLogExpMonomial) -> None:
        if self.variable != other.variable or self.point != other.point:
            raise ValueError("recursive monomials use different variables or asymptotic points")

    def __mul__(self, other: RecursiveLogExpMonomial) -> RecursiveLogExpMonomial:
        if not isinstance(other, RecursiveLogExpMonomial):
            return NotImplemented
        self._check(other)
        return RecursiveLogExpMonomial(
            self.variable,
            self.point,
            self.exponential + other.exponential,
            self.powers + other.powers,
        )

    def __truediv__(self, other: RecursiveLogExpMonomial) -> RecursiveLogExpMonomial:
        if not isinstance(other, RecursiveLogExpMonomial):
            return NotImplemented
        self._check(other)
        return RecursiveLogExpMonomial(
            self.variable,
            self.point,
            self.exponential - other.exponential,
            self.powers + tuple((base, -exponent) for base, exponent in other.powers),
        )

    def __pow__(self, exponent: sp.Expr) -> RecursiveLogExpMonomial:
        exponent = sp.sympify(exponent)
        if self.variable in exponent.free_symbols:
            raise ValueError("recursive monomials only support variable-independent powers")
        return RecursiveLogExpMonomial(
            self.variable,
            self.point,
            exponent * self.exponential,
            tuple((base, exponent * power) for base, power in self.powers),
        )

    def compare(self, other: RecursiveLogExpMonomial) -> GrowthComparison:
        self._check(other)
        return compare_log_exp_scales(
            self.expression, other.expression, self.variable, point=self.point
        )


def _local_base(variable: sp.Symbol, point: sp.Expr) -> sp.Expr:
    point = sp.sympify(point)
    if point is sp.oo:
        return variable
    if point is -sp.oo:
        return -variable
    return variable - point


def canonical_recursive_logexp_monomial(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
) -> tuple[sp.Expr, RecursiveLogExpMonomial]:
    """Split one multiplicative LE term into coefficient and recursive monomial.

    The accepted group contains finite products of variable-independent
    coefficients, exponentials, powers of the local base, and powers of nested
    logarithms.  Arbitrary oscillatory/special-function factors are rejected.
    """

    expr = analytic_powsimp(sp.expand_power_base(sp.sympify(expr), force=False))
    coefficient = sp.S.One
    exponential = sp.S.Zero
    powers = []
    local = _local_base(variable, point)

    def add_factor(factor: sp.Expr) -> None:
        nonlocal coefficient, exponential
        if variable not in factor.free_symbols:
            coefficient *= factor
            return
        if factor.func is sp.exp:
            exponential += factor.args[0]
            return
        base, exponent = factor.as_base_exp()
        if variable in exponent.free_symbols:
            raise ValueError(f"variable-dependent exponent is not a monomial power: {factor}")
        if base.func is sp.exp:
            exponential += exponent * base.args[0]
            return
        # Base powers and nested logarithmic generators are multiplicative LE
        # generators.  More general additive bases are deliberately refused.
        if base == variable or base == local or base.func is sp.log:
            powers.append((base, exponent))
            return
        if isinstance(base, sp.Pow) and variable in base.free_symbols:
            # ``(x**a)**b`` after conservative SymPy normalization.
            inner_base, inner_exp = base.as_base_exp()
            if variable not in inner_exp.free_symbols and (
                inner_base == variable or inner_base == local
            ):
                powers.append((inner_base, sp.simplify(inner_exp * exponent)))
                return
        raise ValueError(f"unsupported recursive log-exp monomial factor: {factor}")

    for factor in sp.Mul.make_args(expr):
        add_factor(factor)

    return sp.simplify(coefficient), RecursiveLogExpMonomial(
        variable, point, sp.expand(exponential), tuple(powers)
    )


def recursive_logexp_scale(
    expr: sp.Expr, variable: sp.Symbol, *, point: sp.Expr = sp.oo
) -> LogExpScale:
    """Return the finite-height scale descriptor for ``expr``."""

    return LogExpScale.from_expr(expr, variable, point=point)
