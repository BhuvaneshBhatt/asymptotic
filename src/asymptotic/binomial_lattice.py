"""Discrete Binomial endpoint saddle expansions with lattice corrections."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_policy import bounded_assumption_sign, bounded_limit
from .remainder import AsymptoticRemainder
from .stirling import (
    StirlingLocalMassExpansion,
    StirlingNormalization,
    stirling_local_mass_expansion,
)


@dataclass(frozen=True)
class BinomialLatticeTailCertificate:
    """Theorem data for a one-sided Binomial endpoint lattice expansion.

    The certificate uses exact adjacent-mass ratios, a positive-real Stieltjes
    boundary-mass expansion, and geometric domination with limiting ratio
    ``rho < 1``.  Its replay method checks the algebraic hypotheses retained
    by the construction; the analytic theorem then yields a relative
    ``O(parameter**(-terms))`` remainder, with an additional exponentially
    small finite-tail truncation.
    """

    side: str
    scale: sp.Expr
    probability: sp.Expr
    ratio_limit: sp.Expr
    lattice_offset: sp.Expr
    parameter: sp.Symbol
    terms: int
    boundary_normalization: StirlingNormalization
    conditions: tuple[sp.Expr, ...]

    @property
    def certified(self) -> bool:
        return self.replay() is True

    def replay(self) -> bool | None:
        if not self.boundary_normalization.certified:
            return False
        if self.side not in {"lower", "upper"} or self.terms < 1:
            return False
        if bounded_assumption_sign(self.scale) != 1:
            return None
        if bounded_assumption_sign(1 - self.scale) != 1:
            return None
        if bounded_assumption_sign(self.probability) != 1:
            return None
        if bounded_assumption_sign(1 - self.probability) != 1:
            return None
        if bounded_assumption_sign(self.ratio_limit) != 1:
            return None
        if bounded_assumption_sign(1 - self.ratio_limit) != 1:
            return None
        ordering = (
            bounded_assumption_sign(self.scale - self.probability)
            if self.side == "upper"
            else bounded_assumption_sign(self.probability - self.scale)
        )
        if ordering != 1:
            return None
        if self.lattice_offset.has(sp.floor, sp.ceiling):
            return True
        try:
            bound = bounded_limit(
                sp.Abs(self.lattice_offset), self.parameter, sp.oo, allow_general=True
            )
        except (ValueError, TypeError, NotImplementedError):
            return None
        return True if getattr(bound, "is_finite", None) is True else None


@dataclass(frozen=True)
class BinomialLatticeTailExpansion:
    expression: sp.Expr
    boundary: sp.Expr
    scale: sp.Expr
    lattice_offset: sp.Expr
    ratio: sp.Expr
    lattice_factor: sp.Expr
    side: str
    conditions: tuple[sp.Expr, ...]
    normalization: StirlingNormalization
    local_mass: StirlingLocalMassExpansion
    certificate: BinomialLatticeTailCertificate
    remainder: AsymptoticRemainder


def _geometric_power_sum(power: int, rho: sp.Expr) -> sp.Expr:
    value = 1 / (1 - rho)
    for _ in range(power):
        value = analytic_powsimp(rho * sp.diff(value, rho)) if isinstance(rho, sp.Symbol) else value
    return value


def _sum_polynomial_times_geometric(polynomial: sp.Expr, index: sp.Symbol, rho: sp.Expr) -> sp.Expr:
    try:
        poly = sp.Poly(sp.expand(polynomial), index)
    except sp.PolynomialError as exc:
        raise NotImplementedError(
            "lattice correction is not polynomial in the offset index"
        ) from exc
    r = sp.Dummy("_rho")
    moments = [1 / (1 - r)]
    for _ in range(poly.degree()):
        moments.append(sp.factor(r * sp.diff(moments[-1], r)))
    total = sp.S.Zero
    for (degree,), coefficient in poly.terms():
        total += coefficient * moments[degree]
    return sp.factor(sp.cancel(total.xreplace({r: rho})))


def _ratio_lattice_factor(
    *,
    side: str,
    scale: sp.Expr,
    offset: sp.Expr,
    probability: sp.Expr,
    parameter: sp.Symbol,
    terms: int,
) -> tuple[sp.Expr, sp.Expr]:
    """Sum the boundary mass-ratio expansion that supplies the discrete tail factor."""
    h = sp.Dummy("_binomial_h", positive=True)
    delta = sp.Dummy("_binomial_delta", real=True)
    j = sp.Dummy("_j", integer=True, nonnegative=True)
    i = sp.Dummy("_i", integer=True, positive=True)
    p = sp.sympify(probability)
    q = 1 - p
    k = scale / h + delta
    n = 1 / h
    if side == "upper":
        ratio_i = analytic_powsimp((n - k - i + 1) * p / ((k + i) * q))
        rho = analytic_powsimp(p * (1 - scale) / (q * scale))
    elif side == "lower":
        ratio_i = analytic_powsimp((k - i + 1) * q / ((n - k + i) * p))
        rho = analytic_powsimp(scale * q / ((1 - scale) * p))
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    relative = sp.factor(sp.cancel(ratio_i / rho))
    try:
        log_relative = sp.series(sp.log(relative), h, 0, max(2, terms)).removeO()
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not expand the Binomial adjacent-mass ratio") from exc
    log_product = sp.summation(log_relative, (i, 1, j))
    try:
        product_correction = sp.series(sp.exp(log_product), h, 0, max(2, terms)).removeO()
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise NotImplementedError("could not expand the Binomial lattice product") from exc

    factor_h = sp.S.Zero
    expanded = sp.expand(product_correction)
    for power in range(max(1, terms)):
        coefficient = expanded.coeff(h, power)
        if coefficient == 0:
            continue
        factor_h += _sum_polynomial_times_geometric(coefficient, j, rho) * h**power
    factor = analytic_powsimp(factor_h.xreplace({h: 1 / parameter, delta: offset}))
    return rho, factor


def binomial_lattice_tail_expansion(
    normalization: StirlingNormalization,
    *,
    variable: sp.Symbol,
    parameter: sp.Symbol,
    probability: sp.Expr,
    lower: sp.Expr,
    upper: sp.Expr,
    count: sp.Expr,
    terms: int = 4,
) -> BinomialLatticeTailExpansion | None:
    """Expand a one-sided Binomial tail from its boundary mass and mass ratios."""

    if count != parameter:
        return None
    lower = sp.sympify(lower)
    upper = sp.sympify(upper)
    if lower == 0 and upper != count:
        side = "lower"
        boundary = upper
    elif upper == count and lower != 0:
        side = "upper"
        boundary = lower
    else:
        return None

    local = stirling_local_mass_expansion(
        normalization,
        location=boundary,
        variable=variable,
        parameter=parameter,
        terms=terms,
    )
    a = local.scale
    p = sp.sympify(probability)
    if side == "upper":
        if bounded_assumption_sign(a - p) != 1:
            return None
        conditions = (sp.Gt(a, p), sp.Gt(a, 0), sp.Lt(a, 1), sp.Gt(p, 0), sp.Lt(p, 1))
    else:
        if bounded_assumption_sign(p - a) != 1:
            return None
        conditions = (sp.Lt(a, p), sp.Gt(a, 0), sp.Lt(a, 1), sp.Gt(p, 0), sp.Lt(p, 1))
    rho, factor = _ratio_lattice_factor(
        side=side,
        scale=a,
        offset=local.lattice_offset,
        probability=p,
        parameter=parameter,
        terms=terms,
    )
    if bounded_assumption_sign(1 - rho) != 1:
        return None
    expression = analytic_powsimp(local.expression * factor)
    certificate = BinomialLatticeTailCertificate(
        side, a, p, rho, local.lattice_offset, parameter, terms, normalization, conditions
    )
    if certificate.replay() is not True:
        return None
    remainder = AsymptoticRemainder.big_o(
        sp.Abs(expression) / parameter**terms,
        parameter,
        sp.oo,
        source=(
            "Binomial endpoint lattice theorem: Stieltjes boundary mass, "
            "analytic adjacent-ratio expansion, and geometric domination"
        ),
    )
    return BinomialLatticeTailExpansion(
        expression,
        boundary,
        a,
        local.lattice_offset,
        rho,
        factor,
        side,
        conditions,
        normalization,
        local,
        certificate,
        remainder,
    )
