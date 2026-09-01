from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import sympy as sp

from ._power_simplify import analytic_powsimp
from ._symbolic_errors import SYMBOLIC_ERRORS
from .context import AsymptoticContext, context_for


class OscillationKind(Enum):
    PERIODIC = auto()
    BOUNDED_OSCILLATORY = auto()
    UNBOUNDED_PERIODIC = auto()


@dataclass(frozen=True)
class OscillatoryFactor:
    """A bounded or periodic factor separated from its growth envelope.

    ``template`` is a function of ``phase_symbol``.  Substituting ``phase`` for
    that symbol reconstructs ``expr``.  ``period`` is measured in the phase
    variable, not necessarily in the original asymptotic variable.
    """

    expr: sp.Expr
    phase: sp.Expr
    phase_symbol: sp.Symbol
    template: sp.Expr
    period: sp.Expr | None
    lower_bound: sp.Expr | None
    upper_bound: sp.Expr | None
    all_intermediate_values: bool | None
    kind: OscillationKind = OscillationKind.PERIODIC

    def reconstruct(self) -> sp.Expr:
        return self.template.xreplace({self.phase_symbol: self.phase})

    @property
    def bounded(self) -> bool:
        return (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound not in (-sp.oo, sp.oo)
            and self.upper_bound not in (-sp.oo, sp.oo)
        )


@dataclass(frozen=True)
class PeriodicDecomposition:
    """Separate asymptotically ordered growth from oscillatory factors."""

    expr: sp.Expr
    envelope: sp.Expr
    factors: tuple[OscillatoryFactor, ...]

    @property
    def has_oscillation(self) -> bool:
        return bool(self.factors)

    @property
    def oscillatory_part(self) -> sp.Expr:
        if not self.factors:
            return sp.S.One
        return sp.Mul(*(factor.expr for factor in self.factors))

    def reconstruct(self) -> sp.Expr:
        return analytic_powsimp(self.envelope * self.oscillatory_part)


def _known_period_and_bounds(template: sp.Expr, t: sp.Symbol):
    """Return (period, lower, upper, all-values) for safe known templates."""

    if template == sp.sin(t) or template == sp.cos(t):
        return 2 * sp.pi, -sp.S.One, sp.S.One, True
    if template == sp.tan(t) or template == sp.cot(t):
        return sp.pi, -sp.oo, sp.oo, True
    if template == sp.sec(t) or template == sp.csc(t):
        return 2 * sp.pi, -sp.oo, sp.oo, False

    # Monotone outer compositions preserve periodicity and allow exact range
    # propagation when their inner range is finite.
    if template.func is sp.exp and len(template.args) == 1:
        inner = _known_period_and_bounds(template.args[0], t)
        if inner is not None:
            p, lo, hi, allq = inner
            if lo not in (-sp.oo, sp.oo) and hi not in (-sp.oo, sp.oo):
                return p, sp.exp(lo), sp.exp(hi), allq

    if template.is_Pow and template.exp.is_integer:
        inner = _known_period_and_bounds(template.base, t)
        if inner is not None:
            p, lo, hi, allq = inner
            n = int(template.exp)
            if lo not in (-sp.oo, sp.oo) and hi not in (-sp.oo, sp.oo):
                if n < 0 and lo <= 0 <= hi:
                    # Reciprocal powers acquire poles wherever the inner
                    # periodic function vanishes.  Periodicity survives, but
                    # endpoint propagation cannot provide finite bounds.
                    return p, None, None, False
                vals = [sp.simplify(lo**n), sp.simplify(hi**n)]
                if n > 0 and n % 2 == 0 and lo <= 0 <= hi:
                    vals.append(sp.S.Zero)
                return p, min(vals), max(vals), allq

    # SymPy's exact periodicity detector is useful for algebraic combinations
    # such as sin(t)+cos(2*t).  Bounds are intentionally left unknown unless
    # they are established structurally above.
    try:
        period = sp.calculus.util.periodicity(template, t)
    except SYMBOLIC_ERRORS:
        period = None
    if period not in (None, sp.S.Zero):
        return period, None, None, None
    return None


def _candidate_factor(node: sp.Expr, x: sp.Symbol) -> OscillatoryFactor | None:
    """Recognize a maximal phase-composed periodic expression."""

    # A valid common phase leaves a template independent of x after replacement
    # by a dummy symbol. Candidate subexpressions are tested against that invariant.
    candidates = [sub for sub in sp.preorder_traversal(node) if sub != node and sub.has(x)]
    # Prefer larger phases; direct arguments such as log(x) are usually found
    # before their internal x.
    candidates.sort(key=sp.count_ops, reverse=True)
    for phase in candidates:
        t = sp.Dummy("theta", real=True)
        template = node.xreplace({phase: t})
        if template.has(x) or not template.has(t):
            continue
        info = _known_period_and_bounds(template, t)
        if info is None:
            continue
        period, lo, hi, allq = info
        kind = (
            OscillationKind.UNBOUNDED_PERIODIC
            if lo in (-sp.oo, sp.oo) or hi in (-sp.oo, sp.oo)
            else OscillationKind.PERIODIC
        )
        return OscillatoryFactor(node, phase, t, template, period, lo, hi, allq, kind)

    # Direct sin(x), cos(x), etc. have phase x itself.
    t = sp.Dummy("theta", real=True)
    template = node.xreplace({x: t})
    if template.has(t):
        info = _known_period_and_bounds(template, t)
        if info is not None:
            period, lo, hi, allq = info
            kind = (
                OscillationKind.UNBOUNDED_PERIODIC
                if lo in (-sp.oo, sp.oo) or hi in (-sp.oo, sp.oo)
                else OscillationKind.PERIODIC
            )
            return OscillatoryFactor(node, x, t, template, period, lo, hi, allq, kind)
    return None


def periodic_decomposition(
    expr: sp.Expr,
    variable: sp.Symbol,
    *,
    point: sp.Expr = sp.oo,
    context: AsymptoticContext | None = None,
) -> PeriodicDecomposition:
    """Extract multiplicative periodic/oscillatory factors conservatively.

    The routine deliberately does *not* declare periodic functions to belong to
    an ordered asymptotic scale.  Instead they remain exact coefficient-like
    factors while ``envelope`` is passed to growth-scale discovery.
    """

    expr = analytic_powsimp(sp.sympify(expr))
    _ = context_for(variable, point, context)

    if expr.is_Mul:
        factors = []
        envelope_parts = []
        for part in sp.Mul.make_args(expr):
            factor = _candidate_factor(part, variable)
            if factor is None or not factor.bounded:
                envelope_parts.append(part)
            else:
                factors.append(factor)
        envelope = analytic_powsimp(sp.Mul(*envelope_parts))
        return PeriodicDecomposition(expr, envelope, tuple(factors))

    factor = _candidate_factor(expr, variable)
    if factor is not None and factor.bounded:
        return PeriodicDecomposition(expr, sp.S.One, (factor,))
    return PeriodicDecomposition(expr, expr, ())


def periodic_bounds(expr: sp.Expr, variable: sp.Symbol):
    """Return exact known bounds for a phase-periodic expression, if available."""

    factor = _candidate_factor(sp.sympify(expr), variable)
    if factor is None:
        return None
    return factor.lower_bound, factor.upper_bound, factor.all_intermediate_values
