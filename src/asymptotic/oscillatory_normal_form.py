"""Exact normalization of common periodic oscillatory germs and cluster intervals."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class OscillatoryClusterInterval:
    lower: sp.Expr
    upper: sp.Expr
    all_intermediate: bool
    phase: sp.Expr
    provider: str = "oscillatory_interval"
    exact_image: sp.Set | None = None

    @property
    def set(self):
        return (
            self.exact_image
            if self.exact_image is not None
            else sp.Interval(self.lower, self.upper)
        )


def oscillatory_cluster_interval(expr, variable):
    expr = sp.expand_trig(sp.sympify(expr))
    variable = sp.sympify(variable)
    # Exact affine sin/cos image; shared single phase only.
    a = sp.Wild("a", exclude=[variable])
    b = sp.Wild("b", exclude=[variable])
    q = sp.Wild("q")
    for fn in (sp.sin, sp.cos):
        m = expr.match(a + b * fn(q))
        if (
            m
            and variable in m[q].free_symbols
            and variable not in m[a].free_symbols
            and variable not in m[b].free_symbols
        ):
            amp = sp.Abs(m[b])
            return OscillatoryClusterInterval(
                sp.simplify(m[a] - amp), sp.simplify(m[a] + amp), True, m[q]
            )
    # Exact commensurate trigonometric polynomial.  Keep the compact
    # continuous image symbolically instead of forcing radical expressions for
    # high-degree extrema; this is an exact joint range, not a numeric enclosure.
    atoms = tuple(expr.atoms(sp.sin, sp.cos))
    if atoms:
        phases = [a.args[0] for a in atoms]
        q0 = phases[0]
        ratios = [sp.simplify(q / q0) for q in phases]
        if all(r.is_Integer for r in ratios) and variable in q0.free_symbols:
            theta = sp.Dummy("periodic_phase", real=True)
            # Current certified normalization requires the common phase to be x up to
            # a nonzero integer factor.
            base_ratio = sp.simplify(q0 / variable)
            if base_ratio.is_Integer and base_ratio != 0:
                normalized = expr.xreplace({variable: theta})
                image = sp.ImageSet(
                    sp.Lambda(theta, normalized), sp.Interval(0, 2 * sp.pi)
                )
                lo = sp.Function("PeriodicMinimum")(normalized)
                hi = sp.Function("PeriodicMaximum")(normalized)
                return OscillatoryClusterInterval(
                    lo, hi, True, theta, exact_image=image
                )
    return None
