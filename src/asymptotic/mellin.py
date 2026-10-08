"""Mellin-transform asymptotics from certified meromorphic pole data."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_polynomial_roots


@dataclass(frozen=True)
class MellinResult:
    """Finite inverse-Mellin residue expansion."""

    expression: sp.Expr | None
    transform: sp.Expr | None
    poles: tuple[sp.Expr, ...]
    certified: bool
    obligations: tuple[str, ...] = ()


def mellin(
    expr: sp.Expr,
    variable: sp.Symbol,
    transform_variable: sp.Symbol,
    *,
    toward_zero: bool = True,
    max_poles: int = 4,
) -> MellinResult:
    """Extract inverse-Mellin asymptotic terms from explicit transform poles.

    Only transforms returned explicitly by SymPy with a fundamental strip are
    accepted. Pole enumeration is restricted to finite polynomial/rational
    denominators; unresolved Gamma lattices and contour-growth conditions are
    reported as obligations instead of being guessed.
    """
    expr = sp.sympify(expr)
    variable = sp.sympify(variable)
    s = sp.sympify(transform_variable)
    try:
        transformed, _strip, _ = sp.mellin_transform(expr, variable, s)
    except (TypeError, ValueError, NotImplementedError):
        return MellinResult(None, None, (), False, ("explicit Mellin transform",))
    den = sp.denom(sp.cancel(transformed))
    if transformed.has(sp.gamma, sp.polygamma, sp.zeta):
        return MellinResult(
            None, transformed, (), False, ("complete meromorphic pole enumeration",)
        )
    try:
        poly = sp.Poly(den, s)
        roots = bounded_polynomial_roots(poly.as_expr(), s)
    except (sp.PolynomialError, NotImplementedError):
        return MellinResult(
            None, transformed, (), False, ("finite meromorphic pole set",)
        )
    if roots is None:
        return MellinResult(
            None, transformed, (), False, ("finite pole enumeration budget",)
        )
    if not roots:
        return MellinResult(sp.S.Zero, transformed, (), True)
    # For x->0 shift left; for x->oo shift right. Sort only provably real poles.
    if not all(r.is_real is True for r in roots):
        return MellinResult(None, transformed, roots, False, ("real pole ordering",))
    ordered = sorted(roots, key=lambda r: float(sp.N(r)), reverse=not toward_zero)[
        :max_poles
    ]
    terms = []
    for pole in ordered:
        try:
            residue = sp.residue(transformed * variable ** (-s), s, pole)
        except (NotImplementedError, ValueError):
            return MellinResult(
                None, transformed, tuple(ordered), False, ("residue evaluation",)
            )
        terms.append(residue)
    return MellinResult(sp.simplify(sum(terms)), transformed, tuple(ordered), True)
