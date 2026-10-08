"""Certified power expansion for an explicit small ratio."""

from __future__ import annotations

import sympy as sp


def certified_power_expand(
    expr, *, small, large, order=3, statement="explicit small-ratio regime"
):
    """Certify a Taylor power expansion under ``small/large = o(1)``."""
    from .stratified_expansion import (
        CertifiedPowerExpansion,
        ExpansionRegime,
        SmallQuantity,
    )

    expr = sp.sympify(expr)
    small = sp.sympify(small)
    large = sp.sympify(large)
    ratio = sp.simplify(small / large)
    u = sp.Dummy("u", positive=True)
    transformed = sp.cancel(sp.together(expr.subs(small, u * large)))
    try:
        trunc = sp.series(transformed, u, 0, order).removeO()
        poly = sp.Poly(sp.expand(trunc), u)
        if poly.degree() >= order:
            return None
    except (
        TypeError,
        ValueError,
        NotImplementedError,
        sp.PoleError,
        sp.PolynomialError,
    ):
        return None
    return CertifiedPowerExpansion(
        expr,
        sp.simplify(trunc.subs(u, ratio)),
        sp.simplify(ratio**order),
        ExpansionRegime((SmallQuantity(ratio),), statement=statement),
        order,
        True,
        "explicit_small_ratio_taylor",
    )
