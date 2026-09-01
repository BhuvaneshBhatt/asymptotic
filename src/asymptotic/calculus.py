from __future__ import annotations

import sympy as sp

from .algebra import AsymptoticElement, asymptotic_element


def differentiate(obj, order: int = 1):
    """Differentiate any supported asymptotic object through the common protocol.

    Native inputs return their native representation; passing an
    :class:`AsymptoticElement` keeps the unified wrapper.
    """
    wrapped = asymptotic_element(obj)
    result = wrapped.differentiate(order=order)
    return result if isinstance(obj, AsymptoticElement) else result.native


def integrate(obj, *, constant: sp.Expr = 0, terms: int | None = None):
    """Integrate any supported asymptotic object through the common protocol."""
    wrapped = asymptotic_element(obj)
    count = 6 if terms is None else int(terms)
    result = wrapped.integrate(constant=constant, terms=count)
    return result if isinstance(obj, AsymptoticElement) else result.native
