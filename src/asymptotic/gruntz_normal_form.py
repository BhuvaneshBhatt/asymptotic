"""Certified finite-height Gruntz/MRV normal form for one-variable germs."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_refine
from .logexp_transseries import (
    RecursiveLogExpMonomial,
    canonical_recursive_logexp_monomial,
)


@dataclass(frozen=True)
class GruntzNormalForm:
    expression: sp.Expr
    variable: sp.Symbol
    point: sp.Expr
    local_variable: sp.Symbol
    transformed: sp.Expr
    monomial: RecursiveLogExpMonomial | None
    positive_side: bool
    exact_identity: bool = True
    mrv_representative: sp.Expr | None = None
    comparability_classes: tuple[tuple[sp.Expr, ...], ...] = ()
    provider: str = "gruntz_normal_form"


def _branch_safe_power_rewrite(e, t):
    repl = {}
    for p in e.atoms(sp.Pow):
        base, exponent = p.as_base_exp()
        if t in exponent.free_symbols and (
            base.is_positive is True or (base == t and t.is_positive)
        ):
            repl[p] = sp.exp(exponent * sp.log(base))
    return e.xreplace(repl)


def gruntz_normal_form(
    expr, variable, *, point=sp.oo, direction="+", assumptions=sp.S.true
):
    expr = bounded_refine(sp.sympify(expr), sp.sympify(assumptions))
    variable = sp.sympify(variable)
    point = sp.sympify(point)
    t = sp.Dummy("gruntz_t", positive=True)
    if point is sp.oo:
        transformed = expr.xreplace({variable: 1 / t})
        positive = True
    elif point is -sp.oo:
        transformed = expr.xreplace({variable: -1 / t})
        positive = True
    else:
        sign = 1 if direction == "+" else -1
        transformed = expr.xreplace({variable: point + sign * t})
        positive = direction == "+"
    transformed = _branch_safe_power_rewrite(
        sp.powsimp(sp.cancel(transformed), force=False), t
    )
    transformed = sp.powsimp(sp.factor_terms(transformed), force=False)
    try:
        mono = canonical_recursive_logexp_monomial(transformed, t)
    except (TypeError, ValueError, NotImplementedError):
        mono = None
    rep = None
    classes = ()
    # Compute MRV classes in the original infinity coordinate.  This is
    # metadata used to order normalization; it is never itself a proof.
    try:
        from .mrv import mrv_decomposition

        inf_expr = transformed.xreplace({t: 1 / variable})
        dec = mrv_decomposition(inf_expr, variable, sp.oo, assumptions=assumptions)
        rep = dec.representative
        classes = tuple(tuple(c.members) for c in dec.classes)
    except (ImportError, TypeError, ValueError, NotImplementedError, AttributeError):
        pass
    # x -> +/-1/t or point +/- t is an exact substitution, so downstream
    # certification can trust the normalized identity without trusting MRV.
    return GruntzNormalForm(
        expr, variable, point, t, transformed, mono, positive, True, rep, classes
    )


__all__ = ["GruntzNormalForm", "gruntz_normal_form"]
