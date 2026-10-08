"""Unified entry point for real, sectorial, and oscillatory stratification."""

from __future__ import annotations

import sympy as sp

from .complex_domain import ComplexBranchMetadata, ComplexSector
from .sectorial_transseries import branch_aware_rewrite
from .stokes_transseries import dominance_sector_atlas
from .stratified_expansion import stratified_expand


def unified_stratified_expand(
    expr,
    variables,
    *,
    target=None,
    order=3,
    sector: ComplexSector | None = None,
    branch: ComplexBranchMetadata | None = None,
):
    """Route one request through the real or complex certified stratifier.

    A supplied complex sector selects branch-aware sectorial semantics.  For a
    sum of two exponential scales, an exact Stokes dominance atlas is returned;
    otherwise branch-aware LE rewriting is used.  Without a sector the existing
    real power/log-exp/oscillatory stratifier remains authoritative.
    """
    expr = sp.sympify(expr)
    variables = (variables,) if isinstance(variables, sp.Symbol) else tuple(variables)
    if sector is None:
        return stratified_expand(expr, variables, target=target, order=order)
    if len(variables) != 1:
        return branch_aware_rewrite(
            expr, variables[0], sector=sector, branch=branch or ComplexBranchMetadata()
        )
    terms = sp.Add.make_args(expr)
    if len(terms) == 2 and all(term.func is sp.exp for term in terms):
        return dominance_sector_atlas(terms[0], terms[1], variables[0])
    return branch_aware_rewrite(
        expr, variables[0], sector=sector, branch=branch or ComplexBranchMetadata()
    )
