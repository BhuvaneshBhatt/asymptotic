"""Multivariate matched/PDE asymptotic geometry.

This module certifies anisotropic stretched-coordinate balances for linear PDE
operators and provides explicit overlap/matching and inclusion--exclusion
composite construction.  It does not pretend to solve arbitrary PDEs: layer
profiles remain caller- or solver-supplied and every balance is auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.core.function import AppliedUndef

from ._symbolic_policy import bounded_limit


@dataclass(frozen=True)
class PDELayerTerm:
    """One differential-operator term under an anisotropic layer scaling."""

    derivative_orders: tuple[int, ...]
    coefficient: sp.Expr
    epsilon_order: sp.Expr
    stretched_order: sp.Expr


@dataclass(frozen=True)
class PDELayerBalance:
    """Dominant terms and proof obligations for a PDE layer scaling."""

    variables: tuple[sp.Symbol, ...]
    weights: tuple[sp.Expr, ...]
    terms: tuple[PDELayerTerm, ...]
    dominant_indices: tuple[int, ...]
    certified: bool
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchingCertificate:
    """Comparison of outer and inner overlap limits."""

    outer_overlap: sp.Expr | None
    inner_overlap: sp.Expr | None
    discrepancy: sp.Expr | None
    certified: bool
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompositeExpansion:
    """Matched composite formed from an outer solution and local layers."""

    expression: sp.Expr
    outer: sp.Expr
    layers: tuple[sp.Expr, ...]
    overlaps: tuple[sp.Expr, ...]
    certified: bool
    obligations: tuple[str, ...] = ()


def _epsilon_order(expr, eps):
    expr = sp.cancel(sp.sympify(expr))
    try:
        lead = expr.as_leading_term(eps)
        p = sp.sympify(lead.as_powers_dict().get(eps, 0))
        return p if p.is_real is True else None
    except (NotImplementedError, ValueError, TypeError, sp.PoleError):
        return None


def linear_pde_layer_balance(
    equation,
    unknown: AppliedUndef,
    variables: Sequence[sp.Symbol],
    epsilon: sp.Symbol,
    weights: Sequence[sp.Expr],
):
    """Certify an anisotropic layer scaling x_i=epsilon**alpha_i X_i.

    The residual must be linear in ``unknown`` and its derivatives.  A valid
    distinguished layer has at least two terms tied at the least epsilon
    valuation after derivative rescaling.
    """
    equation = sp.expand(sp.sympify(equation))
    variables = tuple(variables)
    weights = tuple(map(sp.sympify, weights))
    if len(variables) != len(weights):
        raise ValueError("one layer weight is required per variable")
    if any(w.is_nonnegative is not True for w in weights):
        raise ValueError("layer weights must be provably nonnegative")
    atoms = sorted(
        [d for d in equation.atoms(sp.Derivative) if d.expr == unknown], key=str
    )
    basis = [unknown, *atoms]
    dummies = sp.symbols(f"_q_D0:{len(basis)}")
    alg = equation.xreplace(dict(zip(basis, dummies, strict=True)))
    try:
        poly = sp.Poly(alg, *dummies)
    except sp.PolynomialError:
        return PDELayerBalance(
            variables, weights, (), (), False, ("linear differential operator",)
        )
    if poly.total_degree() > 1:
        return PDELayerBalance(
            variables, weights, (), (), False, ("linear differential operator",)
        )
    terms = []
    obligations = []
    for b, dummy in zip(basis, dummies, strict=True):
        coeff = sp.expand(poly.coeff_monomial(dummy))
        if coeff == 0:
            continue
        eo = _epsilon_order(coeff, epsilon)
        if eo is None:
            obligations.append("coefficient epsilon valuation")
            continue
        counts = tuple(0 if b == unknown else b.variables.count(v) for v in variables)
        stretched = sp.simplify(
            eo - sum((w * c for w, c in zip(weights, counts, strict=True)))
        )
        terms.append(PDELayerTerm(counts, coeff, eo, stretched))
    if not terms:
        obligations.append("nonzero differential terms")
    vals = [t.stretched_order for t in terms]
    dominant = ()
    if vals:
        candidates = []
        for i, value in enumerate(vals):
            differences = [sp.simplify(other - value) for other in vals]
            if all(diff.is_nonnegative is True for diff in differences):
                candidates.append(i)
        if candidates:
            minv = vals[candidates[0]]
            dominant = tuple(
                i for i, value in enumerate(vals) if sp.simplify(value - minv) == 0
            )
        else:
            obligations.append("comparable stretched valuations")
    if len(dominant) < 2:
        obligations.append("distinguished balance requires at least two dominant terms")
    return PDELayerBalance(
        variables,
        weights,
        tuple(terms),
        dominant,
        not obligations,
        tuple(dict.fromkeys(obligations)),
    )


def match_overlap(
    outer, inner, outer_variable, inner_variable, *, outer_limit=0, inner_limit=sp.oo
):
    """Compare outer and inner overlap limits exactly when SymPy can prove them."""
    outer, inner = map(sp.sympify, (outer, inner))
    obligations = []
    ol = bounded_limit(
        outer, outer_variable, outer_limit, direction="+", allow_general=True
    )
    il = bounded_limit(inner, inner_variable, inner_limit, allow_general=True)
    if ol is None or il is None:
        return MatchingCertificate(ol, il, None, False, ("overlap limits",))
    d = sp.simplify(ol - il)
    if d != 0:
        obligations.append("outer and inner overlap agree")
    return MatchingCertificate(ol, il, d, not obligations, tuple(obligations))


def composite_expansion(outer, layers, overlaps, *, matching_certificates=()):
    """Construct the first inclusion--exclusion composite outer+layers-overlaps."""
    outer = sp.sympify(outer)
    layers = tuple(map(sp.sympify, layers))
    overlaps = tuple(map(sp.sympify, overlaps))
    if len(layers) != len(overlaps):
        raise ValueError("one overlap subtraction is required per layer")
    obligations = []
    for cert in matching_certificates:
        if not cert.certified:
            obligations.append("all supplied overlap matches are certified")
    expr = sp.simplify(outer + sum(layers, sp.S.Zero) - sum(overlaps, sp.S.Zero))
    return CompositeExpansion(
        expr,
        outer,
        layers,
        overlaps,
        not obligations,
        tuple(dict.fromkeys(obligations)),
    )
