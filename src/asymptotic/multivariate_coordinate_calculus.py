"""Certified coordinate calculus for multivariate asymptotic germs.

This module supplies exact finite-support dominance cones, correlated monomial
paths, scale-cone comparisons, formal composition/inversion, and chart/atlas
compatibility checks.  The routines return uncertified results
when exact symbolic obligations cannot be discharged.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_limit, bounded_solve_system
from .newton_geometry import NewtonPolyhedralFan, newton_polyhedral_fan


@dataclass(frozen=True)
class CorrelatedPathResult:
    """Certified leading behavior along a correlated monomial path."""

    expression: sp.Expr
    parameter: sp.Symbol
    substitutions: tuple[tuple[sp.Symbol, sp.Expr], ...]
    leading_power: sp.Expr | None
    leading_coefficient: sp.Expr | None
    limit: sp.Expr | None
    certified: bool
    reason: str = ""


@dataclass(frozen=True)
class ScaleCone:
    """Valuations and pairwise ordering of scales along a positive weight vector."""

    variables: tuple[sp.Symbol, ...]
    weights: tuple[sp.Expr, ...]
    constraints: sp.Expr
    valuations: tuple[sp.Expr, ...]
    order: tuple[tuple[int, int, sp.Expr], ...]
    certified: bool


@dataclass(frozen=True)
class CompositionCertificate:
    """Result and proof obligations for a multivariate composition."""

    expression: sp.Expr
    composed: sp.Expr
    variables: tuple[sp.Symbol, ...]
    inner: tuple[sp.Expr, ...]
    certified: bool
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class InverseMapCertificate:
    """Exact local inverse data and its Jacobian proof obligations."""

    variables: tuple[sp.Symbol, ...]
    map: tuple[sp.Expr, ...]
    inverse: tuple[sp.Expr, ...] | None
    jacobian_determinant: sp.Expr
    certified: bool
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChartTransport:
    """Expression transported between two local coordinate charts."""

    expression: sp.Expr
    transported: sp.Expr
    source_variables: tuple[sp.Symbol, ...]
    target_variables: tuple[sp.Symbol, ...]
    transition: tuple[sp.Expr, ...]
    certified: bool
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class AtlasConsistency:
    """Consistency result for local expressions on chart overlaps."""

    consistent: bool | None
    certified: bool
    discrepancies: tuple[sp.Expr, ...]
    checked_pairs: tuple[tuple[int, int], ...]
    obligations: tuple[str, ...] = ()


def newton_dominance_fan(expressions, variables, target=None) -> NewtonPolyhedralFan:
    """Exact positive-weight common Newton fan for finite rational supports."""
    return newton_polyhedral_fan(tuple(expressions), tuple(variables), target)


def correlated_path(expr, variables, weights, *, coefficients=None, parameter=None):
    """Restrict a germ to x_i=c_i*t**w_i and certify its leading monomial."""
    variables, weights = (tuple(variables), tuple(map(sp.sympify, weights)))
    if len(variables) != len(weights) or any(
        w.is_positive is not True for w in weights
    ):
        raise ValueError("one provably positive weight is required per variable")
    t = parameter or sp.Dummy("t", positive=True)
    coeffs = tuple(map(sp.sympify, coefficients or (sp.S.One,) * len(variables)))
    if len(coeffs) != len(variables):
        raise ValueError("coefficient arity mismatch")
    subs = tuple(
        ((x, c * t**w) for x, c, w in zip(variables, coeffs, weights, strict=True))
    )
    restricted = sp.cancel(sp.sympify(expr).subs(dict(subs), simultaneous=True))
    try:
        lead = restricted.as_leading_term(t)
        p = sp.sympify(lead.as_powers_dict().get(t, 0))
        c = sp.simplify(lead / t**p)
        lim = bounded_limit(restricted, t, 0, direction="+", allow_general=True)
        ok = p.is_real is True and lim is not None
        return CorrelatedPathResult(
            restricted,
            t,
            subs,
            p,
            c,
            lim,
            ok,
            "" if ok else "symbolic leading/limit obligation unresolved",
        )
    except (NotImplementedError, ValueError, TypeError):
        return CorrelatedPathResult(
            restricted,
            t,
            subs,
            None,
            None,
            None,
            False,
            "leading term could not be certified",
        )


def generalized_scale_cone(scales, variables, weights, *, constraints=sp.true):
    """Compare generalized scales after a positive monomial valuation substitution."""
    scales, variables = (tuple(map(sp.sympify, scales)), tuple(variables))
    t = sp.Dummy("t", positive=True)
    path = {x: t ** sp.sympify(w) for x, w in zip(variables, weights, strict=True)}
    vals = []
    for s in scales:
        r = sp.cancel(s.subs(path, simultaneous=True))
        try:
            vals.append(sp.sympify(r.as_leading_term(t).as_powers_dict().get(t, 0)))
        except (NotImplementedError, ValueError, TypeError):
            vals.append(sp.nan)
    order = []
    ok = all(v is not sp.nan and v.is_real is True for v in vals)
    if ok:
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                order.append((i, j, sp.sign(sp.simplify(vals[i] - vals[j]))))
        ok = all(o[2] in (-1, 0, 1) for o in order)
    return ScaleCone(
        variables,
        tuple(map(sp.sympify, weights)),
        sp.sympify(constraints),
        tuple(vals),
        tuple(order),
        ok,
    )


def certified_multivariate_compose(expr, variables, inner, *, target=None):
    """Compose a multivariate germ and record target-compatibility obligations."""
    variables, inner = (tuple(variables), tuple(map(sp.sympify, inner)))
    expr = sp.sympify(expr)
    if len(variables) != len(inner):
        raise ValueError("composition arity mismatch")
    target = tuple(map(sp.sympify, target or (sp.S.Zero,) * len(variables)))
    obligations = []
    for g, a in zip(inner, target, strict=True):
        if (
            sp.simplify(
                g.subs(dict.fromkeys(set().union(*(h.free_symbols for h in inner)), 0))
                - a
            )
            != 0
        ):
            obligations.append("inner map target compatibility")
            break
    composed = sp.cancel(expr.subs(dict(zip(variables, inner)), simultaneous=True))
    return CompositionCertificate(
        expr, composed, variables, inner, not obligations, tuple(obligations)
    )


def certified_local_inverse(mapping, variables, *, target=None):
    """Certify an exact symbolic local inverse when the Jacobian is nonzero."""
    mapping, variables = (tuple(map(sp.sympify, mapping)), tuple(variables))
    if len(mapping) != len(variables):
        raise ValueError("inverse requires a square map")
    det = sp.simplify(sp.Matrix(mapping).jacobian(variables).det())
    origin = dict.fromkeys(variables, 0)
    d0 = sp.simplify(det.subs(origin))
    if d0.is_zero is not False:
        return InverseMapCertificate(
            variables, mapping, None, det, False, ("nonzero Jacobian at target",)
        )
    ys = sp.symbols(f"_o_y0:{len(variables)}")
    sols = bounded_solve_system(
        [sp.Eq(y, f) for y, f in zip(ys, mapping)],
        variables,
        allow_general=True,
    )
    complete = [s for s in (sols or ()) if all(v in s for v in variables)]
    if len(complete) != 1:
        return InverseMapCertificate(
            variables, mapping, None, det, False, ("unique symbolic inverse branch",)
        )
    inv = tuple(sp.simplify(complete[0][v]) for v in variables)
    return InverseMapCertificate(variables, mapping, inv, det, True, ())


def transport_chart(
    expr, source_variables, target_variables, transition, *, inverse_transition=None
):
    source_variables, target_variables = (
        tuple(source_variables),
        tuple(target_variables),
    )
    transition = tuple(map(sp.sympify, transition))
    if len(source_variables) != len(transition):
        raise ValueError("transition arity mismatch")
    transported = sp.cancel(
        sp.sympify(expr).subs(
            dict(zip(source_variables, transition)), simultaneous=True
        )
    )
    obligations = []
    if inverse_transition is not None:
        inv = tuple(map(sp.sympify, inverse_transition))
        back = [
            sp.simplify(g.subs(dict(zip(target_variables, inv)), simultaneous=True) - x)
            for g, x in zip(transition, source_variables)
        ]
        if any(v != 0 for v in back):
            obligations.append("transition maps are mutual inverses")
    else:
        obligations.append("inverse transition not supplied")
    return ChartTransport(
        sp.sympify(expr),
        transported,
        source_variables,
        target_variables,
        transition,
        not obligations,
        tuple(obligations),
    )


def check_atlas_consistency(local_expressions, transitions):
    """Check exact equality of local representatives on supplied chart overlaps.

    transitions maps (i,j) to (source_variables, target_variables, substitution),
    where substitution expresses chart-i variables in chart-j variables.
    """
    local = tuple(map(sp.sympify, local_expressions))
    diffs = []
    pairs = []
    obligations = []
    for (i, j), (src, dst, sub) in transitions.items():
        pairs.append((i, j))
        moved = transport_chart(local[i], src, dst, sub)
        d = sp.simplify(moved.transported - local[j])
        diffs.append(d)
        if d != 0 and d.equals(0) is not False:
            obligations.append(f"overlap {i}-{j} equality undecided")
    inconsistent = any(d.equals(0) is False for d in diffs)
    if inconsistent:
        return AtlasConsistency(False, True, tuple(diffs), tuple(pairs), ())
    if obligations:
        return AtlasConsistency(
            None, False, tuple(diffs), tuple(pairs), tuple(obligations)
        )
    return AtlasConsistency(True, True, tuple(diffs), tuple(pairs), ())
