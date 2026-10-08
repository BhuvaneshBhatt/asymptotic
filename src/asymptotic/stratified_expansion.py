"""Certified finite stratifications for elementary multivariate power expansions.

The stratified expansion engine covers finite targets and power scales.  It discovers
only scale comparisons that occur in the expression and delegates each strict
small-ratio cell to an ordinary Taylor expansion in a certified infinitesimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .proof_obligations import ObligationKind, ProofObligation


class RegimeRelation(Enum):
    LITTLE_O = "o"
    COMPARABLE = "theta"


@dataclass(frozen=True)
class SmallQuantity:
    expression: sp.Expr
    relation: RegimeRelation = RegimeRelation.LITTLE_O


@dataclass(frozen=True)
class ExpansionRegime:
    facts: tuple[SmallQuantity, ...] = ()
    condition: sp.Expr = sp.S.true
    scaled_coordinates: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()
    statement: str = ""


@dataclass(frozen=True)
class CertifiedPowerExpansion:
    expression: sp.Expr
    approximation: sp.Expr
    remainder_scale: sp.Expr
    regime: ExpansionRegime
    order: int
    certified: bool
    method: str
    obligations: tuple[ProofObligation, ...] = ()


@dataclass(frozen=True)
class StratificationCoverageCertificate:
    certified: bool
    statement: str
    conditions: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class StratifiedExpansion:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    branches: tuple[object, ...]
    coverage_certificate: StratificationCoverageCertificate
    obligations: tuple[ProofObligation, ...] = ()

    @property
    def certified(self) -> bool:
        return self.coverage_certificate.certified and all(
            getattr(b, "certified", False) for b in self.branches
        )


def _series_in_small_ratio(expr, small, large, order, statement):
    from .power_expansion import certified_power_expand

    return certified_power_expand(
        expr, small=small, large=large, order=order, statement=statement
    )


def _comparable_cell(expr, small, large, order):
    lam = sp.Dummy("lambda", positive=True)
    transformed = sp.simplify(expr.subs(small, lam * large))
    # The scaled coordinate is retained exactly.  This is a certified power
    # representation of the comparable cell, not a Taylor series in lambda.
    approximation = transformed.subs(lam, small / large)
    return CertifiedPowerExpansion(
        expr,
        approximation,
        sp.S.Zero,
        ExpansionRegime(
            condition=sp.S.true,
            scaled_coordinates=((lam, sp.simplify(small / large)),),
            statement=f"{sp.sstr(small)} and {sp.sstr(large)} are comparable",
        ),
        order,
        True,
        "comparable_scaled_coordinate",
    )


def _two_scale_stratification(expr, variables, target, order):
    if len(variables) != 2 or any(t != 0 for t in target):
        return None
    x, y = variables
    if not (expr.has(x) and expr.has(y)):
        return None
    xy = _series_in_small_ratio(expr, x, y, order, f"{x}=o({y})")
    yx = _series_in_small_ratio(expr, y, x, order, f"{y}=o({x})")
    from .multivariate_transseries import certified_transseries_expand
    from .oscillatory_scales import certified_oscillatory_expand

    if xy is None:
        xy = certified_oscillatory_expand(
            expr, small=x, large=y, order=order, statement=f"{x}=o({y})"
        )
    if yx is None:
        yx = certified_oscillatory_expand(
            expr, small=y, large=x, order=order, statement=f"{y}=o({x})"
        )
    if xy is None:
        xy = certified_transseries_expand(
            expr, small=x, large=y, order=order, statement=f"{x}=o({y})"
        )
    if yx is None:
        yx = certified_transseries_expand(
            expr, small=y, large=x, order=order, statement=f"{y}=o({x})"
        )
    middle = _comparable_cell(expr, x, y, order)
    branches = tuple(branch for branch in (xy, middle, yx) if branch is not None)
    complete = xy is not None and yx is not None
    coverage = StratificationCoverageCertificate(
        complete,
        (
            "positive magnitudes satisfy the three asymptotic scale classes"
            if complete
            else "at least one strict scale cell is outside the certified power/log-exp/oscillatory hierarchy"
        ),
        (sp.S.true, sp.S.true, sp.S.true),
    )
    obligations = (
        ()
        if complete
        else (
            ProofObligation(
                ObligationKind.ORDER_RELATION,
                "a discovered scale cell is outside the certified power/log-exp/oscillatory hierarchy",
                provider="two_scale_stratification",
                expression=sp.sstr(expr),
            ),
        )
    )
    return StratifiedExpansion(expr, variables, target, branches, coverage, obligations)


def _symbolic_power_fraction(expr, variable, target, order):
    """Stratify x**a/(x**b+x**c) on the critical comparison b ? c.

    This theorem creates the symbolic exponent hypersurface that changes the
    denominator's dominant monomial.
    """
    if len(variable) != 1 or target != (sp.S.Zero,):
        return None
    x = variable[0]
    num, den = sp.fraction(sp.cancel(expr))
    terms = sp.Add.make_args(sp.expand(den))
    if len(terms) != 2:
        return None

    def power(t):
        coeff, exponent = t.as_coeff_exponent(x)
        return coeff, exponent

    c1, b = power(terms[0])
    c2, c = power(terms[1])
    cn, a = power(num)
    if any(v.has(x) for v in (c1, c2, cn, a, b, c)) or c1 != 1 or c2 != 1:
        return None
    params = (a.free_symbols | b.free_symbols | c.free_symbols) - {x}
    if not params:
        return None
    branches = []
    # b<c: factor x**b and expand 1/(1+x**(c-b))
    for cond, lead, delta, label in (
        (sp.Lt(b, c), b, c - b, f"{b} < {c}"),
        (sp.Gt(b, c), c, b - c, f"{b} > {c}"),
    ):
        approx = sp.Add(
            *[(-1) ** k * cn * x ** (a - lead + k * delta) for k in range(order)]
        )
        branches.append(
            CertifiedPowerExpansion(
                expr,
                approx,
                x ** (a - lead + order * delta),
                ExpansionRegime(condition=cond, statement=label),
                order,
                True,
                "symbolic_exponent_geometric",
            )
        )
    eq = sp.Eq(b, c)
    branches.insert(
        1,
        CertifiedPowerExpansion(
            expr,
            sp.simplify(cn * x ** (a - b) / 2),
            sp.S.Zero,
            ExpansionRegime(condition=eq, statement=f"{b} = {c}"),
            order,
            True,
            "symbolic_exponent_equal",
        ),
    )
    return StratifiedExpansion(
        expr,
        variable,
        target,
        tuple(branches),
        StratificationCoverageCertificate(
            True,
            "trichotomy of the real critical exponent comparison",
            (sp.Lt(b, c), eq, sp.Gt(b, c)),
        ),
    )


def stratified_expand(expr, variables, *, target=None, order=3, assumptions=sp.S.true):
    """Discover a finite certified power/log-exp/oscillatory stratification.

    The multivariate transseries engine supports two competing finite-target power/log-exp scales and the critical
    symbolic-exponent denominator family. Unsupported geometry returns a
    structured obligation rather than an uncertified formal expansion.
    """
    expr = sp.sympify(expr)
    assumptions = sp.sympify(assumptions)
    variables = (variables,) if isinstance(variables, sp.Symbol) else tuple(variables)
    target = (
        tuple(sp.S.Zero for _ in variables)
        if target is None
        else tuple(map(sp.sympify, target))
    )
    if order < 1:
        raise ValueError("order must be positive")
    if assumptions is sp.S.false:
        raise ValueError("assumptions must describe a satisfiable context")
    result = _symbolic_power_fraction(expr, variables, target, order)
    if result is None:
        result = _two_scale_stratification(expr, variables, target, order)
    if result is None and len(variables) >= 3:
        from .nscale_stratification import nscale_stratified_expand

        result = nscale_stratified_expand(expr, variables, target=target, order=order)
    if result is not None:
        return result
    obligation = ProofObligation(
        ObligationKind.ORDER_RELATION,
        "no finite dominance stratification was certified for the requested expression",
        provider="stratified_expand",
        expression=sp.sstr(expr),
    )
    return StratifiedExpansion(
        expr,
        variables,
        target,
        (),
        StratificationCoverageCertificate(False, "coverage not proved", ()),
        (obligation,),
    )
