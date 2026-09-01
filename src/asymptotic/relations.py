"""Asymptotic relations with conservative directed-ray reduction.

Univariate decisions are made by the shared :class:`AsymptoticContext` growth
and limit oracles.  At finite multivariate points, deterministic rays are used
only as a falsification device: one failing ray proves the proposed relation
false, while agreement on finitely many rays is intentionally reported as
undecided rather than as a proof of a multivariate limit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import sympy as sp

from ._symbolic_policy import bounded_ask
from .context import AsymptoticContext, GrowthComparison

RelationKind = Literal[
    "equivalent",
    "equal",
    "less",
    "less-equal",
    "greater",
    "greater-equal",
    "little-o",
    "big-o",
    "same-order",
]


@dataclass(frozen=True)
class DirectedRelationEvidence:
    direction: tuple[sp.Expr, ...]
    value: bool | None
    ratio_limit: sp.Expr | None = None


@dataclass(frozen=True)
class AsymptoticRelationResult:
    relation: RelationKind
    left: sp.Expr
    right: sp.Expr
    variables: tuple[sp.Symbol, ...]
    points: tuple[sp.Expr, ...]
    value: bool | None
    certified: bool
    evidence: tuple[DirectedRelationEvidence, ...] = ()
    reason: str = ""


def _univariate_relation(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    relation: RelationKind,
    *,
    direction: str = "+",
    assumptions: sp.Expr | bool = sp.S.true,
) -> tuple[bool | None, sp.Expr | None]:
    """Decide one directed univariate relation and retain the evidence used."""
    left = sp.refine(sp.sympify(left), assumptions)
    right = sp.refine(sp.sympify(right), assumptions)
    ctx = AsymptoticContext(variable, point=point, direction=direction)
    if ctx.is_zero(right) is True:
        if relation == "equivalent":
            return (True, sp.S.One) if ctx.is_zero(left) is True else (False, None)
        return None, None
    if relation == "equivalent":
        ratio = ctx.limit(left / right)
        equality = (
            sp.refine(sp.Eq(ratio, 1), assumptions) if not isinstance(ratio, sp.Limit) else None
        )
        if ratio == 1 or equality is sp.S.true:
            return True, ratio
        if equality is sp.S.false:
            return False, ratio
        return None, ratio
    comparison, ratio = ctx.compare_growth(left, right)
    limit_ratio = ctx.limit(left / right)
    decision_ratio = ratio if ratio is not None else limit_ratio
    if decision_ratio is not None and not isinstance(decision_ratio, sp.Limit):
        zero = bounded_ask(sp.Q.zero(decision_ratio), assumptions)
        finite = bounded_ask(sp.Q.finite(decision_ratio), assumptions)
        nonzero = bounded_ask(sp.Q.nonzero(decision_ratio), assumptions)
        if relation in {"little-o", "less"} and zero is True:
            return True, decision_ratio
        if relation in {"big-o", "less-equal"} and finite is True:
            return True, decision_ratio
        if relation in {"same-order", "equal"} and finite is True and nonzero is True:
            return True, decision_ratio
    if relation in {"little-o", "less"}:
        if comparison is GrowthComparison.SMALLER:
            return True, ratio
        if comparison in {GrowthComparison.SAME_ORDER, GrowthComparison.LARGER}:
            return False, ratio
        return None, ratio
    if relation in {"big-o", "less-equal"}:
        if comparison in {GrowthComparison.SMALLER, GrowthComparison.SAME_ORDER}:
            return True, ratio
        if comparison is GrowthComparison.LARGER:
            return False, ratio
        return None, ratio
    if relation == "greater":
        if comparison is GrowthComparison.LARGER:
            return True, ratio
        if comparison in {GrowthComparison.SAME_ORDER, GrowthComparison.SMALLER}:
            return False, ratio
        return None, ratio
    if relation == "greater-equal":
        if comparison in {GrowthComparison.LARGER, GrowthComparison.SAME_ORDER}:
            return True, ratio
        if comparison is GrowthComparison.SMALLER:
            return False, ratio
        return None, ratio
    if relation in {"same-order", "equal"}:
        if comparison is GrowthComparison.SAME_ORDER:
            return True, ratio
        if comparison in {GrowthComparison.SMALLER, GrowthComparison.LARGER}:
            return False, ratio
        return None, ratio
    raise ValueError(f"unknown asymptotic relation {relation!r}")


def _real_directions(dimension: int, samples: int) -> tuple[tuple[sp.Expr, ...], ...]:
    vectors: list[tuple[sp.Expr, ...]] = []
    for index in range(dimension):
        unit = [sp.S.Zero] * dimension
        unit[index] = sp.S.One
        vectors.append(tuple(unit))
        unit[index] = -sp.S.One
        vectors.append(tuple(unit))
    rng = random.Random(1234)
    while len(vectors) < 2 * dimension + samples:
        vector = tuple(sp.Integer(rng.randint(-16, 16)) for _ in range(dimension))
        if any(component != 0 for component in vector) and vector not in vectors:
            vectors.append(vector)
    return tuple(vectors)


def _complex_directions(dimension: int, samples: int) -> tuple[tuple[sp.Expr, ...], ...]:
    vectors = list(_real_directions(dimension, 0))
    for index in range(dimension):
        for unit_value in (sp.I, -sp.I):
            unit = [sp.S.Zero] * dimension
            unit[index] = unit_value
            vectors.append(tuple(unit))
    rng = random.Random(1234)
    while len(vectors) < 4 * dimension + samples:
        vector = tuple(
            sp.Integer(rng.randint(-16, 16)) + sp.I * sp.Integer(rng.randint(-16, 16))
            for _ in range(dimension)
        )
        if any(component != 0 for component in vector) and vector not in vectors:
            vectors.append(vector)
    return tuple(vectors)


def asymptotic_relation(
    left: sp.Expr,
    right: sp.Expr,
    variables: sp.Symbol | tuple[sp.Symbol, ...] | list[sp.Symbol],
    points: sp.Expr | tuple[sp.Expr, ...] | list[sp.Expr],
    *,
    relation: RelationKind = "equivalent",
    directions: Literal["real", "complex"] = "real",
    ray_samples: int = 8,
    assumptions: sp.Expr | bool = sp.S.true,
) -> AsymptoticRelationResult:
    """Decide or conservatively test an asymptotic relation.

    For one variable at a finite real point, ``directions="real"`` checks both
    one-sided germs and certifies ``True`` only when both sides agree.  For
    several variables, coordinate and deterministic pseudo-random rays can
    *disprove* a relation, but finite ray sampling never certifies a positive
    multivariate statement.
    """

    left = sp.sympify(left)
    right = sp.sympify(right)
    vars_tuple = (
        tuple(variables) if isinstance(variables, (tuple, list, sp.Tuple)) else (variables,)
    )
    pts_tuple = tuple(points) if isinstance(points, (tuple, list, sp.Tuple)) else (points,)
    if len(vars_tuple) != len(pts_tuple):
        raise ValueError("variables and points must have the same length")
    if not vars_tuple or not all(isinstance(variable, sp.Symbol) for variable in vars_tuple):
        raise TypeError("variables must be SymPy symbols")
    if ray_samples < 0:
        raise ValueError("ray_samples must be nonnegative")

    if len(vars_tuple) == 1:
        variable = vars_tuple[0]
        point = sp.sympify(pts_tuple[0])
        if point in (sp.oo, -sp.oo):
            value, ratio = _univariate_relation(
                left, right, variable, point, relation, assumptions=assumptions
            )
            evidence = (DirectedRelationEvidence((sp.S.One,), value, ratio),)
            return AsymptoticRelationResult(
                relation,
                left,
                right,
                vars_tuple,
                (point,),
                value,
                value is not None,
                evidence,
                "direct univariate germ",
            )
        rays = (-sp.S.One, sp.S.One) if directions == "real" else (-1, 1, -sp.I, sp.I)
        evidence: list[DirectedRelationEvidence] = []
        all_true = True
        for ray in rays:
            local = sp.Dummy("_relation_t", positive=True)
            ff = left.subs(variable, point + ray * local)
            gg = right.subs(variable, point + ray * local)
            value, ratio = _univariate_relation(
                ff, gg, local, 0, relation, direction="+", assumptions=assumptions
            )
            evidence.append(DirectedRelationEvidence((sp.sympify(ray),), value, ratio))
            if value is False:
                return AsymptoticRelationResult(
                    relation,
                    left,
                    right,
                    vars_tuple,
                    (point,),
                    False,
                    True,
                    tuple(evidence),
                    "relation fails on a directed germ",
                )
            if value is not True:
                all_true = False
        certified = directions == "real" and all_true
        return AsymptoticRelationResult(
            relation,
            left,
            right,
            vars_tuple,
            (point,),
            True if certified else None,
            certified,
            tuple(evidence),
            "both real one-sided germs agree" if certified else "directed tests are inconclusive",
        )

    if any(sp.sympify(point) in (sp.oo, -sp.oo) for point in pts_tuple):
        return AsymptoticRelationResult(
            relation,
            left,
            right,
            vars_tuple,
            tuple(map(sp.sympify, pts_tuple)),
            None,
            False,
            (),
            "multivariate infinite-point ray localization is uncertified",
        )
    ray_vectors = (
        _real_directions(len(vars_tuple), ray_samples)
        if directions == "real"
        else _complex_directions(len(vars_tuple), ray_samples)
    )
    local = sp.Dummy("_relation_t", positive=True)
    evidence = []
    substitutions_base = tuple(map(sp.sympify, pts_tuple))
    for vector in ray_vectors:
        substitutions = {
            variable: point + local * component
            for variable, point, component in zip(vars_tuple, substitutions_base, vector)
        }
        ff = left.xreplace(substitutions)
        gg = right.xreplace(substitutions)
        if ff.has(sp.nan, sp.zoo) or gg.has(sp.nan, sp.zoo):
            continue
        value, ratio = _univariate_relation(
            ff, gg, local, 0, relation, direction="+", assumptions=assumptions
        )
        evidence.append(DirectedRelationEvidence(vector, value, ratio))
        if value is False:
            return AsymptoticRelationResult(
                relation,
                left,
                right,
                vars_tuple,
                substitutions_base,
                False,
                True,
                tuple(evidence),
                "relation fails on a deterministic directed ray",
            )
    return AsymptoticRelationResult(
        relation,
        left,
        right,
        vars_tuple,
        substitutions_base,
        None,
        False,
        tuple(evidence),
        "finite ray agreement cannot certify a multivariate relation",
    )


def asymptotic_equivalent(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether two expressions are asymptotically equivalent at a germ."""

    return asymptotic_relation(left, right, variable, point, relation="equivalent", **kwargs).value


def asymptotic_little_o(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` is little-o of ``right`` at the requested germ."""

    return asymptotic_relation(left, right, variable, point, relation="little-o", **kwargs).value


def asymptotic_big_o(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` is big-O of ``right`` at the requested germ."""

    return asymptotic_relation(left, right, variable, point, relation="big-o", **kwargs).value


def asymptotic_same_order(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether two expressions have the same asymptotic growth order."""

    return asymptotic_relation(left, right, variable, point, relation="same-order", **kwargs).value


def asymptotic_equal(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` and ``right`` are asymptotically Theta-equivalent.

    This is the coarse, two-sided order equivalence: each expression must be
    asymptotically bounded by a constant multiple of the other.  It corresponds
    as an order-equivalence predicate and is weaker than ratio-1
    :func:`asymptotic_equivalent`.
    """

    return asymptotic_relation(left, right, variable, point, relation="equal", **kwargs).value


def asymptotic_less(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` grows strictly slower than ``right`` (little-o)."""

    return asymptotic_relation(left, right, variable, point, relation="less", **kwargs).value


def asymptotic_less_equal(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` is asymptotically bounded above by ``right`` (big-O)."""

    return asymptotic_relation(left, right, variable, point, relation="less-equal", **kwargs).value


def asymptotic_greater(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` grows strictly faster than ``right`` (little-omega)."""

    return asymptotic_relation(left, right, variable, point, relation="greater", **kwargs).value


def asymptotic_greater_equal(left, right, variable, point=sp.oo, **kwargs) -> bool | None:
    """Return whether ``left`` is asymptotically bounded below by ``right`` (big-Omega)."""

    return asymptotic_relation(
        left, right, variable, point, relation="greater-equal", **kwargs
    ).value
