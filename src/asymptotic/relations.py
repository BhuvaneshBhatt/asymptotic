"""Asymptotic relations with conservative directed-ray reduction.

Univariate decisions are made by the shared :class:`AsymptoticContext` growth
and limit oracles.  At finite multivariate points, deterministic rays are used
only as a falsification device: one failing ray proves the proposed relation
false, while agreement on finitely many rays is reported as
undecided rather than as a proof of a multivariate limit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import sympy as sp
from funcprops import ConditionalExpression, normalize_assumptions

from ._symbolic_policy import bounded_ask
from .context import AsymptoticContext, AsymptoticGrowthComparison

_RELATION_LOCAL = sp.Dummy("_relation_t", positive=True)

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
class RelationStratum:
    """One parameter condition under which a relation has a definite value."""

    condition: sp.Expr
    value: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", normalize_assumptions(self.condition))


@dataclass(frozen=True)
class DirectedRelationEvidence:
    direction: tuple[sp.Expr, ...]
    value: bool | None
    ratio_limit: sp.Expr | None = None


@dataclass(frozen=True)
class UnivariateRelationFacts:
    """Shared facts used by every relation predicate for one directed germ."""

    left: sp.Expr
    right: sp.Expr
    left_zero: bool | None
    right_zero: bool | None
    comparison: AsymptoticGrowthComparison | None
    ratio_limit: sp.Expr | None


@dataclass(frozen=True)
class RelationResult:
    relation: RelationKind
    left: sp.Expr
    right: sp.Expr
    variables: tuple[sp.Symbol, ...]
    points: tuple[sp.Expr, ...]
    value: bool | ConditionalExpression | None
    certified: bool
    strata: tuple[RelationStratum, ...] = ()
    evidence: tuple[DirectedRelationEvidence, ...] = ()
    reason: str = ""


def _reciprocal(expr: sp.Expr) -> sp.Expr:
    return sp.Pow(sp.sympify(expr), -1, evaluate=False)


def _direct_ratio_condition(
    ratio: sp.Expr | None, relation: RelationKind
) -> sp.Expr | None:
    """Return a symbolic sufficient condition when a finite ratio is available."""
    if ratio is None or isinstance(ratio, (sp.Limit, sp.AccumBounds)):
        return None
    ratio = sp.sympify(ratio)
    if not ratio.free_symbols:
        return None
    if relation in {"little-o", "less"}:
        return sp.Eq(ratio, 0, evaluate=False)
    if relation in {"big-o", "less-equal"}:
        return sp.Ne(_reciprocal(ratio), 0, evaluate=False)
    if relation == "greater":
        return sp.Eq(_reciprocal(ratio), 0, evaluate=False)
    if relation == "greater-equal":
        return sp.Ne(ratio, 0, evaluate=False)
    if relation in {"same-order", "equal"}:
        return sp.Ne(ratio, 0, evaluate=False)
    if relation == "equivalent":
        return sp.Eq(ratio, 1, evaluate=False)
    return None


def _log_vs_symbolic_power_condition(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    relation: RelationKind,
) -> ConditionalExpression | None:
    """Recognize the classical log-versus-power hierarchy under a sign condition."""
    if (
        left != sp.log(variable)
        or not isinstance(right, sp.Pow)
        or right.base != variable
    ):
        return None
    exponent = right.exp
    if not exponent.free_symbols or variable in exponent.free_symbols:
        return None
    positive = sp.StrictGreaterThan(exponent, 0, evaluate=False)
    if point == sp.oo:
        value = relation in {"little-o", "less", "big-o", "less-equal"}
        return ConditionalExpression(value, positive)
    if point == 0:
        if relation == "greater-equal":
            return ConditionalExpression(
                True, sp.StrictGreaterThan(sp.re(exponent), 0, evaluate=False)
            )
        value = relation in {"greater", "greater-equal"}
        return ConditionalExpression(value, positive)
    return None


def _shifted_power_relation(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    relation: RelationKind,
    assumptions: sp.Expr,
) -> bool | ConditionalExpression | None:
    """Recognize ``(a+x)^b`` versus ``x^b`` under positive parameters."""
    if not isinstance(left, sp.Pow) or not isinstance(right, sp.Pow):
        return None
    if right.base != variable or left.exp != right.exp:
        return None
    exponent = right.exp
    shift = sp.expand(left.base - variable)
    if variable in shift.free_symbols or not shift.free_symbols:
        return None
    from funcprops import entails

    positive = sp.And(
        sp.StrictGreaterThan(shift, 0, evaluate=False),
        sp.StrictGreaterThan(exponent, 0, evaluate=False),
        evaluate=False,
    )
    if entails(positive, assumptions) is not True:
        return None
    if point == sp.oo:
        if relation == "equivalent":
            return True
        return relation in {
            "big-o",
            "less-equal",
            "greater-equal",
            "same-order",
            "equal",
        }
    if point == 0:
        return relation in {"greater", "greater-equal"}
    if point == 1:
        if relation == "equivalent":
            return ConditionalExpression(
                True,
                sp.Eq(sp.Pow(1 + shift, exponent), 1, evaluate=False),
            )
        return relation in {
            "big-o",
            "less-equal",
            "greater-equal",
            "same-order",
            "equal",
        }
    return None


def _candidate_parameter_conditions(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    assumptions: sp.Expr,
) -> tuple[sp.Expr, ...]:
    params = tuple(
        sorted(
            (left.free_symbols | right.free_symbols) - {variable},
            key=sp.default_sort_key,
        )
    )
    out: list[sp.Expr] = []
    for symbol in params:
        # Sign strata are particularly important for symbolic powers and
        # exponential rates. They are sufficient conditions, not a claim that
        # every parameter is real by default.
        out.extend(
            (
                sp.StrictGreaterThan(symbol, 0, evaluate=False),
                sp.StrictLessThan(symbol, 0, evaluate=False),
                sp.Eq(symbol, 0, evaluate=False),
            )
        )
    unique = []
    seen = set()
    for condition in out:
        normalized = normalize_assumptions(condition)
        key = sp.srepr(normalized)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return tuple(unique)


def _conditional_value(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    relation: RelationKind,
    ratio: sp.Expr | None,
    *,
    direction: str,
    assumptions: sp.Expr,
) -> tuple[ConditionalExpression | None, tuple[RelationStratum, ...]]:
    direct = _direct_ratio_condition(ratio, relation)
    if direct is not None:
        # If assumptions already entail the sufficient condition the public
        # result should be unconditional.
        from funcprops import entails

        verdict = entails(direct, assumptions)
        if verdict is True:
            return ConditionalExpression(True, sp.S.true), (
                RelationStratum(sp.S.true, True),
            )
        if verdict is False:
            return ConditionalExpression(False, sp.S.true), (
                RelationStratum(sp.S.true, False),
            )
        return ConditionalExpression(True, direct), (RelationStratum(direct, True),)

    strata: list[RelationStratum] = []
    for condition in _candidate_parameter_conditions(
        left, right, variable, assumptions
    ):
        combined = normalize_assumptions(sp.And(assumptions, condition, evaluate=False))
        value, _ = _univariate_relation(
            left,
            right,
            variable,
            point,
            relation,
            direction=direction,
            assumptions=combined,
        )
        if value is not None:
            strata.append(RelationStratum(condition, value))
    if not strata:
        return None, ()
    # Prefer the simplest sufficient stratum. This mirrors a ConditionalExpression
    # rather than claiming the finite set of probed strata is exhaustive.
    strata.sort(
        key=lambda item: (
            sp.count_ops(item.condition),
            sp.default_sort_key(item.condition),
        )
    )
    chosen = strata[0]
    return ConditionalExpression(chosen.value, chosen.condition), tuple(strata)


@lru_cache(maxsize=4096)
def _univariate_relation_facts(
    left: sp.Expr,
    right: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr,
    direction: str,
    assumptions: sp.Expr,
) -> UnivariateRelationFacts:
    """Compute the expensive pair facts once for all relation predicates."""
    left = sp.refine(sp.sympify(left), assumptions)
    right = sp.refine(sp.sympify(right), assumptions)
    ctx = AsymptoticContext(variable, point=point, direction=direction)
    left_zero = ctx.is_zero(left)
    right_zero = ctx.is_zero(right)
    if right_zero is True:
        return UnivariateRelationFacts(
            left,
            right,
            left_zero,
            right_zero,
            None,
            sp.S.One if left_zero is True else None,
        )
    comparison, comparison_ratio = ctx.compare_growth(left, right)
    ratio_limit = (
        comparison_ratio if comparison_ratio is not None else ctx.limit(left / right)
    )
    return UnivariateRelationFacts(
        left, right, left_zero, right_zero, comparison, ratio_limit
    )


@lru_cache(maxsize=4096)
def _ratio_properties(
    ratio: sp.Expr | None,
    assumptions: sp.Expr,
) -> tuple[bool | None, bool | None, bool | None]:
    """Return zero, finite, and nonzero facts for a previously computed ratio."""
    if ratio is None or isinstance(ratio, sp.Limit):
        return None, None, None
    if isinstance(ratio, sp.AccumBounds):
        lower = ratio.min
        upper = ratio.max
        finite = (
            bounded_ask(sp.Q.finite(lower), assumptions) is True
            and bounded_ask(sp.Q.finite(upper), assumptions) is True
        )
        zero = lower == 0 and upper == 0
        nonzero = (
            bounded_ask(sp.Q.positive(lower), assumptions) is True
            or bounded_ask(sp.Q.negative(upper), assumptions) is True
        )
        return zero, finite, nonzero
    return (
        bounded_ask(sp.Q.zero(ratio), assumptions),
        bounded_ask(sp.Q.finite(ratio), assumptions),
        bounded_ask(sp.Q.nonzero(ratio), assumptions),
    )


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
    """Decide one directed relation from a shared pair-fact bundle."""
    assumptions = normalize_assumptions(assumptions)
    facts = _univariate_relation_facts(
        sp.sympify(left),
        sp.sympify(right),
        variable,
        sp.sympify(point),
        direction,
        assumptions,
    )
    ratio = facts.ratio_limit
    if facts.right_zero is True:
        if relation == "equivalent":
            return (True, sp.S.One) if facts.left_zero is True else (False, None)
        return None, None

    if relation == "equivalent":
        if isinstance(ratio, sp.AccumBounds):
            if ratio.min == 1 and ratio.max == 1:
                return True, ratio
            return False, ratio
        equality = (
            sp.refine(sp.Eq(ratio, 1), assumptions)
            if ratio is not None and not isinstance(ratio, sp.Limit)
            else None
        )
        if ratio == 1 or equality is sp.S.true:
            return True, ratio
        if equality is sp.S.false:
            return False, ratio
        return None, ratio

    comparison = facts.comparison
    zero, finite, nonzero = _ratio_properties(ratio, assumptions)
    if isinstance(ratio, sp.AccumBounds):
        if relation in {"little-o", "less"}:
            return zero is True, ratio
        if relation in {"big-o", "less-equal"} and finite is True:
            return True, ratio
        if relation == "greater":
            return False, ratio
        if relation == "greater-equal" and nonzero is True:
            return True, ratio
        if relation in {"same-order", "equal"} and finite is True and nonzero is True:
            return True, ratio

    if relation in {"little-o", "less"} and zero is True:
        return True, ratio
    if relation in {"big-o", "less-equal"} and finite is True:
        return True, ratio
    if relation in {"same-order", "equal"} and finite is True and nonzero is True:
        return True, ratio

    if relation in {"little-o", "less"}:
        if comparison is AsymptoticGrowthComparison.SMALLER:
            return True, ratio
        if comparison in {
            AsymptoticGrowthComparison.SAME_ORDER,
            AsymptoticGrowthComparison.LARGER,
        }:
            return False, ratio
        return None, ratio
    if relation in {"big-o", "less-equal"}:
        if comparison in {
            AsymptoticGrowthComparison.SMALLER,
            AsymptoticGrowthComparison.SAME_ORDER,
        }:
            return True, ratio
        if comparison is AsymptoticGrowthComparison.LARGER:
            return False, ratio
        return None, ratio
    if relation == "greater":
        if comparison is AsymptoticGrowthComparison.LARGER:
            return True, ratio
        if comparison in {
            AsymptoticGrowthComparison.SAME_ORDER,
            AsymptoticGrowthComparison.SMALLER,
        }:
            return False, ratio
        return None, ratio
    if relation == "greater-equal":
        if comparison in {
            AsymptoticGrowthComparison.LARGER,
            AsymptoticGrowthComparison.SAME_ORDER,
        }:
            return True, ratio
        if comparison is AsymptoticGrowthComparison.SMALLER:
            return False, ratio
        return None, ratio
    if relation in {"same-order", "equal"}:
        if comparison is AsymptoticGrowthComparison.SAME_ORDER:
            return True, ratio
        if comparison in {
            AsymptoticGrowthComparison.SMALLER,
            AsymptoticGrowthComparison.LARGER,
        }:
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


def _complex_directions(
    dimension: int, samples: int
) -> tuple[tuple[sp.Expr, ...], ...]:
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


def relation(
    left: sp.Expr,
    right: sp.Expr,
    variables: sp.Symbol | tuple[sp.Symbol, ...] | list[sp.Symbol],
    points: sp.Expr | tuple[sp.Expr, ...] | list[sp.Expr],
    *,
    relation: RelationKind = "equivalent",
    directions: Literal["real", "complex"] = "real",
    ray_samples: int = 8,
    assumptions: sp.Expr | bool = sp.S.true,
) -> RelationResult:
    """Decide or conservatively test an asymptotic relation.

    For one variable at a finite real point, ``directions="real"`` checks both
    one-sided germs and certifies ``True`` only when both sides agree.  For
    several variables, coordinate and deterministic pseudo-random rays can
    *disprove* a relation, but finite ray sampling never certifies a positive
    multivariate statement.
    """

    left = sp.sympify(left)
    right = sp.sympify(right)
    assumptions = normalize_assumptions(assumptions)
    vars_tuple = (
        tuple(variables)
        if isinstance(variables, (tuple, list, sp.Tuple))
        else (variables,)
    )
    pts_tuple = (
        tuple(points) if isinstance(points, (tuple, list, sp.Tuple)) else (points,)
    )
    if len(vars_tuple) != len(pts_tuple):
        raise ValueError("variables and points must have the same length")
    if not vars_tuple or not all(
        isinstance(variable, sp.Symbol) for variable in vars_tuple
    ):
        raise TypeError("variables must be SymPy symbols")
    if ray_samples < 0:
        raise ValueError("ray_samples must be nonnegative")

    if len(vars_tuple) == 1:
        variable = vars_tuple[0]
        point = sp.sympify(pts_tuple[0])
        try:
            constant_ratio = sp.cancel(left / right)
        except (TypeError, ValueError, NotImplementedError):
            constant_ratio = left / right
        if variable not in constant_ratio.free_symbols and constant_ratio.free_symbols:
            condition = _direct_ratio_condition(constant_ratio, relation)
            if condition is not None:
                from funcprops import entails

                verdict = entails(condition, assumptions)
                if verdict is True:
                    public: bool | ConditionalExpression = True
                    strata: tuple[RelationStratum, ...] = ()
                elif verdict is False:
                    public = False
                    strata = ()
                else:
                    public = ConditionalExpression(True, condition)
                    strata = (RelationStratum(condition, True),)
                return RelationResult(
                    relation,
                    left,
                    right,
                    vars_tuple,
                    (point,),
                    public,
                    verdict is not None,
                    strata,
                    (),
                    "parameter-conditioned constant ratio",
                )
        shifted = _shifted_power_relation(
            left, right, variable, point, relation, assumptions
        )
        if shifted is not None:
            if isinstance(shifted, ConditionalExpression):
                return RelationResult(
                    relation,
                    left,
                    right,
                    vars_tuple,
                    (point,),
                    shifted,
                    False,
                    (RelationStratum(shifted.condition, bool(shifted.value)),),
                    (),
                    "parameter-conditioned shifted-power relation",
                )
            return RelationResult(
                relation,
                left,
                right,
                vars_tuple,
                (point,),
                bool(shifted),
                True,
                (),
                (),
                "shifted-power relation under supplied assumptions",
            )
        special = _log_vs_symbolic_power_condition(
            left, right, variable, point, relation
        )
        if special is not None and point != 1:
            from funcprops import entails

            if entails(special.condition, assumptions) is True:
                special_value = bool(special.value)
                return RelationResult(
                    relation,
                    left,
                    right,
                    vars_tuple,
                    (point,),
                    special_value,
                    True,
                    (),
                    (),
                    "classical log-versus-power hierarchy under supplied assumptions",
                )
            return RelationResult(
                relation,
                left,
                right,
                vars_tuple,
                (point,),
                special,
                False,
                (RelationStratum(special.condition, bool(special.value)),),
                (),
                "parameter-conditioned log-versus-power hierarchy",
            )

    if len(vars_tuple) == 1:
        variable = vars_tuple[0]
        point = sp.sympify(pts_tuple[0])
        if point in (sp.oo, -sp.oo):
            value, ratio = _univariate_relation(
                left, right, variable, point, relation, assumptions=assumptions
            )
            evidence = (DirectedRelationEvidence((sp.S.One,), value, ratio),)
            strata: tuple[RelationStratum, ...] = ()
            public_value: bool | ConditionalExpression | None = value
            if value is None:
                conditional = _log_vs_symbolic_power_condition(
                    left, right, variable, point, relation
                )
                if conditional is not None:
                    strata = (
                        RelationStratum(conditional.condition, bool(conditional.value)),
                    )
                    public_value = conditional
                else:
                    conditional, strata = _conditional_value(
                        left,
                        right,
                        variable,
                        point,
                        relation,
                        ratio,
                        direction="+",
                        assumptions=assumptions,
                    )
                    if conditional is not None:
                        public_value = conditional
            return RelationResult(
                relation,
                left,
                right,
                vars_tuple,
                (point,),
                public_value,
                value is not None,
                strata,
                evidence,
                "direct univariate germ"
                if value is not None
                else "parameter-conditioned univariate germ",
            )
        rays = (-sp.S.One, sp.S.One) if directions == "real" else (-1, 1, -sp.I, sp.I)
        evidence: list[DirectedRelationEvidence] = []
        all_true = True
        for ray in rays:
            local = _RELATION_LOCAL
            ff = left.subs(variable, point + ray * local)
            gg = right.subs(variable, point + ray * local)
            value, ratio = _univariate_relation(
                ff, gg, local, 0, relation, direction="+", assumptions=assumptions
            )
            evidence.append(DirectedRelationEvidence((sp.sympify(ray),), value, ratio))
            if value is False:
                return RelationResult(
                    relation,
                    left,
                    right,
                    vars_tuple,
                    (point,),
                    False,
                    True,
                    (),
                    tuple(evidence),
                    "relation fails on a directed germ",
                )
            if value is not True:
                all_true = False
        certified = directions == "real" and all_true
        return RelationResult(
            relation,
            left,
            right,
            vars_tuple,
            (point,),
            True if certified else None,
            certified,
            (),
            tuple(evidence),
            "both real one-sided germs agree"
            if certified
            else "directed tests are inconclusive",
        )

    if any(sp.sympify(point) in (sp.oo, -sp.oo) for point in pts_tuple):
        return RelationResult(
            relation,
            left,
            right,
            vars_tuple,
            tuple(map(sp.sympify, pts_tuple)),
            None,
            False,
            (),
            (),
            "multivariate infinite-point ray localization is uncertified",
        )
    ray_vectors = (
        _real_directions(len(vars_tuple), ray_samples)
        if directions == "real"
        else _complex_directions(len(vars_tuple), ray_samples)
    )
    local = _RELATION_LOCAL
    evidence = []
    substitutions_base = tuple(map(sp.sympify, pts_tuple))
    for vector in ray_vectors:
        substitutions = {
            variable: point + local * component
            for variable, point, component in zip(
                vars_tuple, substitutions_base, vector
            )
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
            return RelationResult(
                relation,
                left,
                right,
                vars_tuple,
                substitutions_base,
                False,
                True,
                (),
                tuple(evidence),
                "relation fails on a deterministic directed ray",
            )
    return RelationResult(
        relation,
        left,
        right,
        vars_tuple,
        substitutions_base,
        None,
        False,
        (),
        tuple(evidence),
        "finite ray agreement cannot certify a multivariate relation",
    )


def equivalent(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether two expressions are asymptotically equivalent at a germ."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="equivalent",
        assumptions=assumptions,
        **kwargs,
    ).value


def little_o(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` is little-o of ``right`` at the requested germ."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="little-o",
        assumptions=assumptions,
        **kwargs,
    ).value


def big_o(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` is big-O of ``right`` at the requested germ."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="big-o",
        assumptions=assumptions,
        **kwargs,
    ).value


def same_order(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether two expressions have the same asymptotic growth order."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="same-order",
        assumptions=assumptions,
        **kwargs,
    ).value


def equal(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` and ``right`` are asymptotically Theta-equivalent.

    This is the coarse, two-sided order equivalence: each expression must be
    asymptotically bounded by a constant multiple of the other.  It corresponds
    as an order-equivalence predicate and is weaker than ratio-1
    :func:`equivalent`.
    """

    return relation(
        left,
        right,
        variable,
        point,
        relation="equal",
        assumptions=assumptions,
        **kwargs,
    ).value


def less(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` grows strictly slower than ``right`` (little-o)."""

    return relation(
        left, right, variable, point, relation="less", assumptions=assumptions, **kwargs
    ).value


def less_equal(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` is asymptotically bounded above by ``right`` (big-O)."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="less-equal",
        assumptions=assumptions,
        **kwargs,
    ).value


def greater(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` grows strictly faster than ``right`` (little-omega)."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="greater",
        assumptions=assumptions,
        **kwargs,
    ).value


def greater_equal(
    left, right, variable, point=sp.oo, *, assumptions=True, **kwargs
) -> bool | ConditionalExpression | None:
    """Return whether ``left`` is asymptotically bounded below by ``right`` (big-Omega)."""

    return relation(
        left,
        right,
        variable,
        point,
        relation="greater-equal",
        assumptions=assumptions,
        **kwargs,
    ).value
