"""Certified maps of complete cluster sets through discontinuous outer functions."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_policy import bounded_ask
from .coverage import CoverageCertificate
from .multivariate_limits_advanced import AdvancedLimitStatus


@dataclass(frozen=True)
class ClusterMapResult:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    target: tuple[sp.Expr, ...]
    cluster_set: sp.Set | None
    status: AdvancedLimitStatus
    provider: str
    statement: str
    inner_cluster_set: sp.Set | None = None
    coverage: CoverageCertificate | None = None

    @property
    def certified(self):
        return (
            self.status is AdvancedLimitStatus.CERTIFIED
            and self.coverage is not None
            and self.coverage.certified
        )

    @property
    def limit_semantics(self):
        from .cluster_semantics import cluster_limit_semantics

        return cluster_limit_semantics(self)


def _integer_points(interval):
    if not isinstance(interval, sp.Interval):
        return None
    if interval.start in (sp.oo, -sp.oo) or interval.end in (sp.oo, -sp.oo):
        return None
    lo = sp.ceiling(interval.start)
    hi = sp.floor(interval.end)
    if not (lo.is_integer is True and hi.is_integer is True):
        return None
    if int(hi - lo) > 256:
        return None
    return tuple(sp.Integer(k) for k in range(int(lo), int(hi) + 1))


def _map_interval(func, interval):
    a, b = interval.start, interval.end
    if func is sp.Heaviside:
        values = []
        if a.is_negative is True or a is -sp.oo:
            values.append(sp.S.Zero)
        if interval.contains(0) is not sp.S.false:
            values.append(sp.Rational(1, 2))
        if b.is_positive is True or b is sp.oo:
            values.append(sp.S.One)
        return sp.FiniteSet(*values) if values else None
    if func is sp.sign:
        values = []
        if a.is_negative is True or a is -sp.oo:
            values.append(sp.S.NegativeOne)
        if interval.contains(0) is not sp.S.false:
            values.append(sp.S.Zero)
        if b.is_positive is True or b is sp.oo:
            values.append(sp.S.One)
        return sp.FiniteSet(*values) if values else None
    if func is sp.floor:
        if a in (-sp.oo, sp.oo) or b in (-sp.oo, sp.oo):
            return sp.S.Integers
        lo, hi = sp.floor(a), sp.floor(b)
        if lo.is_integer and hi.is_integer:
            return sp.Range(lo, hi + 1)
    if func is sp.ceiling:
        if a in (-sp.oo, sp.oo) or b in (-sp.oo, sp.oo):
            return sp.S.Integers
        lo, hi = sp.ceiling(a), sp.ceiling(b)
        if lo.is_integer and hi.is_integer:
            return sp.Range(lo, hi + 1)
    if func is sp.frac:
        if a in (-sp.oo, sp.oo) or b in (-sp.oo, sp.oo):
            return sp.Interval(0, 1)
        integers = _integer_points(interval)
        if integers is None:
            return None
        if not integers:
            return sp.Interval(sp.frac(a), sp.frac(b))
        # Crossing an integer gives values arbitrarily close to 1 and the
        # integer itself gives 0, so the cluster closure is [0,1].
        return sp.Interval(0, 1)
    return None


def map_cluster_set(outer, cluster_set):
    """Map a certified real cluster set through a supported discontinuous outer."""
    if not isinstance(cluster_set, sp.Set):
        return None
    if isinstance(cluster_set, sp.Union):
        pieces = [map_cluster_set(outer, part) for part in cluster_set.args]
        if any(piece is None for piece in pieces):
            return None
        return sp.Union(*pieces)
    if isinstance(cluster_set, sp.FiniteSet):
        return sp.FiniteSet(*(sp.simplify(outer(v)) for v in cluster_set))
    if isinstance(cluster_set, sp.Interval):
        return _map_interval(outer, cluster_set)
    return None


def discontinuous_outer_cluster_set(
    expression, variables, target, *, inner_cluster_result=None
):
    """Push a complete inner cluster set through a discontinuous outer function.

    This consumes only a certified complete inner cluster result.
    It never infers existence from sampled paths.
    """
    expression = sp.sympify(expression)
    variables = (
        tuple(variables) if isinstance(variables, (tuple, list)) else (variables,)
    )
    target = tuple(target) if isinstance(target, (tuple, list)) else (target,)
    if len(expression.args) != 1 or expression.func not in (
        sp.floor,
        sp.ceiling,
        sp.sign,
        sp.frac,
        sp.Heaviside,
    ):
        return ClusterMapResult(
            expression,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "cluster_map",
            "outer cluster map not recognized",
            coverage=CoverageCertificate.unknown(
                "cluster_map_unknown", "outer function is unsupported", ("outer_map",)
            ),
        )
    inner = expression.args[0]
    if inner_cluster_result is None:
        # Avoid recursive frontend calls: use the common cluster engines directly.
        from .multivariate_limits_advanced import newton_fan_cluster_set

        inner_cluster_result = newton_fan_cluster_set(inner, variables, target)
    complete = bool(getattr(inner_cluster_result, "certified", False))
    inner_set = getattr(inner_cluster_result, "cluster_set", None)
    if not complete:
        return ClusterMapResult(
            expression,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "cluster_map",
            "cannot map an incomplete inner cluster set",
            inner_set,
            coverage=CoverageCertificate.partial(
                "inner_cluster_incomplete",
                "inner cluster set is not certified complete",
                (),
                ("inner_cluster",),
            ),
        )
    mapped = map_cluster_set(expression.func, inner_set)
    if mapped is None:
        return ClusterMapResult(
            expression,
            variables,
            target,
            None,
            AdvancedLimitStatus.UNKNOWN,
            "cluster_map",
            "outer cluster image not certified",
            inner_set,
            coverage=CoverageCertificate.unknown(
                "cluster_map_range_unknown",
                "outer image could not be certified",
                ("outer_image",),
            ),
        )
    coverage = CoverageCertificate.complete(
        "complete_cluster_map",
        "complete inner cluster set mapped with all discontinuity boundary values included",
        (inner_set,),
    )
    return ClusterMapResult(
        expression,
        variables,
        target,
        mapped,
        AdvancedLimitStatus.CERTIFIED,
        "discontinuous_outer_cluster_map",
        "complete cluster image under a supported discontinuous outer function",
        inner_set,
        coverage=coverage,
    )


def mod_cluster_set(inner_cluster_set, modulus):
    """Exact cluster map for positive constant modulus on finite sets/intervals."""
    modulus = sp.sympify(modulus)
    if modulus.is_positive is not True:
        return None
    if isinstance(inner_cluster_set, sp.FiniteSet):
        return sp.FiniteSet(*(sp.Mod(v, modulus) for v in inner_cluster_set))
    if isinstance(inner_cluster_set, sp.Interval):
        a, b = inner_cluster_set.start, inner_cluster_set.end
        if a in (-sp.oo, sp.oo) or b in (-sp.oo, sp.oo):
            return sp.Interval(0, modulus)
        width = sp.simplify(b - a)
        if bounded_ask(sp.Q.nonnegative(width - modulus)) is True:
            return sp.Interval(0, modulus)
        qa, qb = sp.floor(a / modulus), sp.floor(b / modulus)
        if sp.simplify(qa - qb) == 0:
            return sp.Interval(sp.Mod(a, modulus), sp.Mod(b, modulus))
        return sp.Interval(0, modulus)
    return None


def piecewise_domain_cluster_set(
    expression, variables, target, *, domain=sp.S.true, evaluator=None
):
    """Evaluate Piecewise branches on their accumulating semialgebraic domains."""
    expression = sp.sympify(expression)
    if not isinstance(expression, sp.Piecewise):
        return None
    from .domain_cluster_geometry import domain_relative_cluster_set

    component_results = []
    sets = []
    for branch, condition in expression.args:
        branch_domain = sp.And(domain, condition)
        result = domain_relative_cluster_set(
            branch, variables, target, domain=branch_domain, evaluator=evaluator
        )
        if not result.certified:
            return None
        component_results.append(result)
        sets.append(result.cluster_set)
    return sp.Union(*sets) if sets else sp.S.EmptySet


__all__ = [
    "ClusterMapResult",
    "discontinuous_outer_cluster_set",
    "map_cluster_set",
    "mod_cluster_set",
    "piecewise_domain_cluster_set",
]
