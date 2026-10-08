"""Exact projective cluster-set decomposition for bivariate homogeneous quotients."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.calculus.util import function_range

from ._symbolic_policy import bounded_limit


@dataclass(frozen=True)
class ProjectiveClusterDecomposition:
    """Complete extended-real image of a rational function on RP^1."""

    angular_function: sp.Expr
    projective_variable: sp.Symbol
    components: tuple[sp.Set, ...]
    cluster_set: sp.Set
    certified: bool
    provider: str
    statement: str


def _homogeneous_degree(expr, variables):
    try:
        poly = sp.Poly(sp.expand(expr), *variables)
    except sp.PolynomialError:
        return None
    degrees = {sum(mon) for mon, coeff in poly.terms() if coeff != 0}
    return next(iter(degrees)) if len(degrees) == 1 else None


def _extended_closure(value_set):
    """Close real image components in the extended-real cluster topology."""
    if isinstance(value_set, sp.Union):
        return sp.Union(*(_extended_closure(arg) for arg in value_set.args))
    if isinstance(value_set, sp.Interval):
        return sp.Interval(value_set.start, value_set.end)
    if isinstance(value_set, sp.FiniteSet):
        return value_set
    return value_set


def _components(value_set):
    if isinstance(value_set, sp.Union):
        return tuple(value_set.args)
    return (value_set,)


def rational_projective_cluster_set(numerator, denominator, variables):
    """Compute the complete extended image on the real projective line.

    For homogeneous ``P/Q`` this uses the affine chart ``[1:t]`` and restores
    the unit-sphere normalization.  The missing projective point ``[0:1]`` is
    included explicitly.  The closure of each real range component supplies
    finite boundary cluster values and directed infinities at denominator
    divisors.

    Parameter-dependent coefficient topology is delegated to the
    parameter-cell layer; this routine certifies only parameter-free geometry.
    """
    variables = tuple(variables)
    numerator = sp.sympify(numerator)
    denominator = sp.sympify(denominator)
    if len(variables) != 2:
        return None
    x, y = variables
    dp = _homogeneous_degree(numerator, variables)
    dq = _homogeneous_degree(denominator, variables)
    if dp is None or dq is None:
        return None
    parameters = (numerator.free_symbols | denominator.free_symbols) - set(variables)
    if parameters:
        return None
    degree_delta = dq - dp
    if degree_delta % 2:
        return None

    from .blowup_geometry import projective_chart

    chart = projective_chart(variables, (0, 0), 0)
    t = chart.angular_variables[1]
    p_chart = sp.expand(numerator.subs({x: 1, y: t}))
    q_chart = sp.expand(denominator.subs({x: 1, y: t}))
    angular = sp.cancel(p_chart * (1 + t**2) ** (degree_delta // 2) / q_chart)
    if not angular.is_rational_function(t):
        return None
    try:
        image = function_range(angular, t, sp.S.Reals)
    except (NotImplementedError, ValueError, TypeError, sp.PoleError):
        return None

    # Add the omitted projective point [0:1] whenever its spherical value is
    # finite.  If it is a pole, the adjacent affine chart already contributes
    # the corresponding unbounded cluster component through closure.
    other_chart = projective_chart(variables, (0, 0), 1)
    s = other_chart.angular_variables[0]
    other = sp.cancel(
        sp.expand(numerator.subs({x: s, y: 1}))
        * (1 + s**2) ** (degree_delta // 2)
        / sp.expand(denominator.subs({x: s, y: 1}))
    )
    try:
        infinity_value = bounded_limit(other, s, 0, allow_general=True)
    except (NotImplementedError, ValueError, TypeError, sp.PoleError):
        infinity_value = None
    if (
        infinity_value is not None
        and infinity_value not in (sp.zoo, sp.nan)
        and infinity_value.is_finite is True
    ):
        image = sp.Union(image, sp.FiniteSet(infinity_value))

    cluster_set = _extended_closure(image)
    return ProjectiveClusterDecomposition(
        angular,
        t,
        _components(cluster_set),
        cluster_set,
        True,
        "rational_projective_range_decomposition",
        "complete extended-real image from exact RP^1 rational range decomposition",
    )


@dataclass(frozen=True)
class ParameterProjectiveClusterStratum:
    """A complete projective cluster set on one exact parameter cell."""

    condition: sp.Expr
    decomposition: ProjectiveClusterDecomposition


def _linear_fractional_parameter_strata(numerator, denominator, variables, condition):
    """Exact RP^1 topology for homogeneous linear fractional forms."""
    if len(variables) != 2:
        return ()
    x, y = variables
    if (
        _homogeneous_degree(numerator, variables) != 1
        or _homogeneous_degree(denominator, variables) != 1
    ):
        return ()
    a = sp.expand(numerator).coeff(x)
    b = sp.expand(numerator).coeff(y)
    c = sp.expand(denominator).coeff(x)
    d = sp.expand(denominator).coeff(y)
    if (
        sp.expand(numerator - a * x - b * y) != 0
        or sp.expand(denominator - c * x - d * y) != 0
    ):
        return ()
    determinant = sp.factor(a * d - b * c)
    parameters = tuple(
        sorted(
            (numerator.free_symbols | denominator.free_symbols) - set(variables),
            key=sp.default_sort_key,
        )
    )
    if not parameters:
        return ()
    from .parameter_cells import parameter_truth_cells

    cells = parameter_truth_cells((sp.Eq(determinant, 0),), parameters, condition)
    out = []
    for cell in cells:
        degenerate = cell.signature[0]
        if degenerate:
            # Proportional forms give one constant value where denominator != 0.
            candidates = []
            if c != 0:
                candidates.append(sp.cancel(a / c))
            if d != 0:
                candidates.append(sp.cancel(b / d))
            if not candidates:
                continue
            value = candidates[0]
            cluster_set = sp.FiniteSet(value)
            angular = value
            provider = "parameter_projective_degenerate_mobius"
        else:
            # Every nonconstant real Möbius map of RP^1 is onto RP^1.  In the
            # extended-real cluster convention its finite cluster values are R.
            cluster_set = sp.S.Reals
            angular = sp.cancel(
                (a + b * sp.Symbol("_projective_t", real=True))
                / (c + d * sp.Symbol("_projective_t", real=True))
            )
            provider = "parameter_projective_mobius"
        out.append(
            ParameterProjectiveClusterStratum(
                cell.condition,
                ProjectiveClusterDecomposition(
                    angular,
                    sp.Symbol("_projective_t", real=True),
                    (cluster_set,),
                    cluster_set,
                    True,
                    provider,
                    "complete RP^1 image on an exact determinant parameter cell",
                ),
            )
        )
    return tuple(out)


def parameterized_projective_cluster_strata(
    numerator, denominator, variables, condition=sp.S.true
):
    """Refine symbolic projective geometry by exact algebraic topology cells.

    The first exact family is the full homogeneous Möbius family.  Its
    determinant is the pole/critical topology discriminant: nonzero cells are
    projective automorphisms, while the zero cell collapses to a singleton.
    Parameter-free residual geometry continues through the general rational
    range decomposition.
    """
    numerator = sp.sympify(numerator)
    denominator = sp.sympify(denominator)
    variables = tuple(variables)
    strata = _linear_fractional_parameter_strata(
        numerator, denominator, variables, sp.sympify(condition)
    )
    if strata:
        return strata
    if not ((numerator.free_symbols | denominator.free_symbols) - set(variables)):
        result = rational_projective_cluster_set(numerator, denominator, variables)
        if result is not None:
            return (ParameterProjectiveClusterStratum(sp.sympify(condition), result),)
    return ()


__all__ = [
    "ParameterProjectiveClusterStratum",
    "ProjectiveClusterDecomposition",
    "parameterized_projective_cluster_strata",
    "rational_projective_cluster_set",
]
