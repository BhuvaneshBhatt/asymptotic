"""Certified selection of asymptotic regimes before theorem dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._symbolic_policy import bounded_limit


class AsymptoticRegime(str, Enum):
    FIXED_PARAMETER_TAIL = "fixed-parameter-large-argument"
    SCALED_ORDER_TAIL = "large-order-scaled-argument"
    TURNING_POINT = "turning-point"
    COALESCING_SADDLES = "coalescing-saddles"
    STOKES_BOUNDARY = "stokes-boundary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegimeCertificate:
    regime: AsymptoticRegime
    large_parameter: sp.Symbol
    scaled_variable: sp.Expr | None
    transition_parameter: sp.Expr | None
    conditions: tuple[sp.Expr, ...]
    theorem: str | None
    verified: bool
    reason: str
    limiting_scaled_variable: sp.Expr | None = None
    transition_limit: sp.Expr | None = None


@dataclass(frozen=True)
class RegimeGeometry:
    """Geometry supplied by certified saddle or Stokes analyzers."""

    coalescing_saddles: tuple[sp.Expr, ...] = ()
    saddle_separation_scale: sp.Expr | None = None
    stokes_singulants: tuple[sp.Expr, ...] = ()
    stokes_distance_scale: sp.Expr | None = None
    hypotheses_verified: bool = False


_BESSEL = (sp.besselj, sp.bessely, sp.hankel1, sp.hankel2, sp.besseli, sp.besselk)


def _independent(expr: sp.Expr, variable: sp.Symbol) -> bool:
    return variable not in sp.sympify(expr).free_symbols


def _proved_limit(expr: sp.Expr, variable: sp.Symbol) -> sp.Expr | None:
    """Return a finite symbolic limit only when SymPy actually proves one."""
    try:
        value = bounded_limit(expr, variable, sp.oo)
    except (TypeError, ValueError, NotImplementedError, RecursionError, sp.PoleError):
        return None
    if value is None or isinstance(value, sp.Limit) or value.has(sp.nan, sp.zoo):
        return None
    if value in (sp.oo, -sp.oo) or value.is_finite is not True:
        return None
    return value


def select_asymptotic_regime(
    expr: sp.Expr,
    variable: sp.Symbol,
    point: sp.Expr = sp.oo,
    *,
    geometry: RegimeGeometry | None = None,
) -> RegimeCertificate:
    """Select an asymptotic theorem only after proving its scaling hypotheses."""
    variable = sp.sympify(variable)
    if geometry is None and isinstance(expr, sp.Integral):
        from .saddle_geometry import analyze_exponential_integral

        saddle = analyze_exponential_integral(expr, variable)
        if saddle is not None and saddle.hypotheses_verified:
            geometry = saddle.geometry
    expr = sp.sympify(expr)
    if geometry is not None and geometry.hypotheses_verified:
        if geometry.coalescing_saddles and geometry.saddle_separation_scale is not None:
            return RegimeCertificate(
                AsymptoticRegime.COALESCING_SADDLES,
                variable,
                None,
                geometry.saddle_separation_scale,
                (),
                "Chester-Friedman-Ursell Airy reduction",
                True,
                "saddle analyzer certified coalescence scaling",
            )
        if geometry.stokes_singulants and geometry.stokes_distance_scale is not None:
            return RegimeCertificate(
                AsymptoticRegime.STOKES_BOUNDARY,
                variable,
                None,
                geometry.stokes_distance_scale,
                (),
                "terminant Stokes expansion",
                True,
                "Stokes analyzer certified boundary scaling",
            )
    if point != sp.oo:
        return RegimeCertificate(
            AsymptoticRegime.UNKNOWN,
            variable,
            None,
            None,
            (),
            None,
            False,
            "uniform large-parameter dispatch requires +infinity",
        )
    if expr.func in _BESSEL:
        order, argument = expr.args
        order_scale = _proved_limit(sp.cancel(order / variable), variable)
        if order == variable or order_scale == 1:
            transition = sp.cancel((argument - order) / variable ** sp.Rational(1, 3))
            transition_limit = _proved_limit(transition, variable)
            if transition_limit is not None:
                scaled = sp.cancel(argument / variable)
                scaled_limit = _proved_limit(scaled, variable)
                return RegimeCertificate(
                    AsymptoticRegime.TURNING_POINT,
                    variable,
                    scaled,
                    transition,
                    (sp.Q.positive(variable), sp.Q.finite(transition_limit)),
                    "Bessel transition expansion",
                    True,
                    "order/parameter tends to one and the transition variable "
                    "has a proved finite limit",
                    scaled_limit,
                    transition_limit,
                )
            scaled = sp.cancel(argument / variable)
            scaled_limit = _proved_limit(scaled, variable)
            if scaled_limit is not None and scaled_limit != 0:
                return RegimeCertificate(
                    AsymptoticRegime.SCALED_ORDER_TAIL,
                    variable,
                    scaled,
                    None,
                    (sp.Q.positive(variable), sp.Q.nonzero(scaled_limit)),
                    "Bessel Airy-uniform large-order expansion",
                    True,
                    "order/parameter tends to one and argument/parameter has a "
                    "proved finite nonzero limit",
                    scaled_limit,
                    None,
                )
            return RegimeCertificate(
                AsymptoticRegime.UNKNOWN,
                variable,
                None,
                None,
                (),
                None,
                False,
                "neither transition nor fixed-scaled-argument behavior is proved",
            )
        if _independent(order, variable) and variable in argument.free_symbols:
            return RegimeCertificate(
                AsymptoticRegime.FIXED_PARAMETER_TAIL,
                variable,
                None,
                None,
                (sp.Q.finite(order),),
                "fixed-order large-argument expansion",
                True,
                "order is independent of the large argument",
            )
    return RegimeCertificate(
        AsymptoticRegime.UNKNOWN,
        variable,
        None,
        None,
        (),
        None,
        False,
        "no certified asymptotic scaling recognized",
    )
