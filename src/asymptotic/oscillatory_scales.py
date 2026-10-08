"""Certified real oscillatory scales and valuation-driven phase geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from .blowup_geometry import Valuation
from .proof_obligations import ObligationKind, ProofObligation


class PhaseBehavior(Enum):
    VANISHING = "vanishing"
    FINITE = "finite"
    DIVERGENT = "divergent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OscillatoryScale:
    """An amplitude-phase transmonomial ``A*exp(I*phi)``."""

    amplitude: sp.Expr
    phase: sp.Expr
    behavior: PhaseBehavior
    frequency_scale: sp.Expr
    certified: bool


@dataclass(frozen=True)
class CertifiedOscillatoryExpansion:
    """Exact oscillatory representation on one certified asymptotic cell."""

    expression: sp.Expr
    representation: sp.Expr
    regime: object
    scales: tuple[OscillatoryScale, ...]
    order: int
    certified: bool
    method: str
    obligations: tuple[ProofObligation, ...] = ()


@dataclass(frozen=True)
class MultivariatePhaseGeometry:
    """Valuation of a monomial phase under a weighted approach to a target."""

    phase: sp.Expr
    valuation: Valuation
    radial_order: sp.Expr
    angular_factor: sp.Expr
    behavior: PhaseBehavior
    certified: bool
    statement: str
    obligations: tuple[ProofObligation, ...] = ()


def _real_expression(expr: sp.Expr) -> bool:
    return sp.sympify(expr).is_real is True


def _inverse_power_order(expr: sp.Expr, variable: sp.Symbol):
    """Recognize ``c*u**q`` with real nonzero c and exact integer q."""
    expr = sp.factor_terms(sp.sympify(expr))
    powers = expr.as_powers_dict()
    exponent = powers.get(variable, sp.S.Zero)
    if exponent.is_Integer is not True:
        return None
    coefficient = sp.simplify(expr / variable**exponent)
    if variable in coefficient.free_symbols or coefficient.is_real is not True:
        return None
    if coefficient.is_zero is not False:
        return None
    return coefficient, int(exponent)


def _phase_behavior(expr: sp.Expr, variable: sp.Symbol):
    data = _inverse_power_order(expr, variable)
    if data is None:
        return PhaseBehavior.UNKNOWN, None
    coefficient, exponent = data
    if exponent < 0:
        return PhaseBehavior.DIVERGENT, sp.Abs(coefficient) * variable**exponent
    if exponent > 0:
        return PhaseBehavior.VANISHING, sp.Abs(coefficient) * variable**exponent
    return PhaseBehavior.FINITE, sp.Abs(coefficient)


def _oscillatory_terms(expr: sp.Expr) -> tuple[tuple[sp.Expr, sp.Expr], ...] | None:
    """Return exact (amplitude, phase) exponential terms for elementary oscillation."""
    expr = sp.sympify(expr)
    if expr.func is sp.sin:
        phase = expr.args[0]
        return ((1 / (2 * sp.I), phase), (-1 / (2 * sp.I), -phase))
    if expr.func is sp.cos:
        phase = expr.args[0]
        return ((sp.Rational(1, 2), phase), (sp.Rational(1, 2), -phase))
    if expr.func is sp.exp:
        exponent = expr.args[0]
        phase = sp.simplify(exponent / sp.I)
        if sp.simplify(exponent - sp.I * phase) == 0 and _real_expression(phase):
            return ((sp.S.One, phase),)
    if expr.is_Mul:
        for factor in expr.args:
            terms = _oscillatory_terms(factor)
            if terms is not None:
                amplitude = sp.simplify(expr / factor)
                if _real_expression(amplitude) or amplitude.is_complex is True:
                    return tuple((sp.simplify(amplitude * a), p) for a, p in terms)
    return None


def certified_oscillatory_expand(
    expr, *, small, large, order=3, statement="explicit oscillatory regime"
):
    """Certify an elementary oscillatory cell under ``small/large=o(1)``.

    The oscillatory-scale engine handles exact sin/cos and amplitude times ``exp(I*phi)`` when the
    transformed phase is a real inverse integral power of the positive cell
    coordinate.  More general phases decline conservatively.
    """
    from .stratified_expansion import ExpansionRegime, SmallQuantity

    expr = sp.sympify(expr)
    small = sp.sympify(small)
    large = sp.sympify(large)
    ratio = sp.simplify(small / large)
    u = sp.Dummy("osc_u", positive=True)
    transformed = sp.simplify(expr.subs(small, u * large))
    terms = _oscillatory_terms(transformed)
    if terms is None:
        return None

    scales = []
    for amplitude, phase in terms:
        if not _real_expression(phase):
            return None
        behavior, frequency = _phase_behavior(phase, u)
        if behavior is not PhaseBehavior.DIVERGENT:
            return None
        scales.append(OscillatoryScale(amplitude, phase, behavior, frequency, True))

    regime = ExpansionRegime((SmallQuantity(ratio),), statement=statement)
    representation_u = sp.Add(
        *(scale.amplitude * sp.exp(sp.I * scale.phase) for scale in scales)
    )
    representation = sp.simplify(representation_u.subs(u, ratio))
    return CertifiedOscillatoryExpansion(
        expr,
        representation,
        regime,
        tuple(scales),
        order,
        True,
        "exact_divergent_phase",
    )


def phase_geometry(phase, variables, *, target=None, weights=None):
    """Classify a monomial phase by its weighted radial valuation.

    The geometry theorem is exact: the translated
    phase must be a Laurent monomial.  This covers phases such as ``x**2/y**3``
    without pretending that cancellation-prone sums have one global valuation.
    """
    phase = sp.sympify(phase)
    variables = tuple(variables)
    target = tuple(sp.S.Zero for _ in variables) if target is None else tuple(target)
    weights = tuple(1 for _ in variables) if weights is None else tuple(weights)
    valuation = Valuation(variables, weights)
    translated = sp.expand(
        phase.subs({v: v + a for v, a in zip(variables, target, strict=True)})
    )
    powers = translated.as_powers_dict()
    exponents = []
    for variable in variables:
        exponent = powers.get(variable, sp.S.Zero)
        if exponent.is_Integer is not True:
            return _unknown_geometry(
                phase, valuation, "phase is not an exact Laurent monomial"
            )
        exponents.append(int(exponent))
    angular_factor = sp.simplify(
        translated / sp.Mul(*(v**e for v, e in zip(variables, exponents, strict=True)))
    )
    if (
        any(v in angular_factor.free_symbols for v in variables)
        or angular_factor.is_zero is not False
    ):
        return _unknown_geometry(
            phase, valuation, "phase coefficient is not certified nonzero"
        )
    radial_order = sum(w * e for w, e in zip(valuation.weights, exponents, strict=True))
    behavior = (
        PhaseBehavior.DIVERGENT
        if radial_order < 0
        else PhaseBehavior.VANISHING
        if radial_order > 0
        else PhaseBehavior.FINITE
    )
    return MultivariatePhaseGeometry(
        phase,
        valuation,
        sp.Integer(radial_order),
        angular_factor,
        behavior,
        True,
        f"weighted Laurent-monomial phase has radial order {radial_order}",
    )


def _unknown_geometry(phase, valuation, statement):
    obligation = ProofObligation(
        ObligationKind.ORDER_RELATION,
        statement,
        provider="phase_geometry",
        expression=sp.sstr(phase),
    )
    return MultivariatePhaseGeometry(
        phase,
        valuation,
        sp.nan,
        sp.S.NaN,
        PhaseBehavior.UNKNOWN,
        False,
        statement,
        (obligation,),
    )
