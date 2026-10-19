"""Asymptotic function-head adapter and limit-only fallback policy."""

import sympy as sp
from exprtools._recurrence_families import RecurrenceFamilies

from .reference_normalization import RegularizedHypergeometric0F1
from .special_functions import (
    HypergeometricU,
    KelvinBei,
    KelvinBer,
    KelvinKei,
    KelvinKer,
    LegendreQ,
    ParabolicCylinderD,
    RegularizedHypergeometric1F1,
    RegularizedHypergeometric2F1,
    StruveH,
    StruveL,
    WhittakerM,
    WhittakerW,
)

FAMILIES = RecurrenceFamilies(
    RegularizedHypergeometric0F1=RegularizedHypergeometric0F1,
    HypergeometricU=HypergeometricU,
    KelvinBei=KelvinBei,
    KelvinBer=KelvinBer,
    KelvinKei=KelvinKei,
    KelvinKer=KelvinKer,
    LegendreQ=LegendreQ,
    ParabolicCylinderD=ParabolicCylinderD,
    RegularizedHypergeometric1F1=RegularizedHypergeometric1F1,
    RegularizedHypergeometric2F1=RegularizedHypergeometric2F1,
    StruveH=StruveH,
    StruveL=StruveL,
    WhittakerM=WhittakerM,
    WhittakerW=WhittakerW,
)
POLYNOMIALS = FAMILIES.POLYNOMIALS
KELVIN = FAMILIES.KELVIN
EXTRA_HEADS = FAMILIES.EXTRA_HEADS
TWO_BASIS = FAMILIES.TWO_BASIS
signatures = FAMILIES.signatures
function_at = FAMILIES.function_at
next_basis = FAMILIES.next_basis


def uncertified_germ_reason(expr, variables, target):
    """Prevent generic native/composition fallback from inventing new germs.

    Called only after the certified first pass and recurrence retry. Exposing a
    standard function's definition does not certify a parameter-dependent limit.
    """
    if not isinstance(expr, sp.Expr):
        return None
    opaque = (
        ParabolicCylinderD,
        HypergeometricU,
        WhittakerM,
        WhittakerW,
        RegularizedHypergeometric1F1,
        RegularizedHypergeometric2F1,
        LegendreQ,
        *KELVIN,
    )
    for atom in expr.atoms(*opaque):
        if atom.has(*variables):
            return "a translated special-function germ needs a dedicated certified provider after exact recurrence reduction"
    for atom in expr.atoms(*POLYNOMIALS):
        if not atom.has(*variables):
            continue
        degree = atom.args[0]
        if degree.has(*variables):
            return "a varying polynomial degree needs a controlled large-degree asymptotic theorem"
        if degree.is_integer is not True or degree.is_nonnegative is not True:
            return "a non-polynomial degree or associated-function endpoint needs a branch-aware germ theorem"
    for atom in expr.atoms(StruveH, StruveL):
        if not atom.has(*variables):
            continue
        if any(p in (sp.oo, -sp.oo) for p in target):
            return "a Struve infinity germ or cancellation needs a propagated asymptotic remainder"
        argument = atom.args[-1]
        if sp.count_ops(argument) > 32:
            return "a composed Struve germ needs a controlled argument chart"
        boundary = argument.subs(dict(zip(variables, target)))
        if boundary.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            return "a composed Struve infinity germ needs a propagated asymptotic remainder"
    return None
