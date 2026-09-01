"""Elementary, composition, and remainder reference cases."""

import sympy as sp

from asymptotic import (
    AsymptoticRemainder,
    RemainderKind,
    multiseries,
)
from asymptotic.remainder_theorems import certify_reciprocal_remainder

from . import CapabilityStatus, ReferenceCase


def _elementary_multiseries() -> bool:
    x = sp.symbols("x", positive=True)
    result = multiseries(sp.exp(1 / x), x, scale=[1 / x], terms=4)
    return sp.simplify(result.truncate(4) - (1 + 1 / x + 1 / (2 * x**2) + 1 / (6 * x**3))) == 0


def _certified_reciprocal() -> bool:
    x = sp.symbols("x", positive=True)
    remainder = AsymptoticRemainder.big_o(x**-2, x, sp.oo)
    cert = certify_reciprocal_remainder(1 + 1 / x, remainder)
    return cert.certified and cert.conclusion.kind is RemainderKind.BIG_O


def _oscillatory_reciprocal_unknown() -> bool:
    x = sp.symbols("x", real=True)
    exact = AsymptoticRemainder.exact_zero(x, sp.oo)
    cert = certify_reciprocal_remainder(sp.sin(x), exact)
    return (not cert.certified) and cert.conclusion.kind is RemainderKind.UNKNOWN


CASES = (
    ReferenceCase(
        "exp-small-parameter", "elementary", CapabilityStatus.FORMAL, _elementary_multiseries
    ),
    ReferenceCase(
        "stable-reciprocal", "remainders", CapabilityStatus.CERTIFIED, _certified_reciprocal
    ),
    ReferenceCase(
        "oscillatory-reciprocal",
        "remainders",
        CapabilityStatus.UNKNOWN,
        _oscillatory_reciprocal_unknown,
    ),
)
