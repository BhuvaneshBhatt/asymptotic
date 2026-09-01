"""Nonlinear ODE and Green-operator reference cases."""

import sympy as sp

from asymptotic import RemainderKind
from asymptotic.nonlinear_ode import nonlinear_differential_transseries
from asymptotic.remainder_theorems import certify_green_inverse_operator_remainder

from . import CapabilityStatus, ReferenceCase


def _green_asymptotically_constant() -> bool:
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.diff(delta(x), x) / x - delta(x)
    cert, green = certify_green_inverse_operator_remainder(
        sp.exp(-x / 2), operator, delta, x, sp.oo
    )
    return (
        cert.certified
        and cert.conclusion.kind is RemainderKind.BIG_O
        and bool(green and green.replay_asymptotic(x))
    )


def _nonconvergent_green_unknown() -> bool:
    x = sp.symbols("x", positive=True)
    delta = sp.Function("delta")
    operator = sp.diff(delta(x), x, 2) + sp.sin(x) * sp.diff(delta(x), x) - delta(x)
    cert, _ = certify_green_inverse_operator_remainder(sp.exp(-x / 2), operator, delta, x, sp.oo)
    return (not cert.certified) and cert.conclusion.kind is RemainderKind.UNKNOWN


def _nonlinear_ode_branch() -> bool:
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    equation = x * sp.diff(y(x), x) - y(x) + y(x) ** 2
    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=3)
    return len(branches) >= 1


CASES = (
    ReferenceCase(
        "asymptotically-constant-green",
        "green",
        CapabilityStatus.CERTIFIED,
        _green_asymptotically_constant,
    ),
    ReferenceCase(
        "nonconvergent-green", "green", CapabilityStatus.UNKNOWN, _nonconvergent_green_unknown
    ),
    ReferenceCase(
        "nonlinear-logistic-balance",
        "nonlinear-ode",
        CapabilityStatus.FORMAL,
        _nonlinear_ode_branch,
    ),
)
