"""Durable Laplace, degenerate, and uniform-saddle reference cases."""

import sympy as sp

from asymptotic import (
    RemainderKind,
    airy_uniform_saddle_asymptotic,
    coalescing_saddle_asymptotic,
    laplace_asymptotic_integral,
)

from . import CapabilityStatus, ReferenceCase


def _canonical_airy() -> bool:
    n = sp.symbols("n", positive=True)
    mu = sp.symbols("mu", real=True)
    x = sp.symbols("x", real=True)
    result = airy_uniform_saddle_asymptotic(
        sp.exp(sp.I * n * (x**3 / 3 + mu * x)),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        control_parameter=mu,
    )
    expected = 2 * sp.pi * n ** (-sp.Rational(1, 3)) * sp.airyai(mu * n ** sp.Rational(2, 3))
    return result.status == "FORMAL" and sp.simplify(result.expression - expected) == 0


def _quartic_coalescence() -> bool:
    n = sp.symbols("n", positive=True)
    mu = sp.symbols("mu", real=True)
    x = sp.symbols("x", real=True)
    result = coalescing_saddle_asymptotic(
        sp.exp(-n * (x**4 / 4 + mu * x**2 / 2)),
        x,
        (-sp.oo, sp.oo),
        parameter=n,
        control_parameter=mu,
    )
    return result.status == "FORMAL" and result.expression.has(mu * sp.sqrt(n))


def _certified_quartic_laplace() -> bool:
    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)
    result = laplace_asymptotic_integral(
        sp.exp(-n * x**4), x, (-sp.oo, sp.oo), parameter=n, terms=2
    )
    return (
        result.status == "CERTIFIED"
        and result.remainder is not None
        and result.remainder.kind is RemainderKind.BIG_O
    )


def _unsupported_non_cubic_turning_point() -> bool:
    n = sp.symbols("n", positive=True)
    mu = sp.symbols("mu", real=True)
    x = sp.symbols("x", real=True)
    try:
        airy_uniform_saddle_asymptotic(
            sp.exp(sp.I * n * (x**4 / 4 + mu * x)),
            x,
            (-sp.oo, sp.oo),
            parameter=n,
            control_parameter=mu,
        )
    except NotImplementedError:
        return True
    return False


CASES = (
    ReferenceCase(
        "canonical-airy-turning-point", "uniform-saddle", CapabilityStatus.FORMAL, _canonical_airy
    ),
    ReferenceCase(
        "quartic-coalescence", "uniform-saddle", CapabilityStatus.FORMAL, _quartic_coalescence
    ),
    ReferenceCase(
        "quartic-laplace", "laplace", CapabilityStatus.CERTIFIED, _certified_quartic_laplace
    ),
    ReferenceCase(
        "non-cubic-airy-refusal",
        "uniform-saddle",
        CapabilityStatus.UNKNOWN,
        _unsupported_non_cubic_turning_point,
    ),
)
