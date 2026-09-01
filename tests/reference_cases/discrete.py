"""Durable recurrence and Birkhoff--Trjitzinsky reference cases."""

import sympy as sp

from asymptotic import asymptotic_rsolve
from asymptotic.discrete_scale import (
    birkhoff_trjitzinsky_branches,
    linear_recurrence_data,
)

from . import CapabilityStatus, ReferenceCase


def _harmonic_resonance() -> bool:
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    result = asymptotic_rsolve(a(n + 1) - a(n) - 1 / n, a(n), n, terms=4)
    particular = result.particular_expression
    if particular is None or not particular.has(sp.log(n)):
        return False
    defect = sp.series((particular.subs(n, n + 1) - particular - 1 / n), n, sp.oo, 5).removeO()
    return sp.expand(defect) == 0


def _repeated_secondary_tertiary_descent() -> bool:
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = (
        n**2 * (a(n + 4) - 4 * a(n + 3) + 6 * a(n + 2) - 4 * a(n + 1) + a(n))
        - 2 * n * (a(n + 2) - 2 * a(n + 1) + a(n))
        + a(n)
    )
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=1)
    return (
        len(branches) == 4
        and all(branch.secondary_mult == 2 for branch in branches)
        and all(branch.lattice_step == sp.Rational(1, 4) for branch in branches)
        and all(branch.residual_order is not None for branch in branches)
    )


CASES = (
    ReferenceCase(
        "first-order-log-resonance", "recurrence", CapabilityStatus.FORMAL, _harmonic_resonance
    ),
    ReferenceCase(
        "repeated-secondary-tertiary-descent",
        "birkhoff-trjitzinsky",
        CapabilityStatus.FORMAL,
        _repeated_secondary_tertiary_descent,
    ),
)
