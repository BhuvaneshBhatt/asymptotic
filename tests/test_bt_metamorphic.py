"""Metamorphic invariants for repeated-secondary BT lifting."""

import pytest
import sympy as sp

from asymptotic.discrete_scale import (
    birkhoff_trjitzinsky_branches,
    linear_recurrence_data,
)


def _problem():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = (
        n**2 * (a(n + 4) - 4 * a(n + 3) + 6 * a(n + 2) - 4 * a(n + 1) + a(n))
        - 2 * n * (a(n + 2) - 2 * a(n + 1) + a(n))
        + a(n)
    )
    return n, a, recurrence


def _lift(recurrence):
    n, a, _ = _problem()
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=1)
    signature = {
        (
            sp.expand(branch.scale.phase),
            sp.simplify(branch.scale.power),
            branch.secondary_mult,
            branch.tertiary_mult,
            branch.lattice_step,
        )
        for branch in branches
    }
    return signature, data, branches


@pytest.fixture(scope="module")
def scalar_equivalent_lifts():
    _, _, recurrence = _problem()
    return _lift(recurrence), _lift(7 * recurrence)


def test_repeated_secondary_bt_is_invariant_under_nonzero_scalar_multiple(scalar_equivalent_lifts):
    baseline, scaled = scalar_equivalent_lifts
    assert scaled[0] == baseline[0]


def test_metamorphic_repeated_secondary_branches_keep_residual_replay_and_hierarchy(
    scalar_equivalent_lifts,
):
    for signature, data, branches in scalar_equivalent_lifts:
        assert len(signature) == 4
        assert all(branch.secondary_mult == 2 for branch in branches)
        assert all(branch.lattice_step == sp.Rational(1, 4) for branch in branches)
        assert all(branch.residual_order is not None for branch in branches)
