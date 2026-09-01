"""Deep repeated-secondary Birkhoff--Trjitzinsky regression."""

import pytest
import sympy as sp

from asymptotic.discrete_scale import (
    birkhoff_trjitzinsky_branches,
    linear_recurrence_data,
)


@pytest.fixture(scope="module")
def tertiary_bt_case():
    """Compute the canonical repeated-secondary lift once for this module."""

    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = (
        n**2 * (a(n + 4) - 4 * a(n + 3) + 6 * a(n + 2) - 4 * a(n + 1) + a(n))
        - 2 * n * (a(n + 2) - 2 * a(n + 1) + a(n))
        + a(n)
    )
    data = linear_recurrence_data(recurrence, a(n), n)
    return n, data, birkhoff_trjitzinsky_branches(data, terms=1)


def test_repeated_secondary_root_descends_to_tertiary_stretched_phase(tertiary_bt_case):
    n, data, branches = tertiary_bt_case
    assert len(branches) == 4
    phases = {sp.expand(branch.scale.phase) for branch in branches}
    assert phases == {
        2 * sp.sqrt(n) - 2 * sp.sqrt(2) * n ** sp.Rational(1, 4),
        2 * sp.sqrt(n) + 2 * sp.sqrt(2) * n ** sp.Rational(1, 4),
        -2 * sp.sqrt(n) - 2 * sp.sqrt(2) * sp.I * n ** sp.Rational(1, 4),
        -2 * sp.sqrt(n) + 2 * sp.sqrt(2) * sp.I * n ** sp.Rational(1, 4),
    }
    assert all(branch.secondary_mult == 2 for branch in branches)
    assert all(branch.lattice_step == sp.Rational(1, 4) for branch in branches)
    assert all(branch.scale.power == sp.Rational(3, 8) for branch in branches)
    assert all(branch.replay_residual(data) is True for branch in branches)
