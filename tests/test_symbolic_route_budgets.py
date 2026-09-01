"""Algorithm-route budgets for Airy and repeated-secondary BT features."""

import sympy as sp

from asymptotic import airy_uniform_saddle_asymptotic
from asymptotic.discrete_scale import (
    birkhoff_trjitzinsky_branches,
    linear_recurrence_data,
)
from asymptotic.instrumentation import symbolic_metrics


def test_repeated_secondary_bt_avoids_general_symbolic_fallbacks():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = (
        n**2 * (a(n + 4) - 4 * a(n + 3) + 6 * a(n + 2) - 4 * a(n + 1) + a(n))
        - 2 * n * (a(n + 2) - 2 * a(n + 1) + a(n))
        + a(n)
    )
    with symbolic_metrics() as metrics:
        data = linear_recurrence_data(recurrence, a(n), n)
        branches = birkhoff_trjitzinsky_branches(data, terms=1)
    assert len(branches) == 4
    snapshot = metrics.snapshot()
    assert snapshot["general_solve_calls"] == 0
    assert snapshot["general_rsolve_calls"] == 0
    assert snapshot["general_limit_calls"] == 0
    assert snapshot["general_integrate_calls"] == 0


def test_airy_uniform_saddle_has_a_bounded_symbolic_route():
    n = sp.symbols("n", positive=True)
    mu = sp.symbols("mu", real=True)
    x = sp.symbols("x", real=True)
    with symbolic_metrics() as metrics:
        result = airy_uniform_saddle_asymptotic(
            sp.exp(sp.I * n * (x**3 / 3 + mu * x)),
            x,
            (-sp.oo, sp.oo),
            parameter=n,
            control_parameter=mu,
        )
    snapshot = metrics.snapshot()
    assert result.status == "FORMAL"
    assert snapshot["stat_coalescing_saddles"] == 1
    assert snapshot["general_solve_calls"] == 0
    assert snapshot["general_limit_calls"] == 0
    assert snapshot["general_integrate_calls"] == 0
