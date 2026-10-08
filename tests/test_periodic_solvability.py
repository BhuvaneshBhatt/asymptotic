import sympy as sp

from asymptotic.perturbation import SolvabilityCondition
from asymptotic.resonance import (
    PeriodicSolvabilityResult,
    periodic_solvability_conditions,
)


def test_periodic_projection_returns_fredholm_adjoint_mode():
    tau = sp.Symbol("tau", real=True)
    a, b = sp.symbols("a b")

    result = periodic_solvability_conditions(
        a * sp.cos(tau) + b * sp.sin(tau) + sp.cos(2 * tau),
        tau,
        (sp.cos(tau), sp.sin(tau)),
        source_order=3,
    )

    assert isinstance(result, PeriodicSolvabilityResult)
    assert result.projections == (a / 2, b / 2)
    assert all(
        isinstance(condition, SolvabilityCondition) for condition in result.conditions
    )
    assert all(condition.source_order == 3 for condition in result.conditions)
    assert result.satisfied is None


def test_periodic_projection_certifies_nonresonant_forcing():
    tau = sp.Symbol("tau", real=True)

    result = periodic_solvability_conditions(
        sp.cos(2 * tau),
        tau,
        (sp.cos(tau), sp.sin(tau)),
    )

    assert result.projections == (0, 0)
    assert result.satisfied is True


def test_periodic_projection_rejects_empty_adjoint_kernel():
    tau = sp.Symbol("tau")

    try:
        periodic_solvability_conditions(sp.cos(tau), tau, ())
    except ValueError as exc:
        assert "adjoint" in str(exc)
    else:
        raise AssertionError("empty adjoint mode family was accepted")
