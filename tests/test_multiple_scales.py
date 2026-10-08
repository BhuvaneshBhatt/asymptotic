from __future__ import annotations

import sympy as sp

from asymptotic.multiple_scales import (
    MultipleScalesResult,
    multiple_scale_derivative,
    multiple_scales,
)


def test_total_derivative_expands_arbitrary_order():
    eps = sp.symbols("eps")
    t0, t1 = sp.symbols("T0 T1")
    u = sp.Function("u")(t0, t1)
    result = multiple_scale_derivative(u, (t0, t1), eps, derivative_order=2)
    assert (
        sp.expand(
            result
            - (
                sp.diff(u, t0, 2)
                + 2 * eps * sp.diff(u, t0, t1)
                + eps**2 * sp.diff(u, t1, 2)
            )
        )
        == 0
    )


def test_nonstandard_scale_orders_are_supported():
    eps = sp.symbols("eps")
    t0, t3 = sp.symbols("T0 T3")
    u = sp.Function("u")(t0, t3)
    result = multiple_scale_derivative(u, (t0, t3), eps, scale_orders=(0, 3))
    assert result == sp.diff(u, t0) + eps**3 * sp.diff(u, t3)


def test_duffing_with_weak_damping_derives_slow_flow():
    t = sp.symbols("t", real=True)
    eps, mu, alpha = sp.symbols("eps mu alpha", real=True)
    y = sp.Function("y")(t)
    result = multiple_scales(
        sp.diff(y, t, 2) + y + eps * (2 * mu * sp.diff(y, t) + alpha * y**3),
        y,
        eps,
    )
    assert isinstance(result, MultipleScalesResult)
    assert result.complete
    assert len(result.slow_flow) == 2
    flow = {item.derivative: item.expression for item in result.slow_flow}
    t1 = result.scales[1]
    a = sp.Function("A_0")(t1)
    b = sp.Function("B_0")(t1)
    assert (
        sp.simplify(flow[sp.diff(a, t1)] - (3 * alpha * b * (a**2 + b**2) / 8 - mu * a))
        == 0
    )
    assert (
        sp.simplify(
            flow[sp.diff(b, t1)] - (-3 * alpha * a * (a**2 + b**2) / 8 - mu * b)
        )
        == 0
    )
    assert len(result.hierarchy.order(1).solvability_conditions) == 2


def test_near_resonant_forcing_uses_slow_detuning_phase():
    t = sp.symbols("t", real=True)
    eps, force, sigma = sp.symbols("eps F sigma", real=True)
    y = sp.Function("y")(t)
    result = multiple_scales(
        sp.diff(y, t, 2) + y - eps * force * sp.cos((1 + eps * sigma) * t),
        y,
        eps,
    )
    t1 = result.scales[1]
    a = sp.Function("A_0")(t1)
    b = sp.Function("B_0")(t1)
    flow = {item.derivative: item.expression for item in result.slow_flow}
    assert sp.simplify(flow[sp.diff(a, t1)] - force * sp.sin(sigma * t1) / 2) == 0
    assert sp.simplify(flow[sp.diff(b, t1)] - force * sp.cos(sigma * t1) / 2) == 0


def test_weakly_coupled_equal_frequency_system_derives_joint_flow():
    t = sp.symbols("t", real=True)
    eps, k = sp.symbols("eps k", real=True)
    y = sp.Function("y")(t)
    z = sp.Function("z")(t)
    result = multiple_scales(
        (
            sp.diff(y, t, 2) + y + eps * k * (y - z),
            sp.diff(z, t, 2) + z + eps * k * (z - y),
        ),
        (y, z),
        eps,
    )
    assert result.complete
    assert len(result.slow_flow) == 4
    assert len(result.profiles) == 2


def test_higher_requested_order_retains_symbolic_tail_conservatively():
    t = sp.symbols("t", real=True)
    eps = sp.symbols("eps")
    y = sp.Function("y")(t)
    result = multiple_scales(sp.diff(y, t, 2) + y + eps * y**3, y, eps, order=2)
    assert not result.complete
    assert result.unresolved_order == 2
    assert result.hierarchy.solved_through == 0
