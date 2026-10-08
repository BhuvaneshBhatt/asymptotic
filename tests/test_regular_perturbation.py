import sympy as sp

from asymptotic import regular_perturbation
from asymptotic.regular_perturbation import (
    RegularPerturbationBranch,
    RegularPerturbationResult,
)


def test_scalar_algebraic_regular_perturbation_preserves_branches():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")

    result = regular_perturbation(u**2 - 1 - eps, u, eps, order=2, return_result=True)

    assert isinstance(result, RegularPerturbationResult)
    assert result.complete
    assert len(result.branches) == 2
    assert all(
        isinstance(branch, RegularPerturbationBranch) for branch in result.branches
    )
    assert all(branch.verified for branch in result.branches)
    assert set(result.approximations) == {
        (-1 - eps / 2 + eps**2 / 8,),
        (1 + eps / 2 - eps**2 / 8,),
    }
    for branch in result.branches:
        residual = sp.expand(branch.residual[0])
        assert residual.coeff(eps, 0) == 0
        assert residual.coeff(eps, 1) == 0
        assert residual.coeff(eps, 2) == 0


def test_algebraic_system_is_solved_recursively_on_one_hierarchy():
    eps = sp.Symbol("eps", positive=True)
    u, v = sp.symbols("u v")

    result = regular_perturbation(
        (u + v - 3 - eps, u - v + 1), (u, v), eps, order=1, return_result=True
    )

    assert result.complete
    assert result.approximations == ((1 + eps / 2, 2 + eps / 2),)
    assert result.branches[0].verified


def test_scalar_ode_ivp_propagates_initial_data_order_by_order():
    eps, x = sp.symbols("eps x", positive=True)
    y = sp.Function("y")

    result = regular_perturbation(
        sp.diff(y(x), x) + y(x) + eps * y(x) ** 2,
        y(x),
        eps,
        order=1,
        conditions=sp.Eq(y(0), 1 + eps),
        return_result=True,
    )

    assert result.complete
    branch = result.branches[0]
    assert branch.verified
    assert branch.approximation == (sp.exp(-x) + eps * sp.exp(-2 * x),)
    assert branch.condition_residual == (0,)
    residual = sp.expand(branch.residual[0])
    assert residual.coeff(eps, 0) == 0
    assert residual.coeff(eps, 1) == 0


def test_second_order_bvp_fixes_each_order():
    eps = sp.Symbol("eps", positive=True)
    x = sp.Symbol("x")
    y = sp.Function("y")

    result = regular_perturbation(
        sp.diff(y(x), x, 2) + eps * y(x),
        y(x),
        eps,
        order=1,
        conditions=(sp.Eq(y(0), 0), sp.Eq(y(1), 1)),
        return_result=True,
    )

    assert result.complete
    approximation = result.branches[0].approximation[0]
    expected = x + eps * (x - x**3) / 6
    assert sp.simplify(approximation - expected) == 0
    assert result.branches[0].condition_residual == (0, 0)
    assert result.branches[0].verified


def test_coupled_ode_system_is_supported_from_the_same_entry_point():
    eps = sp.Symbol("eps", positive=True)
    x = sp.Symbol("x")
    y = sp.Function("y")
    z = sp.Function("z")

    result = regular_perturbation(
        (
            sp.diff(y(x), x) - z(x) - eps * y(x),
            sp.diff(z(x), x) + y(x),
        ),
        (y(x), z(x)),
        eps,
        order=1,
        conditions=(sp.Eq(y(0), 1), sp.Eq(z(0), 0)),
        return_result=True,
    )

    assert result.complete
    y_approx, z_approx = result.branches[0].approximation
    assert (
        sp.simplify(y_approx - (sp.cos(x) + eps * (x * sp.cos(x) + sp.sin(x)) / 2)) == 0
    )
    assert sp.simplify(z_approx - (-sp.sin(x) - eps * x * sp.sin(x) / 2)) == 0
    assert result.branches[0].verified


def test_custom_regular_gauge_sequence():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    gauge = eps * sp.log(1 / eps)

    result = regular_perturbation(
        u - 1 - gauge, u, eps, gauges=(1, gauge), return_result=True
    )

    assert result.complete
    assert sp.simplify(result.approximations[0][0] - (1 + gauge)) == 0
    assert result.branches[0].verified


def test_unsolved_order_is_retained_as_a_partial_branch():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")

    result = regular_perturbation(u + sp.cos(u), u, eps, order=1, return_result=True)

    assert not result.complete
    assert len(result.branches) == 1
    branch = result.branches[0]
    assert not branch.complete
    assert branch.unresolved_order == 0
    assert branch.hierarchy.solved_through is None
    assert "order 0" in branch.limitation
    assert not branch.verified
