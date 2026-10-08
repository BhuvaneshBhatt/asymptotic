import sympy as sp

from asymptotic import lindstedt_poincare
from asymptotic.lindstedt import LindstedtPoincareResult


def _zero_velocity(y, t):
    return sp.Eq(sp.diff(y(t), t).subs(t, 0), 0)


def test_duffing_lindstedt_expansion_derives_second_order():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")

    result = lindstedt_poincare(
        sp.diff(y(t), t, 2) + y(t) + eps * y(t) ** 3,
        y(t),
        eps,
        order=2,
        conditions=(sp.Eq(y(0), 1), _zero_velocity(y, t)),
        return_result=True,
    )

    assert isinstance(result, LindstedtPoincareResult)
    assert result.complete
    assert result.unresolved_order is None
    assert result.verified
    assert sp.simplify(result.frequency - (1 + 3 * eps / 8 - 21 * eps**2 / 256)) == 0
    assert result.hierarchy.order(1).solvability_conditions
    assert result.hierarchy.order(2).solvability_conditions
    assert not result.profile.has(result.fast_variable * sp.sin(result.fast_variable))


def test_base_frequency_is_inferred_for_a_nonunit_linear_oscillator():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")

    result = lindstedt_poincare(
        sp.diff(y(t), t, 2) + 4 * y(t) + eps * y(t) ** 3,
        y(t),
        eps,
        order=1,
        conditions=(sp.Eq(y(0), 1), _zero_velocity(y, t)),
        return_result=True,
    )

    assert result.complete
    assert result.verified
    assert sp.simplify(result.frequency - (2 + 3 * eps / 16)) == 0


def test_explicit_base_frequency_can_be_supplied():
    eps, omega0 = sp.symbols("eps omega0", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")

    result = lindstedt_poincare(
        sp.diff(y(t), t, 2) + omega0**2 * y(t),
        y(t),
        eps,
        order=1,
        base_frequency=omega0,
        conditions=(sp.Eq(y(0), 1), _zero_velocity(y, t)),
        return_result=True,
    )

    assert result.complete
    assert result.verified
    assert result.frequency == omega0


def test_damped_unperturbed_problem_is_rejected_as_non_lindstedt_base():
    eps = sp.Symbol("eps", positive=True)
    t = sp.Symbol("t", real=True)
    y = sp.Function("y")

    try:
        lindstedt_poincare(
            sp.diff(y(t), t, 2) + sp.diff(y(t), t) + y(t) + eps * y(t) ** 3,
            y(t),
            eps,
            order=1,
            return_result=True,
        )
    except NotImplementedError as exc:
        assert "frequency" in str(exc)
    else:
        raise AssertionError("damped base oscillator was accepted")
