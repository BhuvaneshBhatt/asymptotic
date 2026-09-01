import sympy as sp

from asymptotic import (
    AsymptoticDSolveResult,
    AsymptoticRSolveResult,
    asymptotic_dsolve,
    asymptotic_rsolve,
)


def test_asymptotic_dsolve_dispatches_to_nonlinear_transseries():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 2 / x + 3 / x**2
    forcing = sp.diff(target, x) - target**2
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing
    result = asymptotic_dsolve(
        equation,
        y,
        x,
        point=sp.oo,
        terms=4,
        method="nonlinear",
    )
    assert isinstance(result, AsymptoticDSolveResult)
    assert any(sp.simplify(solution - target) == 0 for solution in result.solutions)
    assert result.method == "nonlinear-differential-transseries"


def test_asymptotic_rsolve_exact_then_asymptotic_route():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    result = asymptotic_rsolve(
        a(n + 1) - 2 * a(n),
        a(n),
        n,
        initial_conditions={a(0): 1},
    )
    assert isinstance(result, AsymptoticRSolveResult)
    assert sp.simplify(result.expression / 2**n - 1) == 0
    assert result.method == "exact-rsolve-then-asymptotic"


def test_asymptotic_rsolve_refuses_unsolved_recurrence():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    try:
        asymptotic_rsolve(sp.sin(a(n + 1)) - a(n), a(n), n)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unsupported nonlinear recurrence should not be guessed")


def test_asymptotic_dsolve_residual_contract_for_reported_solution():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 2 / x + 3 / x**2
    forcing = sp.diff(target, x) - target**2
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing
    result = asymptotic_dsolve(equation, y, x, point=sp.oo, terms=4, method="nonlinear")
    residuals = result.residuals(equation)
    assert residuals
    assert any(residual == 0 or residual.is_zero is True for residual in residuals)


def test_asymptotic_rsolve_residual_contract_for_exact_solution():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - 2 * a(n)
    result = asymptotic_rsolve(recurrence, a(n), n, initial_conditions={a(0): 1})
    residual = sp.factor(result.residual(recurrence))
    assert residual == 0 or residual.is_zero is True


def test_asymptotic_rsolve_residual_contract_for_inhomogeneous_solution():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - a(n) - 1
    result = asymptotic_rsolve(recurrence, a(n), n, initial_conditions={a(0): 0})
    residual = sp.factor(result.residual(recurrence))
    assert residual == 0 or residual.is_zero is True
