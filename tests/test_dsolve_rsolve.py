import sympy as sp

from asymptotic import (
    DSolveResult,
    RSolveResult,
    dsolve,
    rsolve,
)


def test_dsolve_dispatches_to_nonlinear_transseries():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 2 / x + 3 / x**2
    forcing = sp.diff(target, x) - target**2
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing
    result = dsolve(
        equation, y, x, point=sp.oo, terms=4, method="nonlinear", return_result=True
    )
    assert isinstance(result, DSolveResult)
    assert any(sp.simplify(solution - target) == 0 for solution in result.solutions)
    assert result.method == "nonlinear-differential-transseries"


def test_rsolve_exact_then_asymptotic_route():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    result = rsolve(
        a(n + 1) - 2 * a(n), a(n), n, initial_conditions={a(0): 1}, return_result=True
    )
    assert isinstance(result, RSolveResult)
    assert sp.simplify(result.expression / 2**n - 1) == 0
    assert result.method == "exact-rsolve-then-asymptotic"


def test_rsolve_refuses_unsolved_recurrence():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    try:
        rsolve(sp.sin(a(n + 1)) - a(n), a(n), n, return_result=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unsupported nonlinear recurrence should not be guessed")


def test_dsolve_residual_contract_for_reported_solution():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 2 / x + 3 / x**2
    forcing = sp.diff(target, x) - target**2
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing
    result = dsolve(
        equation, y, x, point=sp.oo, terms=4, method="nonlinear", return_result=True
    )
    residuals = result.residuals(equation)
    assert residuals
    assert any(residual == 0 or residual.is_zero is True for residual in residuals)


def test_rsolve_residual_contract_for_exact_solution():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - 2 * a(n)
    result = rsolve(
        recurrence, a(n), n, initial_conditions={a(0): 1}, return_result=True
    )
    residual = sp.factor(result.residual(recurrence))
    assert residual == 0 or residual.is_zero is True


def test_rsolve_residual_contract_for_inhomogeneous_solution():
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - a(n) - 1
    result = rsolve(
        recurrence, a(n), n, initial_conditions={a(0): 0}, return_result=True
    )
    residual = sp.factor(result.residual(recurrence))
    assert residual == 0 or residual.is_zero is True
