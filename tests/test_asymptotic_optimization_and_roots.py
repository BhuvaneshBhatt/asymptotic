import sympy as sp
from sympy.stats import Normal

from asymptotic import (
    OptimizationResult,
    argmax,
    argmin,
    expectation,
    maximize,
    minimize,
    root,
)


def test_minimize_tracks_moving_stationary_point():
    x, n = sp.symbols("x n", positive=True)
    result = minimize((x - n) ** 2 + 1 / n, x, parameter=n, return_result=True)
    assert isinstance(result, OptimizationResult)
    assert result.optimizers == (n,)
    assert result.optimum_value == 1 / n
    assert result.certified
    assert argmin((x - n) ** 2 + 1 / n, x, parameter=n) == (n,)


def test_maximize_compares_stationary_point_and_closed_endpoints():
    x, n = sp.symbols("x n", positive=True)
    domain = sp.Interval(0, 2 * n)
    result = maximize(
        -((x - n) ** 2) + n, x, parameter=n, domain=domain, return_result=True
    )
    assert result.optimizers == (n,)
    assert result.optimum_value == n
    assert result.certified
    assert argmax(-((x - n) ** 2) + n, x, parameter=n, domain=domain) == (n,)


def test_open_interval_endpoint_is_not_reported_as_attained_optimizer():
    x, n = sp.symbols("x n", positive=True)
    result = minimize(
        x + 1 / n, x, parameter=n, domain=sp.Interval.open(0, 1), return_result=True
    )
    assert result.optimizers == ()
    assert result.status == "CERTIFIED"


def test_root_uses_mrv_hardy_expansion():
    y, n = sp.symbols("y n", positive=True)
    root_result = root(
        y**2 - (1 + 1 / n) * sp.exp(2 * n),
        y,
        parameter=n,
        terms=4,
        branch=1,
        return_result=True,
    )
    expected = sp.exp(n) * (1 + 1 / (2 * n) - 1 / (8 * n**2) + 1 / (16 * n**3))
    assert sp.simplify(root_result - expected) == 0


def test_root_can_filter_by_prescribed_limit():
    y, n = sp.symbols("y n", positive=True)
    result = root(y**2 - n**2, y, parameter=n, limit=sp.oo, return_result=True)
    assert len(result.branches) == 1
    assert result.branches[0].as_dict()[y] == n


def test_expectation_first_argument_is_the_random_expression():
    n = sp.symbols("n", positive=True)
    x = Normal("x", 1 / n, 1 / sp.sqrt(n))
    result = expectation(x**2 + 3 * x, parameter=n, terms=4, return_result=True)
    assert sp.simplify(result.expression - (3 / n + 1 / n + 1 / n**2)) == 0


def test_expectation_supports_symbol_bindings_and_conditions():
    n = sp.symbols("n", positive=True)
    z = sp.symbols("z", real=True)
    x = Normal("xb", 0, 1 / sp.sqrt(n))
    bound = expectation(z**2, parameter=n, bindings={z: x}, return_result=True)
    assert sp.simplify(bound.expression - 1 / n) == 0

    conditional = expectation(
        z, parameter=n, bindings={z: x}, condition=z > 0, return_result=True
    )
    assert sp.simplify(conditional.expression - sp.sqrt(2 / (sp.pi * n))) == 0


def test_integer_argmin_uses_floor_stationary_point():
    k = sp.symbols("k", integer=True)
    n = sp.symbols("n", positive=True, integer=True)
    objective = (k - n - sp.Rational(1, 3)) ** 2 + 1 / n
    result = minimize(
        objective, k, parameter=n, domain=sp.S.Integers, return_result=True
    )
    assert result.optimizers == (n,)
    assert sp.simplify(result.optimum_value - (sp.Rational(1, 9) + 1 / n)) == 0
    assert result.certified


def test_open_boundary_infimum_is_reported_without_false_argmin():
    x, n = sp.symbols("x n", positive=True)
    result = minimize(
        x + 1 / n, x, parameter=n, domain=sp.Interval.open(0, 1), return_result=True
    )
    assert result.optimum_value == 1 / n
    assert result.optimizers == ()
    assert result.approached_boundaries == (0,)
    assert result.attained is False
    assert result.status == "CERTIFIED"
