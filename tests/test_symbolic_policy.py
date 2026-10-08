from __future__ import annotations

import pytest
import sympy as sp

from asymptotic._symbolic_policy import (
    SymbolicPolicy,
    bounded_assumption_entails,
    bounded_assumption_sign,
    bounded_limit,
    bounded_polynomial_roots,
    bounded_primitive,
    bounded_simplify,
    bounded_solve_one,
    bounded_solve_system,
)


def test_polynomial_roots():
    x = sp.symbols("x")
    assert bounded_solve_one(3 * x - 6, x) == (2,)
    assert set(bounded_solve_one(x**2 - 1, x) or ()) == {-1, 1}


def test_root_degree_budget():
    x = sp.symbols("x")
    policy = SymbolicPolicy(polynomial_degree=3)
    assert bounded_polynomial_roots(x**4 - 1, x, policy=policy) is None


def test_linear_system():
    x, y = sp.symbols("x y")
    result = bounded_solve_system((x + y - 3, x - y - 1), (x, y))
    assert result == ({x: 2, y: 1},)


def test_rational_endpoints():
    x = sp.symbols("x", positive=True)
    assert bounded_limit((2 * x + 1) / (x + 3), x, sp.oo) == 2
    assert bounded_limit(x**2 / (1 + x), x, 0) == 0


def test_primitive_boundary():
    x = sp.symbols("x")
    assert bounded_primitive(x**2, x) == x**3 / 3
    assert bounded_primitive(sp.exp(-(x**2)), x) is None


def test_solver_opt_in():
    x = sp.symbols("x")
    equation = sp.exp(x) - 2
    assert bounded_solve_one(equation, x) is None
    assert bounded_solve_one(equation, x, allow_general=True) == (sp.log(2),)


def test_assumption_budget():
    x = sp.symbols("x", real=True)
    condition = sp.And(*[sp.Symbol(f"p{i}") > 0 for i in range(12)])
    policy = SymbolicPolicy(assumption_ops=2, satisfiable_ops=1)
    assert bounded_assumption_entails(condition, x > 0, policy=policy) is None


def test_sign_budget():
    x = sp.symbols("x", real=True)
    policy = SymbolicPolicy(assumption_ops=1)
    assert bounded_assumption_sign((x + 1) ** 5 + (x - 1) ** 5, policy=policy) is None


@pytest.mark.parametrize(
    "expression, expected",
    [
        (lambda x: 1 / x, None),
        (lambda x: 1 / x**2, sp.oo),
        (lambda x: (x**2 - 1) / (x - 1), 1),
    ],
)
def test_two_sided_rational_limits(expression, expected):
    x = sp.Symbol("x", real=True)
    assert bounded_limit(expression(x), x, 0, direction="+-") == expected


def test_limit_direction_validation():
    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="direction"):
        bounded_limit(1 / x, x, 0, direction="left")


@pytest.mark.parametrize("equation", [sp.S.Zero, sp.Eq(0, 0)])
def test_zero_equation_is_unresolved(equation):
    x = sp.Symbol("x")
    assert bounded_polynomial_roots(equation, x) is None
    assert bounded_solve_one(equation, x, allow_general=True) is None
    assert bounded_solve_system((equation,), (x,), allow_general=True) is None
    assert bounded_solve_one(sp.Eq(0, 1), x, allow_general=True) == ()
    assert bounded_solve_system((sp.Eq(0, 1),), (x,)) == ()
    assert bounded_polynomial_roots(sp.S.One, x) == ()


def test_expansion_budgets():
    x, y = sp.symbols("x y")
    expression = (x + y) ** 1000000 - 1
    assert bounded_polynomial_roots(expression, x) is None
    assert bounded_solve_one(expression, x, allow_general=True) is None
    assert bounded_solve_system((expression,), (x, y), allow_general=True) is None
    assert bounded_simplify(expression) == expression
    assert bounded_limit(expression, x, sp.oo) is None


def test_small_factored_equations():
    x, y = sp.symbols("x y")
    assert set(bounded_polynomial_roots((x - 1) * (x + 1), x)) == {-1, 1}
    assert bounded_solve_system((x + y - 3, x - y - 1), (x, y)) == ({x: 2, y: 1},)
