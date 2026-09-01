from __future__ import annotations

import sympy as sp

from asymptotic._symbolic_policy import (
    SymbolicPolicy,
    bounded_assumption_entails,
    bounded_assumption_sign,
    bounded_limit,
    bounded_polynomial_roots,
    bounded_primitive,
    bounded_solve_one,
    bounded_solve_system,
)


def test_bounded_solver_handles_affine_and_low_degree_polynomials_exactly():
    x = sp.symbols("x")
    assert bounded_solve_one(3 * x - 6, x) == (2,)
    assert set(bounded_solve_one(x**2 - 1, x) or ()) == {-1, 1}


def test_bounded_polynomial_solver_declines_degree_above_policy():
    x = sp.symbols("x")
    policy = SymbolicPolicy(polynomial_degree=3)
    assert bounded_polynomial_roots(x**4 - 1, x, policy=policy) is None


def test_bounded_linear_system_uses_exact_linear_algebra():
    x, y = sp.symbols("x y")
    result = bounded_solve_system((x + y - 3, x - y - 1), (x, y))
    assert result == ({x: 2, y: 1},)


def test_bounded_limit_handles_rational_endpoint():
    x = sp.symbols("x", positive=True)
    assert bounded_limit((2 * x + 1) / (x + 3), x, sp.oo) == 2
    assert bounded_limit(x**2 / (1 + x), x, 0) == 0


def test_bounded_primitive_declines_special_function_without_opt_in():
    x = sp.symbols("x")
    assert bounded_primitive(x**2, x) == x**3 / 3
    assert bounded_primitive(sp.exp(-(x**2)), x) is None


def test_general_solver_fallback_requires_explicit_opt_in():
    x = sp.symbols("x")
    equation = sp.exp(x) - 2
    assert bounded_solve_one(equation, x) is None
    assert bounded_solve_one(equation, x, allow_general=True) == (sp.log(2),)


def test_assumption_fallback_respects_complexity_budget():
    x = sp.symbols("x", real=True)
    condition = sp.And(*[sp.Symbol(f"p{i}") > 0 for i in range(12)])
    policy = SymbolicPolicy(assumption_ops=2, satisfiable_ops=1)
    assert bounded_assumption_entails(condition, x > 0, policy=policy) is None


def test_assumption_sign_declines_large_expression():
    x = sp.symbols("x", real=True)
    policy = SymbolicPolicy(assumption_ops=1)
    assert bounded_assumption_sign((x + 1) ** 5 + (x - 1) ** 5, policy=policy) is None
