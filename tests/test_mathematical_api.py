from dataclasses import replace
from inspect import signature

import pytest
import sympy as sp

import asymptotic as api
from asymptotic.conditional import ConditionalExpression
from asymptotic.mathematical_api import _branches, _expression, _optimization
from asymptotic.optimization import OptimizationResult
from asymptotic.probability import StatisticalResult
from asymptotic.solve import AsymptoticSolutionBranch, SolveResult


def test_sum_prefix():
    n, k = sp.symbols("n k", positive=True, integer=True)
    arguments = (k**-2, k, n, sp.oo)
    keywords = dict(parameter=n, terms=3, method="euler-maclaurin")
    expected = 1 / n + 1 / (2 * n**2) + 1 / (6 * n**3)
    assert api.sum(*arguments, **keywords) == expected
    record = api.sum(*arguments, **keywords, return_result=True)
    assert record.expression == expected
    assert sp.limit(n**5 * record.remainder.scale, n, sp.oo) == sp.Rational(1, 30)
    assert record.certificate.replay() is True
    assert replace(record.certificate, expression=expected + 1 / n**4).replay() is False


@pytest.mark.parametrize("terms", [1, 2, 3, 4, 5])
def test_sum_error_bound(terms):
    n, k = sp.symbols("n k", positive=True, integer=True)
    record = api.sum(
        k**-2,
        k,
        n,
        sp.oo,
        parameter=n,
        terms=terms,
        method="euler-maclaurin",
        return_result=True,
    )
    # The trigamma tail supplies an independent exact value for the sum.
    for index in (10, 20):
        error = abs(
            (sp.polygamma(1, index) - record.expression.subs(n, index)).evalf(60)
        )
        bound = record.remainder.scale.subs(n, index).evalf(60)
        assert error <= bound


def test_exact_euler_maclaurin():
    n, k = sp.symbols("n k", positive=True, integer=True)
    record = api.sum(
        k**2,
        k,
        1,
        n,
        parameter=n,
        terms=4,
        method="euler-maclaurin",
        return_result=True,
    )
    assert sp.expand(record.expression) == n**3 / 3 + n**2 / 2 + n / 6
    assert record.remainder.is_exact
    assert record.certificate.replay() is True


@pytest.mark.parametrize("name", ["series", "multiseries", "puiseux_series"])
def test_series_prefix(name):
    x = sp.symbols("x", positive=True)
    function = getattr(api, name)
    options = {"depth": 2} if name == "nested_series" else {"terms": 2}
    record = function(1 + x**-1, x, point=sp.oo, return_result=True, **options)
    assert function(1 + x**-1, x, point=sp.oo, **options) == record.truncate()
    assert signature(function).parameters["return_result"].default is False


def test_solutions():
    x, n = sp.symbols("x n", positive=True)
    assert api.solve(x - n, x, parameter=n) == ({x: n},)
    assert api.root(x - n, x, parameter=n, branch=0) == n
    k = sp.symbols("k", integer=True)
    u = sp.Function("u")
    assert api.rsolve(u(k + 1) - u(k), u(k), k, initial_conditions={u(0): 2}) == 2


def test_conditional_values():
    x, n, a = sp.symbols("x n a", real=True)
    condition = a > 0
    record = StatisticalResult(a, n, sp.oo, "exact", "EXACT", conditions=(condition,))
    assert _expression(record) == ConditionalExpression(a, condition)
    optimum = OptimizationResult(
        a, (x,), x, n, sp.oo, "min", "EXACT", "exact", (condition,)
    )
    assert _optimization(optimum) == ConditionalExpression(a, condition)
    solution = SolveResult(
        (AsymptoticSolutionBranch(((x, a),), (condition,), "EXACT"),),
        n,
        sp.oo,
        "EXACT",
        "exact",
    )
    assert _branches(solution) == ({x: ConditionalExpression(a, condition)},)


def test_statistics_and_optimization():
    from sympy.stats import Normal

    n = sp.symbols("n", positive=True)
    x = sp.symbols("x", real=True)
    random = Normal("X", n, 1)
    assert api.expectation(random, parameter=n) == n
    assert api.minimize((x - n) ** 2, x, parameter=n) == 0
    assert api.argmax(-((x - n) ** 2), x, parameter=n) == (n,)


def test_leading_terms():
    x = sp.symbols("x", positive=True)
    expansion = api.puiseux_series(x + x**2, x, terms=3, return_result=True)
    assert api.leading_term(expansion) == x
    expansion = api.series(
        1 / x + 1 / x**2, x, method="transseries", return_result=True
    )
    assert api.leading_term(expansion) == 1 / x


def test_unresolved_and_nested_results():
    from asymptotic.nested import NestedExpansion

    x = sp.symbols("x", positive=True)
    n, k = sp.symbols("n k", positive=True, integer=True)
    result = api.sum(sp.Function("f")(k), k, n, sp.oo, parameter=n, method="exact")
    assert result.status == "UNKNOWN"
    nested = api.nested_series(sp.exp(sp.exp(x)), x, depth=1)
    assert isinstance(nested, NestedExpansion)
    assert nested.forms[0].reconstruct() == sp.exp(sp.exp(x))


def test_local_prefix_and_calculus():
    x, y = sp.symbols("x y", real=True)
    assert api.local_series(sp.sin(x), x, depth=4) == x - x**3 / 6
    prefix = api.series(sp.exp(x), x, point=0, terms=3)
    assert api.differentiate(prefix) == 1 + x
    assert api.differentiate(x * y, variable=x) == y
    assert api.differentiate(sp.Integer(3)) == 0
    with pytest.raises(ValueError, match="variable is required"):
        api.differentiate(x * y)
    assert api.truncate(prefix) == prefix
    with pytest.raises(ValueError, match="return_result=True"):
        api.truncate(prefix, 1)
    assert api.integrate(x, x, point=0) == x**2 / 2


def test_products_and_inverse_branches():
    n, k = sp.symbols("n k", positive=True, integer=True)
    x, y = sp.symbols("x y", positive=True)
    assert api.product((k + 1) / k, k, 1, n, parameter=n) == n + 1
    assert api.implicit(y - x - x**2, y, x, terms=3) == (x + x**2,)
    assert api.inverse(x + x**2, x, y, point=0, terms=3) == y - y**2 + 2 * y**3


def test_regular_approximations():
    eps = sp.symbols("eps", positive=True)
    u = sp.Symbol("u")
    assert set(api.regular_perturbation(u**2 - 1 - eps, u, eps, order=2)) == {
        (-1 - eps / 2 + eps**2 / 8,),
        (1 + eps / 2 - eps**2 / 8,),
    }


def test_differential_solutions():
    x = sp.symbols("x", positive=True)
    f = sp.Function("f")
    target = 2 / x + 3 / x**2
    forcing = sp.diff(target, x) - target**2
    solutions = api.dsolve(
        sp.diff(f(x), x) - f(x) ** 2 - forcing, f, x, terms=4, method="nonlinear"
    )
    assert isinstance(solutions, tuple)
    assert all(isinstance(solution, sp.Expr) for solution in solutions)
    assert any(sp.simplify(solution - target) == 0 for solution in solutions)


def test_integer_index_normalization():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", real=True)
    assert (
        api.sum(sp.cos(2 * sp.pi * k), k, 1, n, parameter=n, method="euler-maclaurin")
        == n
    )
    record = api.sum(
        sp.cos(2 * sp.pi * k),
        k,
        1,
        n,
        parameter=n,
        method="euler-maclaurin",
        return_result=True,
    )
    assert record.remainder.is_exact
    assert record.certificate.replay() is True


def test_uncertified_euler_maclaurin():
    n = sp.symbols("n", positive=True, integer=True)
    k = sp.symbols("k", real=True)
    record = api.sum(
        sp.cos(k), k, 1, n, parameter=n, method="euler-maclaurin", return_result=True
    )
    assert record.status == "FORMAL"
    assert not record.remainder.is_certified
    assert record.certificate.replay() is False
    assert (
        api.sum(sp.cos(k), k, 1, n, parameter=n, method="euler-maclaurin").status
        == "FORMAL"
    )


@pytest.mark.parametrize("power", [sp.Rational(3, 2), 3, 4])
def test_shifted_power_tail_bound(power):
    n, k = sp.symbols("n k", positive=True, integer=True)
    record = api.sum(
        2 / (k + sp.Rational(1, 2)) ** power,
        k,
        n,
        sp.oo,
        parameter=n,
        terms=3,
        method="euler-maclaurin",
        return_result=True,
    )
    assert record.status == "CERTIFIED"
    assert record.certificate.replay() is True
    exact = 2 * sp.zeta(power, n + sp.Rational(1, 2))
    for index in (10, 20):
        error = abs((exact - record.expression).subs(n, index).evalf(60))
        assert error <= record.remainder.scale.subs(n, index).evalf(60)
