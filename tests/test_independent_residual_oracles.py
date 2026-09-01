"""Returned branches are checked against original equations independently."""

import sympy as sp

from asymptotic import implicit_asymptotic
from asymptotic.nonlinear_ode import nonlinear_differential_transseries


def test_implicit_branch_substitution_reconstructs_defining_equation():
    x, y = sp.symbols("x y", positive=True)
    for branch in implicit_asymptotic(y**2 - x, y, x, terms=3):
        residual = sp.expand((y**2 - x).subs(y, branch.truncate()))
        assert residual == 0


def test_nonlinear_ode_branch_is_replayed_in_original_equation():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    forcing = sp.simplify(sp.diff(target, x) - target**2)
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing
    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=3)
    branch = next(item for item in branches if sp.simplify(item.series - target) == 0)
    residual = equation.xreplace({y(x): branch.series, sp.diff(y(x), x): sp.diff(branch.series, x)})
    assert sp.simplify(residual) == 0
