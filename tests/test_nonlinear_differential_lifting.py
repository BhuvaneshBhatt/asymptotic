import sympy as sp

from asymptotic.nonlinear_ode import nonlinear_differential_transseries


def _riccati_equation_for_exact_solution(target, y, x):
    forcing = sp.simplify(sp.diff(target, x) - target**2)
    return sp.diff(y(x), x) - y(x) ** 2 - forcing


def test_recursive_nonlinear_differential_lifting_exact_branch_at_zero():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    equation = _riccati_equation_for_exact_solution(target, y, x)

    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=4)
    branch = next(b for b in branches if sp.simplify(b.series - target) == 0)

    assert branch.complete is True
    assert branch.residual == 0
    assert sp.simplify(branch.transseries.truncate() - branch.local_series) == 0
    assert tuple(step.local_exponent for step in branch.steps) == (-1, 1)
    assert tuple(step.coefficient for step in branch.steps) == (1, 1)
    assert branch.steps[1].residual_order_before == 0
    assert branch.steps[1].residual_order_after is None


def test_recursive_lifting_uses_exact_reciprocal_change_at_infinity():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 2 / x + 3 / x**2
    equation = _riccati_equation_for_exact_solution(target, y, x)

    branches = nonlinear_differential_transseries(equation, y, x, point=sp.oo, terms=4)
    branch = next(b for b in branches if sp.simplify(b.series - target) == 0)

    assert branch.complete is True
    assert branch.residual == 0
    assert tuple(step.local_exponent for step in branch.steps) == (1, 2)
    assert tuple(sp.simplify(step.term) for step in branch.steps) == (
        2 / x,
        3 / x**2,
    )
    assert sp.simplify(branch.local_coordinate - 1 / x) == 0


def test_recursive_lifting_residual_valuation_strictly_improves():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    equation = _riccati_equation_for_exact_solution(target, y, x)

    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=4)
    nonexact = next(b for b in branches if sp.simplify(b.series - target) != 0)

    certified = [
        (step.residual_order_before, step.residual_order_after)
        for step in nonexact.steps[1:]
        if step.residual_order_before is not None and step.residual_order_after is not None
    ]
    assert certified
    assert all(after > before for before, after in certified)


def test_recursive_lifting_finite_translated_point():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    h = x - 2
    target = 1 / h + h
    equation = _riccati_equation_for_exact_solution(target, y, x)

    branches = nonlinear_differential_transseries(equation, y, x, point=2, terms=4)
    branch = next(b for b in branches if sp.simplify(b.series - target) == 0)

    assert branch.complete is True
    assert sp.simplify(branch.local_coordinate - h) == 0
    assert tuple(step.local_exponent for step in branch.steps) == (-1, 1)


def test_recursive_lifting_reports_truncation_without_claiming_completion():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    equation = _riccati_equation_for_exact_solution(target, y, x)

    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=1)

    assert branches
    assert all(branch.complete is False for branch in branches)
    assert all(branch.limitation == "term/depth limit reached" for branch in branches)
