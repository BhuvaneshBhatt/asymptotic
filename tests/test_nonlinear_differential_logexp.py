import sympy as sp

from asymptotic.nonlinear_ode import nonlinear_differential_transseries


def test_resonant_logarithmic_leading_correction_is_discovered():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    equation = x * sp.diff(y(x), x) - y(x) - x

    branches = nonlinear_differential_transseries(
        equation, y, x, point=0, terms=4, stratify_parameters=False
    )
    branch = next(b for b in branches if sp.simplify(b.series - x * sp.log(x)) == 0)

    assert branch.complete is True
    assert branch.residual == 0
    assert branch.steps[0].correction_kind == "logarithmic"
    assert branch.steps[0].logarithmic_power == 1
    assert branch.steps[0].local_exponent == 1
    assert sp.simplify(branch.steps[0].coefficient - 1) == 0


def test_exponentially_small_mode_is_added_after_exact_algebraic_branch_at_infinity():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    equation = sp.diff(y(x), x) + y(x) - (1 / x - 1 / x**2)

    branches = nonlinear_differential_transseries(
        equation, y, x, point=sp.oo, terms=3, stratify_parameters=False
    )
    perturbed = next(b for b in branches if b.series.has(sp.exp(-x)))
    exp_step = next(step for step in perturbed.steps if step.correction_kind == "exponential")

    assert sp.simplify(perturbed.series - (1 / x + exp_step.free_parameter * sp.exp(-x))) == 0
    assert exp_step.free_parameter is not None
    assert exp_step.local_exponent is None
    assert perturbed.complete is True
    assert perturbed.residual == 0


def test_nonlinear_exponential_descendants_are_lifted_in_free_parameter():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    equation = sp.diff(y(x), x) - y(x) ** 2 + 1

    branches = nonlinear_differential_transseries(
        equation, y, x, point=sp.oo, terms=4, stratify_parameters=False
    )
    branch = next(
        b
        for b in branches
        if any(step.correction_kind == "exponential" for step in b.steps)
        and sp.simplify(b.series + 1) != 0
    )
    exp_steps = [step for step in branch.steps if step.correction_kind == "exponential"]
    C = exp_steps[0].free_parameter

    assert C is not None
    assert len(exp_steps) == 3
    expected = -1 + C * sp.exp(-2 * x) - C**2 * sp.exp(-4 * x) / 2 + C**3 * sp.exp(-6 * x) / 4
    assert sp.simplify(branch.series - expected) == 0
    assert branch.complete is False
    assert "exponentially small correction tower" in branch.limitation


def test_power_lifting_regressions_keep_power_step_metadata():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    forcing = sp.simplify(sp.diff(target, x) - target**2)
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing

    branches = nonlinear_differential_transseries(
        equation, y, x, point=0, terms=2, stratify_parameters=False
    )
    branch = next(b for b in branches if sp.simplify(b.series - target) == 0)

    assert [step.correction_kind for step in branch.steps] == ["power", "power"]
    assert [step.local_exponent for step in branch.steps] == [-1, 1]
