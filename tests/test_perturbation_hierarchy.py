import sympy as sp

from asymptotic.perturbation import (
    PerturbationHierarchy,
    PerturbationOrder,
    perturbation_hierarchy,
)


def test_scalar_algebraic_hierarchy_collects_nonlinear_orders():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")

    hierarchy = perturbation_hierarchy(u**2 - 1 - eps, u, eps, order=2)
    u0, u1, u2 = hierarchy.coefficient_unknowns[0]

    assert hierarchy.order(0).equations == (u0**2 - 1,)
    assert sp.expand(hierarchy.order(1).equations[0]) == 2 * u0 * u1 - 1
    assert sp.expand(hierarchy.order(2).equations[0]) == 2 * u0 * u2 + u1**2
    assert hierarchy.order(1).unknowns == (u1,)


def test_system_hierarchy_uses_one_common_order_model():
    eps = sp.Symbol("eps", positive=True)
    u, v = sp.symbols("u v")
    hierarchy = perturbation_hierarchy(
        (u + v - 1, u * v - eps),
        (u, v),
        eps,
        order=1,
    )
    u0, u1 = hierarchy.coefficient_unknowns[0]
    v0, v1 = hierarchy.coefficient_unknowns[1]

    assert hierarchy.order(0).equations == (u0 + v0 - 1, u0 * v0)
    assert hierarchy.order(1).equations == (u1 + v1, u0 * v1 + u1 * v0 - 1)
    assert hierarchy.order(1).unknowns == (u1, v1)


def test_ode_hierarchy_expands_derivatives_and_conditions():
    eps, x = sp.symbols("eps x", positive=True)
    y = sp.Function("y")
    hierarchy = perturbation_hierarchy(
        sp.diff(y(x), x) + y(x) + eps * y(x) ** 2,
        y(x),
        eps,
        order=1,
        conditions=sp.Eq(y(0), 1 + eps),
    )
    y0, y1 = hierarchy.coefficient_unknowns[0]

    assert sp.expand(hierarchy.order(0).equations[0]) == y0 + sp.diff(y0, x)
    assert sp.expand(hierarchy.order(1).equations[0]) == y0**2 + y1 + sp.diff(y1, x)
    assert hierarchy.order(0).conditions == (y0.subs(x, 0) - 1,)
    assert hierarchy.order(1).conditions == (y1.subs(x, 0) - 1,)


def test_fractional_power_gauge_sequence_is_supported():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    gauges = (1, sp.sqrt(eps), eps)
    hierarchy = perturbation_hierarchy(u**2 - eps, u, eps, gauges=gauges)
    u0, u1, u2 = hierarchy.coefficient_unknowns[0]

    assert hierarchy.gauges == gauges
    assert hierarchy.order(0).equations == (u0**2,)
    assert hierarchy.order(1).equations == (2 * u0 * u1,)
    assert hierarchy.order(2).equations == (2 * u0 * u2 + u1**2 - 1,)


def test_nonpower_gauges_use_recursive_asymptotic_coefficients():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    gauge = eps * sp.log(1 / eps)
    hierarchy = perturbation_hierarchy(u - 1 - gauge, u, eps, gauges=(1, gauge))
    u0, u1 = hierarchy.coefficient_unknowns[0]

    assert hierarchy.order(0).equations == (u0 - 1,)
    assert sp.simplify(hierarchy.order(1).equations[0] - (u1 - 1)) == 0


def test_manual_order_solutions_reconstruct_without_mutation():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    hierarchy = perturbation_hierarchy(u**2 - 1 - eps, u, eps, order=2)
    _, u1, _ = hierarchy.coefficient_unknowns[0]

    supplied = hierarchy.with_solution(0, 1).with_solution(1, {u1: sp.Rational(1, 2)})
    supplied = supplied.with_solution(2, -sp.Rational(1, 8))

    assert hierarchy.solved_through is None
    assert supplied.solved_through == 2
    assert supplied.approximation() == (1 + eps / 2 - eps**2 / 8,)
    residual = supplied.residual()[0]
    assert sp.expand(residual).coeff(eps, 1) == 0
    assert sp.expand(residual).coeff(eps, 2) == 0


def test_partial_solution_keeps_unsupplied_coefficient_symbolic():
    eps = sp.Symbol("eps", positive=True)
    u, v = sp.symbols("u v")
    hierarchy = perturbation_hierarchy((u - 1, v - 2), (u, v), eps, order=1)
    u0, _ = hierarchy.coefficient_unknowns[0]
    v0, _ = hierarchy.coefficient_unknowns[1]

    partial = hierarchy.with_solution(0, {u0: 1})

    assert partial.order(0).solved is False
    assert partial.approximation(0) == (1, v0)


def test_equalities_are_normalized_to_zero_residuals():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    hierarchy = perturbation_hierarchy(sp.Eq(u, 1 + eps), u, eps, order=1)
    u0, u1 = hierarchy.coefficient_unknowns[0]

    assert hierarchy.equations == (u - eps - 1,)
    assert hierarchy.order(0).equations == (u0 - 1,)
    assert hierarchy.order(1).equations == (u1 - 1,)


def test_invalid_gauge_order_is_rejected():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")

    try:
        perturbation_hierarchy(u, u, eps, gauges=(1, eps**2, eps))
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("unordered gauges were accepted")


def test_hierarchy_types_are_public():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    hierarchy = perturbation_hierarchy(u, u, eps, order=0)
    assert isinstance(hierarchy, PerturbationHierarchy)
    assert isinstance(hierarchy.order(0), PerturbationOrder)


def test_solvability_conditions_have_a_stable_place_in_order_schema():
    from asymptotic.perturbation import SolvabilityCondition

    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    hierarchy = perturbation_hierarchy(u - 1, u, eps, order=1)
    condition = sp.Symbol("resonant_forcing")

    updated = hierarchy.with_solvability_conditions(
        1, condition, reason="kernel projection"
    )

    assert hierarchy.order(1).solvability_conditions == ()
    assert updated.order(1).solvability_conditions == (
        SolvabilityCondition(condition, 1, "kernel projection"),
    )


def test_problem_assumptions_are_retained_for_later_solvers():
    eps = sp.Symbol("eps", positive=True)
    u, a = sp.symbols("u a")
    assumption = a > 0

    hierarchy = perturbation_hierarchy(u - a, u, eps, order=0, assumptions=assumption)

    assert hierarchy.assumptions == assumption


def test_negative_order_index_is_not_python_negative_indexing():
    eps = sp.Symbol("eps", positive=True)
    u = sp.Symbol("u")
    hierarchy = perturbation_hierarchy(u, u, eps, order=1)

    try:
        hierarchy.order(-1)
    except IndexError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("negative perturbation order was accepted")
