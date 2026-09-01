from __future__ import annotations

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from asymptotic.discrete_scale import (
    DiscreteAsymptoticScale,
    birkhoff_trjitzinsky_branches,
    discrete_newton_edges,
    linear_recurrence_data,
)
from asymptotic.mrv import mrv_decomposition
from asymptotic.rsolve import asymptotic_rsolve


def test_factorial_scale_has_exact_shift_ratio():
    n = sp.symbols("n", positive=True, integer=True)
    scale = DiscreteAsymptoticScale(n, factorial_power=1, power=-1)
    assert sp.simplify(scale.expression - sp.gamma(n)) == 0
    assert sp.simplify(scale.ratio(1) - n) == 0


def test_discrete_newton_edge_finds_factorial_balance():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    data = linear_recurrence_data(a(n + 1) - n * a(n), a(n), n)
    edges = discrete_newton_edges(data)
    assert len(edges) == 1
    assert edges[0].factorial_power == 1
    assert sp.expand(edges[0].characteristic).subs(edges[0].characteristic_symbol, 1) == 0


def test_native_birkhoff_trjitzinsky_lifts_factorial_recurrence():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    result = asymptotic_rsolve(a(n + 1) - n * a(n), a(n), n, terms=4, method="native")
    assert result.method == "discrete-newton-birkhoff-trjitzinsky"
    assert len(result.branches) == 1
    branch = result.branches[0]
    assert branch.scale.factorial_power == 1
    assert branch.scale.power == -1
    assert branch.replay_characteristic() is True
    assert sp.simplify(branch.expression - sp.gamma(n)) == 0


def test_native_newton_polygon_can_produce_two_factorial_scales():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 2) - (n + 1) * a(n + 1) - a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=4)
    assert {branch.scale.factorial_power for branch in branches} == {-1, 1}
    assert all(branch.replay_characteristic() is True for branch in branches)


def test_mrv_recognizes_factorial_growth_as_a_scale_generator():
    n = sp.symbols("n", positive=True)
    result = mrv_decomposition(sp.factorial(n) + sp.exp(n), n)
    members = tuple(member for cls in result.classes for member in cls.members)
    assert sp.factorial(n) in members
    # n! grows faster than exp(n) on the logarithmic variation scale.
    assert result.representative == sp.factorial(n)


def _branch_signature(branch):
    return (
        sp.simplify(branch.scale.factorial_power),
        sp.simplify(branch.scale.exponential_base),
        sp.simplify(branch.scale.power),
    )


@given(
    factorial_power=st.integers(min_value=-2, max_value=2),
    power=st.integers(min_value=-2, max_value=2),
    numerator=st.integers(min_value=-3, max_value=3).filter(lambda value: value != 0),
    denominator=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_generated_first_order_recurrences_recover_discrete_scale(
    factorial_power, power, numerator, denominator
):
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    base = sp.Rational(numerator, denominator)
    ratio = base * (n + 1) ** factorial_power * ((n + 1) / n) ** power
    recurrence = a(n + 1) - ratio * a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=3)
    signatures = {_branch_signature(branch) for branch in branches}
    assert (sp.Integer(factorial_power), base, sp.Integer(power)) in signatures
    matching = next(
        branch
        for branch in branches
        if _branch_signature(branch) == (sp.Integer(factorial_power), base, sp.Integer(power))
    )
    assert matching.replay_residual(data) is True


def test_recurrence_normalization_metamorphics_preserve_native_scale():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 1) - 2 * n * a(n)
    variants = (
        recurrence,
        sp.Eq(a(n + 1), 2 * n * a(n)),
        3 * recurrence,
        recurrence / (n + 2),
        recurrence.subs(n, n + 3),
        sp.Add(*reversed(sp.Add.make_args(recurrence))),
    )
    signatures = []
    for variant in variants:
        data = linear_recurrence_data(variant, a(n), n)
        branches = birkhoff_trjitzinsky_branches(data, terms=3)
        signatures.append({_branch_signature(branch) for branch in branches})
        assert all(branch.replay_residual(data) is True for branch in branches)
    assert all(signature == signatures[0] for signature in signatures[1:])


def test_measured_residual_order_is_replayable_and_not_requested_term_count():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 2) - (n + 1) * a(n + 1) - a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=4)
    assert branches
    assert all(branch.residual_order == 5 for branch in branches)
    assert all(branch.residual_order != 4 for branch in branches)
    assert all(branch.replay_residual(data) is True for branch in branches)


def test_characteristic_root_multiplicity_handles_sympy_integer_degree():
    from asymptotic.discrete_scale import _root_multiplicity

    z = sp.symbols("z")
    assert _root_multiplicity((z - 1) ** 3 * (z + 2), z, sp.S.One) == 3


def test_mrv_unknown_comparisons_do_not_choose_a_false_maximum():
    from asymptotic.context import AsymptoticContext, GrowthComparison

    class UnknownGrowthContext(AsymptoticContext):
        def compare_growth(self, f, g):
            return GrowthComparison.UNKNOWN, None

    n = sp.symbols("n", positive=True)
    context = UnknownGrowthContext(n, sp.oo)
    result = mrv_decomposition(sp.exp(n) + sp.exp(n**2), n, context=context)
    assert len(result.classes) > 1
    assert result.most_rapid is None
    assert result.representative is None


def test_gamma_stirling_measure_requires_positive_unbounded_argument():
    from asymptotic.context import AsymptoticContext
    from asymptotic.mrv import _variation_measure

    n = sp.symbols("n", positive=True)
    context = AsymptoticContext(n, sp.oo)
    positive = _variation_measure(sp.gamma(n), context)
    pole_crossing = _variation_measure(sp.gamma(-n), context)
    assert positive != sp.log(1 + sp.Abs(sp.gamma(n)))
    assert pole_crossing == sp.log(1 + sp.Abs(sp.gamma(-n)))


def test_residual_replay_rejects_tampered_order():
    from dataclasses import replace

    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 2) - (n + 1) * a(n + 1) - a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branch = birkhoff_trjitzinsky_branches(data, terms=4)[0]
    assert branch.residual_order is not None
    tampered = replace(branch, residual_order=branch.residual_order + 1)
    assert tampered.replay_residual(data) is False


def test_normalized_recurrence_caches_polynomial_metadata():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    data = linear_recurrence_data(
        (n + 1) * a(n + 2) - (2 * n + 1) * a(n + 1) + n * a(n),
        a(n),
        n,
    )
    assert len(data.polynomial_terms) == len(data.coefficients)
    for cached, (shift, expression) in zip(data.polynomial_terms, data.coefficients):
        assert cached.shift == shift
        assert cached.expression == expression
        assert cached.polynomial.as_expr() == expression
        assert cached.degree == int(cached.polynomial.degree())
        assert cached.leading_coefficient == cached.polynomial.LC()


def test_simple_bt_lift_uses_linear_coefficient_extraction():
    import inspect

    import asymptotic.discrete_scale

    source = inspect.getsource(asymptotic.discrete_scale._solve_lift_equations)
    assert "sp.solve(" not in source
    linear_source = inspect.getsource(asymptotic.discrete_scale._linear_equation_solution)
    assert "sp.Poly" in linear_source
    assert "degree() != 1" in linear_source


def test_constant_coefficient_repeated_root_returns_exact_jordan_chain():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 2) - 2 * a(n + 1) + a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=4)
    assert len(branches) == 2
    assert {sp.expand(branch.expression) for branch in branches} == {sp.S.One, n}
    assert all(branch.characteristic_mult == 2 for branch in branches)
    assert all(branch.residual_order is sp.oo for branch in branches)
    assert all(branch.replay_residual(data) is True for branch in branches)


def test_constant_coefficient_triple_root_returns_three_polynomial_modes():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = a(n + 3) - 3 * a(n + 2) + 3 * a(n + 1) - a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=4)
    assert len(branches) == 3
    assert {sp.expand(branch.expression) for branch in branches} == {
        sp.S.One,
        n,
        n**2,
    }
    assert all(branch.residual_order is sp.oo for branch in branches)


def test_secondary_newton_lift_finds_stretched_exponentials_and_half_lattice():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    recurrence = n * a(n + 2) - 2 * n * a(n + 1) + (n - 1) * a(n)
    data = linear_recurrence_data(recurrence, a(n), n)
    branches = birkhoff_trjitzinsky_branches(data, terms=3)
    assert len(branches) == 2
    assert {branch.scale.phase for branch in branches} == {
        -2 * sp.sqrt(n),
        2 * sp.sqrt(n),
    }
    assert all(branch.scale.power == -sp.Rational(1, 4) for branch in branches)
    assert all(branch.lattice_step == sp.Rational(1, 2) for branch in branches)
    assert {branch.coefficients[1] for branch in branches} == {
        -sp.Rational(65, 48),
        sp.Rational(65, 48),
    }
    assert all(branch.residual_order == 3 for branch in branches)
    assert all(branch.replay_residual(data) is True for branch in branches)


def test_asymptotic_rsolve_uses_repeated_root_native_modes():
    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    result = asymptotic_rsolve(
        a(n + 2) - 2 * a(n + 1) + a(n),
        a(n),
        n,
        method="native",
        terms=4,
    )
    assert len(result.branches) == 2
    assert result.limitation is None
    assert {sp.expand(branch.expression) for branch in result.branches} == {sp.S.One, n}


def test_bt_internal_symbols_are_reused_across_repeated_lifts():
    import asymptotic.discrete_scale

    n = sp.symbols("n", positive=True, integer=True)
    a = sp.Function("a")
    first_ids = (
        id(asymptotic.discrete_scale._EDGE_LAMBDA),
        id(asymptotic.discrete_scale._BT_T),
        id(asymptotic.discrete_scale._BT_THETA),
        tuple(map(id, asymptotic.discrete_scale._BT_COEFFICIENTS[:4])),
    )
    for base in (2, 3, sp.Rational(1, 2), -2, -3):
        recurrence = a(n + 1) - base * n * a(n)
        data = linear_recurrence_data(recurrence, a(n), n)
        branches = birkhoff_trjitzinsky_branches(data, terms=4)
        assert branches
    second_ids = (
        id(asymptotic.discrete_scale._EDGE_LAMBDA),
        id(asymptotic.discrete_scale._BT_T),
        id(asymptotic.discrete_scale._BT_THETA),
        tuple(map(id, asymptotic.discrete_scale._BT_COEFFICIENTS[:4])),
    )
    assert second_ids == first_ids
