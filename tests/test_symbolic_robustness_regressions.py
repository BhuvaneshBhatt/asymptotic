import pytest
import sympy as sp

from asymptotic import (
    AsymptoticAlgebra,
    AsymptoticContext,
    RemainderKind,
    asymptotic_element,
    puiseux_series,
)
from asymptotic._symbolic_primitives import certification_primitive
from asymptotic.function_properties.semantics import entails
from asymptotic.nonlinear_ode import nonlinear_differential_transseries
from asymptotic.parameter_auto import specialize_expression


def test_certification_primitive_handles_exponential_derivative_without_general_integration():
    h = sp.symbols("h", positive=True)
    expr = sp.exp(-2 / h) / h**2
    primitive = certification_primitive(expr, h)
    assert primitive is not None
    assert sp.simplify(sp.diff(primitive, h) - expr) == 0


def test_certification_primitive_refuses_special_function_search():
    x = sp.symbols("x", positive=True)
    assert certification_primitive(sp.exp(x) / x, x) is None


def test_eventual_sign_rational_fast_path_at_zero_finite_point_and_infinity():
    h = sp.symbols("h", positive=True)
    assert AsymptoticContext(h, point=0).eventual_sign(-(h**2)) == -1
    assert AsymptoticContext(h, point=0).eventual_sign(1 / h) == 1

    x = sp.symbols("x", positive=True)
    assert AsymptoticContext(x, point=2).eventual_sign(-((x - 2) ** 2)) == -1
    assert AsymptoticContext(x, point=sp.oo).eventual_sign(-1 / x**2) == -1


def test_analytic_unit_phase_is_not_misclassified_as_beyond_all_orders():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    forcing = sp.simplify(sp.diff(target, x) - target**2)
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing

    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=4)
    exact = next(branch for branch in branches if sp.simplify(branch.series - target) == 0)
    assert exact.complete is True
    assert all(step.correction_kind != "exponential" for step in exact.steps)


def test_structural_entailment_avoids_solver_for_explicit_literals():
    a, b = sp.symbols("a b", real=True)
    assumptions = sp.And(sp.Eq(a, 0), sp.Ne(b, 0))
    assert entails(sp.Eq(a, 0), assumptions) is True
    assert entails(sp.Ne(a, 0), assumptions) is False
    assert entails(sp.Ne(b, 0), assumptions) is True


def test_affine_parameter_specialization_does_not_need_general_solve():
    a, b = sp.symbols("a b")
    expr = a**2 + b
    specialized = specialize_expression(expr, sp.Eq(2 * a + b, 0), parameters=(a, b))
    assert sp.simplify(specialized - (b**2 / 4 + b)) == 0


def test_periodic_negative_power_crossing_zero_does_not_claim_finite_bounds():
    from asymptotic.periodic import periodic_bounds

    x = sp.symbols("x", real=True)
    bounds = periodic_bounds(sp.sin(x) ** -2, x)
    assert bounds == (None, None, False)


def test_differential_balance_uses_numeric_valuation_order():
    from asymptotic.nonlinear_ode import nonlinear_differential_dominant_balances

    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    equation = x**10 * y(x) ** 2 + x**2 * y(x) + x**3
    balances = nonlinear_differential_dominant_balances(equation, y, x, point=0)
    assert balances
    # The active lower envelope must be chosen by rational valuation, not by
    # structural sorting of the values 10+2a, 2+a, and 3.
    assert any(
        balance.valuation == min(term.valuation_at(balance.exponent) for term in balance.terms)
        for balance in balances
    )


def test_eventual_sign_respects_finite_left_hand_direction():
    x = sp.symbols("x", real=True)
    right = AsymptoticContext(x, point=0, direction="+")
    left = AsymptoticContext(x, point=0, direction="-")

    assert right.eventual_sign(x) == 1
    assert left.eventual_sign(x) == -1
    assert left.eventual_sign(-x) == 1


def test_native_dispatch_does_not_swallow_internal_type_error():
    x = sp.symbols("x")

    class BrokenNative:
        variable = x
        point = sp.oo

        def truncate(self):
            return 1 / x

        def differentiate(self, order=1):
            raise TypeError("native implementation bug")

    element = asymptotic_element(BrokenNative())
    with pytest.raises(TypeError, match="native implementation bug"):
        element.differentiate()


def test_unknown_native_without_remainder_contract_stays_unknown():
    x = sp.symbols("x")

    class PrefixOnly:
        variable = x
        point = sp.oo

        def truncate(self):
            return 1 + 1 / x

    element = asymptotic_element(PrefixOnly())
    assert element.remainder.kind is RemainderKind.UNKNOWN


def test_finite_puiseux_prefix_is_not_mistaken_for_exact_expression():
    x = sp.symbols("x", positive=True)
    series = puiseux_series(sp.exp(x), x, point=0, terms=3)
    element = series.asymptotic_element()

    assert element.remainder.kind is RemainderKind.UNKNOWN
    assert element.remainder.exact_expression != 0


def test_algebra_rebinds_existing_element_to_its_context():
    x = sp.symbols("x", positive=True)
    original = asymptotic_element(1 + 1 / x, x)
    context = AsymptoticContext(x, sp.oo, zero_confidence="probable")
    algebra = AsymptoticAlgebra(x, sp.oo, context=context)

    rebound = algebra.element(original)
    assert rebound.context is context


def test_context_rejects_invalid_coordinate_configuration():
    x = sp.symbols("x")
    with pytest.raises(ValueError, match="direction"):
        AsymptoticContext(x, point=0, direction="sideways")
    with pytest.raises(ValueError, match="zero_confidence"):
        AsymptoticContext(x, zero_confidence="guess")
    with pytest.raises(TypeError, match="Symbol"):
        AsymptoticContext(x + 1)


def test_algebra_rejects_mismatched_injected_context():
    x, y = sp.symbols("x y")
    wrong_var = AsymptoticContext(y, sp.oo)
    wrong_point = AsymptoticContext(x, 0)

    with pytest.raises(ValueError, match="context uses different coordinates"):
        AsymptoticAlgebra(x, sp.oo, context=wrong_var)
    with pytest.raises(ValueError, match="context uses different coordinates"):
        AsymptoticAlgebra(x, sp.oo, context=wrong_point)


def test_asymptotic_element_rebind_validates_context_and_coordinates():
    x, y = sp.symbols("x y")
    element = asymptotic_element(1 / x, x)
    probable = AsymptoticContext(x, sp.oo, zero_confidence="probable")

    rebound = asymptotic_element(element, context=probable)
    assert rebound.context is probable
    with pytest.raises(ValueError, match="different coordinates"):
        asymptotic_element(element, y)
    with pytest.raises(ValueError, match="context uses different coordinates"):
        asymptotic_element(element, context=AsymptoticContext(y, sp.oo))


def test_native_dispatch_rejects_unadaptable_result():
    x = sp.symbols("x")

    class BrokenNative:
        variable = x
        point = sp.oo

        def truncate(self):
            return 1 / x

        def differentiate(self, order=1):
            return [order]

    with pytest.raises(TypeError, match="returned unsupported list"):
        asymptotic_element(BrokenNative()).differentiate()


def test_native_objects_reject_mismatched_injected_contexts():
    from asymptotic import AsymptoticScale, Multiseries, NestedExpansion
    from asymptotic.scale import ScaleElement

    x, y = sp.symbols("x y", positive=True)
    wrong = AsymptoticContext(y, sp.oo)
    scale = AsymptoticScale(x, (ScaleElement(1 / x),), point=sp.oo)

    with pytest.raises(ValueError, match="different coordinates"):
        Multiseries(1 + 1 / x, scale, context=wrong)
    with pytest.raises(ValueError, match="different coordinates"):
        NestedExpansion(1 + 1 / x, x, context=wrong)


def test_series_reversion_rejects_context_for_wrong_endpoint():
    from asymptotic import series_reversion

    x, y = sp.symbols("x y")
    with pytest.raises(ValueError, match="different coordinates"):
        series_reversion(x + x**2, x, y, point=0, context=AsymptoticContext(x, sp.oo))


def test_context_normalization_does_not_force_parameter_branch_identities():
    x = sp.symbols("x", positive=True)
    a, b = sp.symbols("a b")
    expr = sp.sqrt(a) * sp.sqrt(b)

    normalized = AsymptoticContext(x).normalize(expr)
    assert normalized == expr
    assert normalized != sp.powsimp(expr, force=True)


def test_remainder_exact_error_preserves_principal_power_branches():
    from asymptotic import AsymptoticRemainder

    x = sp.symbols("x", positive=True)
    a, b = sp.symbols("a b")
    branch_error = sp.sqrt(a) * sp.sqrt(b) - sp.sqrt(a * b)
    remainder = AsymptoticRemainder.unknown(
        x,
        sp.oo,
        exact_expression=branch_error,
        source="branch-sensitive regression",
    )
    assert remainder.exact_expression != 0
    assert remainder.exact_expression == sp.powsimp(branch_error, force=False)


def test_periodic_decomposition_does_not_power_expand_parameter_factors():
    from asymptotic.periodic import periodic_decomposition

    x = sp.symbols("x", real=True)
    a, b = sp.symbols("a b")
    expr = sp.sqrt(a) * sp.sqrt(b) * sp.sin(x)
    dec = periodic_decomposition(expr, x)
    assert dec.reconstruct() == sp.powsimp(expr, force=False)
    assert dec.reconstruct() != sp.sqrt(a * b) * sp.sin(x)


def test_ramification_mapping_only_expands_proved_positive_uniformizer():
    from asymptotic.monomial import RamificationModel

    x = sp.symbols("x", positive=True)
    a, b = sp.symbols("a b")
    model = RamificationModel(x, sp.oo, 2)
    mapped = model.to_parameter(sp.log(a * b / x))
    t = model.parameter
    assert mapped == 2 * sp.log(t) + sp.log(a * b)
    assert mapped != sp.log(a) + sp.log(b) + 2 * sp.log(t)
