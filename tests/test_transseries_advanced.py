import sympy as sp

from asymptotic import (
    GrowthComparison,
    transseries_from_expression,
)
from asymptotic.exp_log_scale import (
    LogExpScale,
    compare_log_exp_scales,
)
from asymptotic.function_properties import (
    PropertyKnowledge,
    PropertyProvenance,
    PropertyRule,
    entails,
    function_property_rules,
)
from asymptotic.function_properties.query import analytic_at
from asymptotic.nonlinear_ode import nonlinear_differential_dominant_balances


def test_native_transseries_composition_reciprocal_and_calculus():
    x = sp.symbols("x", positive=True)
    z = sp.symbols("z")
    s = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)

    composed = s.compose(sp.exp(z), argument=z, terms=4)
    expected = sp.E * (1 + 1 / x + sp.Rational(1, 2) / x**2 + sp.Rational(1, 6) / x**3)
    assert sp.simplify(composed.truncate() - expected) == 0

    reciprocal = s.reciprocal(terms=4)
    assert sp.simplify(reciprocal.truncate() - (1 - 1 / x + 1 / x**2 - 1 / x**3)) == 0

    derivative = s.differentiate()
    assert sp.simplify(derivative.truncate() + 1 / x**2) == 0
    primitive = derivative.integrate(constant=3)
    assert sp.simplify(primitive.truncate() - (3 + 1 / x)) == 0


def test_function_property_entailment_provenance_and_analyticity():
    z = sp.symbols("z", real=True)
    assert entails(z > 0, z > 1) is True
    assert entails(z > 0, z < 0) is False

    provenance = PropertyProvenance("unit-test", "example")
    rule = PropertyRule(z > 0, "positive", PropertyKnowledge.SUFFICIENT, provenance)
    assert rule.provenance == provenance

    rules = function_property_rules(sp.log(z))
    assert any(r.value == "real" and r.knowledge is PropertyKnowledge.SUFFICIENT for r in rules)
    assert analytic_at(sp.log(z), z, 1) is True
    assert analytic_at(sp.log(z), z, 0) is False


def test_nested_logarithmico_exponential_scale_ordering():
    x = sp.symbols("x", positive=True)
    huge = LogExpScale.from_expr(sp.exp(sp.exp(x)), x)
    ordinary = LogExpScale.from_expr(sp.exp(x**100), x)
    assert huge.exponential_height == 2
    assert compare_log_exp_scales(huge, ordinary, x) is GrowthComparison.LARGER
    assert compare_log_exp_scales(sp.log(x), sp.log(sp.log(x)), x) is GrowthComparison.LARGER
    assert (
        compare_log_exp_scales(sp.exp(sp.sqrt(sp.log(x))), sp.log(x) ** 100, x)
        is GrowthComparison.LARGER
    )


def test_nonlinear_differential_power_balance_finite_and_infinity():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    # y' = y^2 admits y ~ -1/x near x=0 (and also as x->oo).
    equation = sp.diff(y(x), x) - y(x) ** 2
    local = nonlinear_differential_dominant_balances(equation, y, x)
    assert any(item.exponent == -1 and -1 in item.roots for item in local)

    infinite = nonlinear_differential_dominant_balances(equation, y, x, point=sp.oo)
    assert any(item.exponent == -1 and -1 in item.roots for item in infinite)


def test_nonlinear_second_order_balance_keeps_derivative_falling_factor():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    # y'' + y^2 = 0: alpha-2 = 2 alpha -> alpha=-2,
    # c*alpha(alpha-1)+c^2=0 gives c=-6.
    balances = nonlinear_differential_dominant_balances(sp.diff(y(x), x, 2) + y(x) ** 2, y, x)
    target = next(item for item in balances if item.exponent == -2)
    assert -6 in target.roots
