import sympy as sp

from asymptotic import (
    AsymptoticElement,
    GrowthComparison,
    asymptotic_element,
    differentiate,
    discover_scale,
    multiseries,
    nested_expansion,
    transseries_from_expression,
)
from asymptotic.algebra import AsymptoticFieldElementProtocol
from asymptotic.asymptotic_field import asymptotic_differential_field
from asymptotic.nonlinear_ode import nonlinear_differential_transseries
from asymptotic.remainder import RemainderKind


def test_common_protocol_adapts_primary_representations_without_erasing_native_type():
    x = sp.symbols("x", positive=True)
    trans = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)
    multi = multiseries(sp.exp(1 / x), x, terms=4)
    nested = nested_expansion(sp.log(x) + 1 / x, x, depth=1)

    for native in (trans, multi, nested):
        element = asymptotic_element(native)
        assert isinstance(element, AsymptoticElement)
        assert isinstance(element, AsymptoticFieldElementProtocol)
        assert element.native is native
        assert element.variable == x
        assert element.point == sp.oo


def test_multiseries_truncation_propagates_first_omitted_term_remainder():
    x = sp.symbols("x", positive=True)
    series = multiseries(sp.exp(1 / x), x, terms=5)
    trunc = series.asymptotic_element().truncation(3)

    assert sp.simplify(trunc.prefix - (1 + 1 / x + 1 / (2 * x**2))) == 0
    assert trunc.remainder.kind is RemainderKind.BIG_O
    assert sp.simplify(trunc.remainder.scale - 1 / (6 * x**3)) == 0
    assert trunc.remainder.check() is True


def test_nested_additive_truncation_uses_transseries_view_not_nested_depth_semantics():
    x = sp.symbols("x", positive=True)
    nested = nested_expansion(sp.log(x) + 1 / x, x, depth=1)
    element = nested.asymptotic_element()

    assert sp.simplify(element.to_transseries(3).truncate() - (sp.log(x) + 1 / x)) == 0
    assert sp.simplify(element.truncate(3) - (sp.log(x) + 1 / x)) == 0


def test_cross_representation_algebra_routes_through_certified_transseries_arithmetic():
    x = sp.symbols("x", positive=True)
    trans = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)
    multi = multiseries(1 / x + 1 / x**2, x, terms=3)

    result = asymptotic_element(trans) * asymptotic_element(multi)
    assert isinstance(result, AsymptoticElement)
    assert sp.simplify(result.truncate() - (1 / x + 2 / x**2 + 1 / x**3)) == 0


def test_protocol_composition_reciprocal_comparison_and_calculus():
    x, z = sp.symbols("x z", positive=True)
    trans = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)
    element = trans.asymptotic_element()

    composed = element.compose(sp.exp(z), argument=z, terms=3)
    expected = sp.E * (1 + 1 / x + 1 / (2 * x**2))
    assert sp.simplify(composed.truncate() - expected) == 0

    reciprocal = element.reciprocal(terms=3)
    assert sp.simplify(reciprocal.truncate() - (1 - 1 / x + 1 / x**2)) == 0

    small = asymptotic_element(1 / x, x, point=sp.oo)
    assert element.compare(small) is GrowthComparison.LARGER

    differentiated = element.differentiate()
    assert sp.simplify(differentiated.as_expr() + 1 / x**2) == 0
    # The backwards-compatible top-level calculus API still returns the native type.
    assert type(differentiate(trans)) is type(trans)


def test_scale_and_shadow_field_elements_join_common_protocol():
    x = sp.symbols("x", positive=True)
    scale = discover_scale(1 / x + sp.exp(-x), x)
    slow = scale.element(0)
    fast = scale.element(1)
    assert slow.compare(fast) is GrowthComparison.LARGER

    field = asymptotic_differential_field(x, (1 / x,))
    field_element = field.element(1 + 1 / x)
    shadow_element = field.shadow_fields[0].element(1 + 1 / x)
    assert sp.simplify(field_element.differentiate().as_expr() + 1 / x**2) == 0
    assert sp.simplify(shadow_element.reciprocal(terms=3).truncate() - (1 - 1 / x + 1 / x**2)) == 0


def test_ode_generated_branch_adapts_via_its_native_transseries():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    forcing = sp.simplify(sp.diff(target, x) - target**2)
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing

    branches = nonlinear_differential_transseries(equation, y, x, point=0, terms=3)
    branch = next(item for item in branches if sp.simplify(item.series - target) == 0)
    element = branch.asymptotic_element()

    assert element.native is branch
    assert element.variable == x
    assert element.point == 0
    assert sp.simplify(element.to_transseries(3).truncate() - branch.transseries.truncate(3)) == 0


def test_cross_representation_product_propagates_a_multiseries_tail_certificate():
    x = sp.symbols("x", positive=True)
    multi = multiseries(sp.exp(1 / x), x, terms=6).asymptotic_element()
    exact = transseries_from_expression(
        1 + 1 / x, x, point=sp.oo, complete=True
    ).asymptotic_element()

    product = multi * exact
    assert product.remainder.kind is RemainderKind.BIG_O
    assert product.remainder.check() is True
    assert product.remainder.exact_expression is not None


def test_public_composition_and_asymptotic_integration_accept_protocol_representations():
    from asymptotic import asymptotic_integrate, compose_transseries

    x, z = sp.symbols("x z", positive=True)
    multi = multiseries(1 / x + 1 / x**2, x, terms=4)
    composed = compose_transseries(sp.exp(z), multi, argument=z, terms=3)
    assert sp.simplify(composed.truncate() - sp.exp(1 / x + 1 / x**2)) == 0

    primitive = asymptotic_integrate(multi, terms=3)
    # d(log(x) - 1/x)/dx = 1/x + 1/x**2
    assert sp.simplify(primitive.truncate() - (sp.log(x) - 1 / x)) == 0


def test_functional_inversion_is_available_through_the_common_protocol():
    x, y = sp.symbols("x y")
    trans = transseries_from_expression(x + x**2, x, point=0, complete=True)
    branch = trans.asymptotic_element().inverse_asymptotic(y, terms=4)

    inverse_prefix = branch.truncate()
    assert sp.expand(inverse_prefix - (y - y**2 + 2 * y**3 - 5 * y**4)) == 0
    assert sp.expand((x + x**2).subs(x, inverse_prefix).series(y, 0, 5).removeO() - y) == 0
