import sympy as sp

from asymptotic import (
    as_element,
    differentiate,
    discover_scale,
    multiseries,
    nested_series,
)
from asymptotic.algebra import (
    AsymptoticElement,
    AsymptoticFieldElementProtocol,
)
from asymptotic.asymptotic_field import asymptotic_differential_field
from asymptotic.context import (
    AsymptoticGrowthComparison,
)
from asymptotic.nonlinear_ode import differential_transseries
from asymptotic.remainder import RemainderKind
from asymptotic.transseries import (
    transseries_from_expression,
)


def test_protocol_preserves_native_representation():
    x = sp.symbols("x", positive=True)
    trans = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)
    multi = multiseries(sp.exp(1 / x), x, terms=4, return_result=True)
    nested = nested_series(sp.log(x) + 1 / x, x, depth=1, return_result=True)

    for native in (trans, multi, nested):
        element = as_element(native)
        assert isinstance(element, AsymptoticElement)
        assert isinstance(element, AsymptoticFieldElementProtocol)
        assert element.native is native
        assert element.variable == x
        assert element.point == sp.oo


def test_multiseries_truncation_propagates_first_term_remainder():
    x = sp.symbols("x", positive=True)
    series = multiseries(sp.exp(1 / x), x, terms=5, return_result=True)
    trunc = series.as_element().truncation(3)

    assert sp.simplify(trunc.prefix - (1 + 1 / x + 1 / (2 * x**2))) == 0
    assert trunc.remainder.kind is RemainderKind.BIG_O
    assert sp.simplify(trunc.remainder.scale - 1 / (6 * x**3)) == 0
    assert trunc.remainder.check() is True


def test_nested_truncation_uses_transseries_view():
    x = sp.symbols("x", positive=True)
    nested = nested_series(sp.log(x) + 1 / x, x, depth=1, return_result=True)
    element = nested.as_element()

    assert sp.simplify(element.to_transseries(3).truncate() - (sp.log(x) + 1 / x)) == 0
    assert sp.simplify(element.truncate(3) - (sp.log(x) + 1 / x)) == 0


def test_cross_representation_certified_arithmetic():
    x = sp.symbols("x", positive=True)
    trans = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)
    multi = multiseries(1 / x + 1 / x**2, x, terms=3, return_result=True)

    result = as_element(trans) * as_element(multi)
    assert isinstance(result, AsymptoticElement)
    assert sp.simplify(result.truncate() - (1 / x + 2 / x**2 + 1 / x**3)) == 0


def test_protocol_composition_reciprocal_comparison_and_calculus():
    x, z = sp.symbols("x z", positive=True)
    trans = transseries_from_expression(1 + 1 / x, x, point=sp.oo, complete=True)
    element = trans.as_element()

    composed = element.compose(sp.exp(z), argument=z, terms=3)
    expected = sp.E * (1 + 1 / x + 1 / (2 * x**2))
    assert sp.simplify(composed.truncate() - expected) == 0

    reciprocal = element.reciprocal(terms=3)
    assert sp.simplify(reciprocal.truncate() - (1 - 1 / x + 1 / x**2)) == 0

    small = as_element(1 / x, x, point=sp.oo)
    assert element.compare(small) is AsymptoticGrowthComparison.LARGER

    differentiated = element.differentiate()
    assert sp.simplify(differentiated.as_expr() + 1 / x**2) == 0
    # The top-level calculus API preserves the native representation.
    assert type(differentiate(trans, return_result=True)) is type(trans)


def test_scale_and_shadow_field_elements_join_common_protocol():
    x = sp.symbols("x", positive=True)
    scale = discover_scale(1 / x + sp.exp(-x), x)
    slow = scale.element(0)
    fast = scale.element(1)
    assert slow.compare(fast) is AsymptoticGrowthComparison.LARGER

    field = asymptotic_differential_field(x, (1 / x,))
    field_element = field.element(1 + 1 / x)
    shadow_element = field.shadow_fields[0].element(1 + 1 / x)
    assert sp.simplify(field_element.differentiate().as_expr() + 1 / x**2) == 0
    assert (
        sp.simplify(
            shadow_element.reciprocal(terms=3).truncate() - (1 - 1 / x + 1 / x**2)
        )
        == 0
    )


def test_ode_generated_branch_adapts_via_its_native_transseries():
    x = sp.symbols("x", positive=True)
    y = sp.Function("y")
    target = 1 / x + x
    forcing = sp.simplify(sp.diff(target, x) - target**2)
    equation = sp.diff(y(x), x) - y(x) ** 2 - forcing

    branches = differential_transseries(equation, y, x, point=0, terms=3)
    branch = next(item for item in branches if sp.simplify(item.series - target) == 0)
    element = branch.as_element()

    assert element.native is branch
    assert element.variable == x
    assert element.point == 0
    assert (
        sp.simplify(
            element.to_transseries(3).truncate() - branch.transseries.truncate(3)
        )
        == 0
    )


def test_cross_representation_product_propagates_tail_certificate():
    x = sp.symbols("x", positive=True)
    multi = multiseries(sp.exp(1 / x), x, terms=6, return_result=True).as_element()
    exact = transseries_from_expression(
        1 + 1 / x, x, point=sp.oo, complete=True
    ).as_element()

    product = multi * exact
    assert product.remainder.kind is RemainderKind.BIG_O
    assert product.remainder.check() is True
    assert product.remainder.exact_expression is not None


def test_public_calculus_accepts_protocol_objects():
    from asymptotic import integrate
    from asymptotic.general_ops import compose_transseries

    x, z = sp.symbols("x z", positive=True)
    multi = multiseries(1 / x + 1 / x**2, x, terms=4, return_result=True)
    composed = compose_transseries(sp.exp(z), multi, argument=z, terms=3)
    assert sp.simplify(composed.truncate() - sp.exp(1 / x + 1 / x**2)) == 0

    primitive = integrate(multi, terms=3, return_result=True)
    # d(log(x) - 1/x)/dx = 1/x + 1/x**2
    assert sp.simplify(primitive.truncate() - (sp.log(x) - 1 / x)) == 0


def test_functional_inversion_is_available_common_protocol():
    x, y = sp.symbols("x y")
    trans = transseries_from_expression(x + x**2, x, point=0, complete=True)
    branch = trans.as_element().inverse(y, terms=4)

    inverse_prefix = branch.truncate()
    assert sp.expand(inverse_prefix - (y - y**2 + 2 * y**3 - 5 * y**4)) == 0
    assert (
        sp.expand((x + x**2).subs(x, inverse_prefix).series(y, 0, 5).removeO() - y) == 0
    )
