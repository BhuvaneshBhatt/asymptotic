import sympy as sp

from asymptotic import (
    AsymptoticAlgebra,
    GrowthComparison,
    implicit_asymptotic,
    multiseries,
    nested_expansion,
    transseries_from_expression,
)
from asymptotic.remainder import RemainderKind


def test_coordinate_algebra_coerces_heterogeneous_representations_once():
    x = sp.symbols("x", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo, terms=4)
    multi = multiseries(sp.exp(1 / x), x, terms=6)
    nested = nested_expansion(1 + 1 / x, x, depth=1)

    product = algebra.multiply(multi, nested)
    assert product.variable == x
    assert product.point == sp.oo
    assert product.remainder.kind is RemainderKind.BIG_O
    assert product.remainder.check() is True


def test_algebra_routes_all_core_operations_through_one_coordinate_boundary():
    x, z = sp.symbols("x z", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo, terms=4)
    value = multiseries(1 + 1 / x, x, terms=5)

    derivative = algebra.differentiate(value)
    assert sp.simplify(derivative.as_expr() + 1 / x**2) == 0

    reciprocal = algebra.reciprocal(value, terms=3)
    assert sp.simplify(reciprocal.truncate() - (1 - 1 / x + 1 / x**2)) == 0

    composed = algebra.compose(value, sp.log(z), argument=z, terms=3)
    assert sp.simplify(composed.truncate() - (1 / x - 1 / (2 * x**2))) == 0

    small = transseries_from_expression(1 / x, x, point=sp.oo, complete=True)
    assert algebra.compare(value, small) is GrowthComparison.LARGER


def test_puiseux_and_implicit_branches_join_the_same_algebra_without_false_exactness():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit_asymptotic(y**2 - x - x**2, y, x, terms=2)
    branch = branches[0]

    element = branch.asymptotic_element()
    assert element.native is branch
    assert element.truncate() == branch.series.truncate()
    # This particular finite algebraic branch is not complete at two terms.
    assert element.remainder.kind is RemainderKind.UNKNOWN
    normal = element.algebra.normal_form(element, terms=2)
    assert normal.remainder.kind is RemainderKind.UNKNOWN


def test_algebra_rejects_cross_coordinate_binary_operations():
    x = sp.symbols("x", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo)
    local = transseries_from_expression(1 + x, x, point=0, complete=True)
    try:
        algebra.add(1 / x, local)
    except ValueError as exc:
        assert "coordinates" in str(exc)
    else:
        raise AssertionError("coordinate mismatch was silently coerced")
