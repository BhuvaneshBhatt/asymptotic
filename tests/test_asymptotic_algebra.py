import sympy as sp

from asymptotic import (
    implicit,
    multiseries,
    nested_series,
)
from asymptotic.algebra import (
    AsymptoticAlgebra,
)
from asymptotic.context import (
    AsymptoticGrowthComparison,
)
from asymptotic.remainder import RemainderKind
from asymptotic.transseries import (
    transseries_from_expression,
)


def test_coordinate_algebra_coerces_heterogeneous_representations_once():
    x = sp.symbols("x", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo, terms=4)
    multi = multiseries(sp.exp(1 / x), x, terms=6, return_result=True)
    nested = nested_series(1 + 1 / x, x, depth=1, return_result=True)

    product = algebra.multiply(multi, nested)
    assert product.variable == x
    assert product.point == sp.oo
    assert product.remainder.kind is RemainderKind.BIG_O
    assert product.remainder.check() is True


def test_algebra_routes_all_core_coordinate_boundary():
    x, z = sp.symbols("x z", positive=True)
    algebra = AsymptoticAlgebra(x, sp.oo, terms=4)
    value = multiseries(1 + 1 / x, x, terms=5, return_result=True)

    derivative = algebra.differentiate(value)
    assert sp.simplify(derivative.as_expr() + 1 / x**2) == 0

    reciprocal = algebra.reciprocal(value, terms=3)
    assert sp.simplify(reciprocal.truncate() - (1 - 1 / x + 1 / x**2)) == 0

    composed = algebra.compose(value, sp.log(z), argument=z, terms=3)
    assert sp.simplify(composed.truncate() - (1 / x - 1 / (2 * x**2))) == 0

    small = transseries_from_expression(1 / x, x, point=sp.oo, complete=True)
    assert algebra.compare(value, small) is AsymptoticGrowthComparison.LARGER


def test_branches_share_algebra_without_false_exactness():
    x, y = sp.symbols("x y", positive=True)
    branches = implicit(y**2 - x - x**2, y, x, terms=2, return_result=True)
    branch = branches[0]

    element = branch.as_element()
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
        raise AssertionError("coordinate mismatch was coerced")
