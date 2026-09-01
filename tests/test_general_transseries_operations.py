import sympy as sp

from asymptotic import (
    asymptotic_integrate,
    compose_transseries,
    inverse_asymptotic,
    transseries_from_expression,
)
from asymptotic.general_ops import LogExpInverseResult, inverse_logexp


def test_general_meromorphic_composition_uses_native_small_scale():
    x = sp.symbols("x", positive=True)
    z = sp.symbols("z")
    inner = transseries_from_expression(1 / x, x, point=sp.oo, complete=True)
    result = compose_transseries(1 / z + sp.sin(z), inner, argument=z, terms=4)
    expected = x + 1 / x - sp.Rational(1, 6) / x**3
    assert sp.simplify(result.truncate() - expected) == 0


def test_nested_logexp_composition_raises_and_lowers_height():
    x = sp.symbols("x", positive=True)
    z = sp.symbols("z")
    inner = transseries_from_expression(x + sp.log(x), x, point=sp.oo)
    exponential = compose_transseries(sp.exp(z), inner, argument=z, terms=4)
    assert sp.simplify(exponential.truncate() - x * sp.exp(x)) == 0
    logarithm = compose_transseries(sp.log(z), inner, argument=z, terms=4)
    expected = sp.log(x) + sp.log(x) / x - sp.log(x) ** 2 / (2 * x**2) + sp.log(x) ** 3 / (3 * x**3)
    assert sp.simplify(logarithm.truncate() - expected) == 0


def test_composition_accepts_outer_transseries():
    x, z = sp.symbols("x z", positive=True)
    outer = transseries_from_expression(sp.exp(z) + 1 / z, z, point=sp.oo)
    inner = transseries_from_expression(x + sp.log(x), x, point=sp.oo)
    result = compose_transseries(outer, inner, terms=3)
    assert result.truncate().has(sp.exp(x))
    assert result.truncate().has(sp.log(x))


def test_ecalle_inverse_iteration_for_x_plus_log_x():
    x, y = sp.symbols("x y", positive=True)
    result = inverse_logexp(x + sp.log(x), x, y, terms=4, iterations=4)
    expected = y - sp.log(y) + sp.log(y) / y + sp.log(y) ** 2 / (2 * y**2)
    assert sp.simplify(result.truncate() - expected) == 0
    assert result.seed == y


def test_log_height_reduction_inverts_x_exp_x():
    x, y = sp.symbols("x y", positive=True)
    result = inverse_asymptotic(x * sp.exp(x), x, y, terms=4)
    assert isinstance(result, LogExpInverseResult)
    expected = (
        sp.log(y)
        - sp.log(sp.log(y))
        + sp.log(sp.log(y)) / sp.log(y)
        + sp.log(sp.log(y)) ** 2 / (2 * sp.log(y) ** 2)
    )
    assert sp.simplify(result.truncate() - expected) == 0
    assert result.transformed_by_log == 1


def test_asymptotic_integration_exponential_by_parts():
    x = sp.symbols("x", positive=True)
    primitive = asymptotic_integrate(sp.exp(-(x**2)), x, point=sp.oo, terms=4)
    expected = sp.exp(-(x**2)) * (
        -1 / (2 * x) + 1 / (4 * x**3) - sp.Rational(3, 8) / x**5 + sp.Rational(15, 16) / x**7
    )
    assert sp.simplify(primitive.truncate() - expected) == 0
    residual = sp.simplify(sp.diff(primitive.truncate(), x) - sp.exp(-(x**2)))
    assert sp.simplify(residual - (-sp.Rational(105, 16) * sp.exp(-(x**2)) / x**8)) == 0


def test_asymptotic_integration_power_log_scale_transitions():
    x = sp.symbols("x", positive=True)
    assert (
        sp.simplify(
            asymptotic_integrate(1 / (x * sp.log(x)), x, point=sp.oo).truncate() - sp.log(sp.log(x))
        )
        == 0
    )
    primitive = asymptotic_integrate(x**2 * sp.log(x) ** 3, x, point=sp.oo, terms=4)
    expected = x**3 * (
        sp.log(x) ** 3 / 3 - sp.log(x) ** 2 / 3 + 2 * sp.log(x) / 9 - sp.Rational(2, 27)
    )
    assert sp.simplify(primitive.truncate() - expected) == 0
