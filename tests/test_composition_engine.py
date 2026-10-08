import sympy as sp

from asymptotic import compose, series
from asymptotic.remainder import Remainder, RemainderKind
from asymptotic.transseries import transseries_from_expression


def test_compose_uses_special_function_provider_at_infinity():
    x = sp.symbols("x", positive=True)
    inner = series(x, x, point=sp.oo, terms=4, return_result=True)

    result = compose(sp.airyai, inner, terms=3, return_result=True)

    expected_leading = sp.exp(-sp.Rational(2, 3) * x ** sp.Rational(3, 2)) / (
        2 * sp.sqrt(sp.pi) * x ** sp.Rational(1, 4)
    )
    assert sp.simplify(result.leading_term.expression / expected_leading) == 1
    assert "special-function" in result.metadata["composition_providers"]
    assert result.remainder.kind is RemainderKind.BIG_O


def test_compose_uses_uniform_provider_for_turning_point_expression():
    x = sp.symbols("x", positive=True)
    a = sp.symbols("a", real=True)
    inner = series(x, x, point=sp.oo, terms=4, return_result=True)

    result = compose(
        lambda z: sp.besselj(z, z + a * z ** sp.Rational(1, 3)),
        inner,
        terms=3,
        return_result=True,
    )

    assert "uniform-asymptotic-regime" in result.metadata["composition_providers"]
    assert result.truncate().has(sp.airyai)


def test_compose_two_series_propagates_outer_remainder():
    x, z = sp.symbols("x z", positive=True)
    inner = series(1 / x, x, point=sp.oo, terms=4, return_result=True)
    outer = transseries_from_expression(
        1 + z + z**2,
        z,
        point=0,
        remainder=Remainder.big_o(z**3, z, 0),
    )

    result = compose(outer, inner, terms=4, return_result=True)

    assert sp.expand(result.truncate() - (1 + 1 / x + x**-2)) == 0
    assert result.remainder.kind is RemainderKind.BIG_O
    assert sp.simplify(result.remainder.scale / x**-3) == 1
    assert "series-on-series" in result.metadata["composition_providers"]


def test_series_composition_requires_matching_germs():
    x, z = sp.symbols("x z", positive=True)
    inner = series(x, x, point=sp.oo, terms=3, return_result=True)
    outer = transseries_from_expression(1 + z, z, point=0, complete=True)

    try:
        compose(outer, inner, terms=3, return_result=True)
    except ValueError as exc:
        assert "expansion point" in str(exc)
    else:
        raise AssertionError("composition of incompatible germs must fail")
