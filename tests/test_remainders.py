import sympy as sp

from asymptotic import (
    AsymptoticContext,
    AsymptoticRemainder,
    RemainderKind,
    TransseriesExpansion,
    compose_transseries,
    transseries_from_expression,
)


def test_exact_truncation_retains_exact_omitted_tail_and_big_o():
    x = sp.symbols("x", positive=True)
    s = transseries_from_expression(1 / x + 2 / x**2 + 3 / x**3, x, point=sp.oo, complete=True)
    t = s.truncation(1)
    assert sp.simplify(t.prefix - 1 / x) == 0
    assert t.remainder.kind is RemainderKind.BIG_O
    assert sp.simplify(t.remainder.scale - x**-2) == 0
    assert sp.simplify(t.remainder.exact_expression - (2 / x**2 + 3 / x**3)) == 0
    assert sp.simplify(t.reconstruct() - s.truncate()) == 0
    assert t.remainder.check() is True


def test_prefix_preserves_preexisting_little_o_information():
    x = sp.symbols("x", positive=True)
    rem = AsymptoticRemainder.little_o(x**-3, x, sp.oo)
    s = transseries_from_expression(1 / x + 1 / x**2, x, point=sp.oo, remainder=rem)
    p = s.prefix(1)
    assert p.remainder.kind is RemainderKind.BIG_O
    assert sp.simplify(p.remainder.scale - x**-2) == 0


def test_remainder_addition_and_product_rules():
    x = sp.symbols("x", positive=True)
    a = TransseriesExpansion.from_terms(
        x,
        sp.oo,
        (),
        center=1,
        remainder=AsymptoticRemainder.big_o(1 / x, x, sp.oo),
    )
    b = transseries_from_expression(
        x,
        x,
        point=sp.oo,
        remainder=AsymptoticRemainder.little_o(1, x, sp.oo),
    )
    product = a * b
    assert product.remainder.kind is RemainderKind.BIG_O
    assert product.remainder.check() is None
    # Dominant error is x * O(1/x) = O(1).
    assert sp.simplify(product.remainder.scale - 1) == 0


def test_differentiation_does_not_unsafely_differentiate_big_o():
    x = sp.symbols("x", positive=True)
    s = transseries_from_expression(
        1 / x,
        x,
        point=sp.oo,
        remainder=AsymptoticRemainder.big_o(x**-2, x, sp.oo),
    )
    d = s.differentiate()
    assert d.remainder.kind is RemainderKind.UNKNOWN


def test_analytic_composition_has_taylor_remainder():
    x, z = sp.symbols("x z", positive=True)
    inner = transseries_from_expression(1 / x, x, point=sp.oo, complete=True)
    out = compose_transseries(sp.sin(z), inner, argument=z, terms=4)
    assert out.remainder.kind is RemainderKind.BIG_O
    # The generic Taylor theorem gives the safe (not necessarily sharp) O(x^-4).
    assert sp.simplify(out.remainder.scale - x**-4) == 0
    assert sp.simplify(out.truncate() - (1 / x - sp.Rational(1, 6) / x**3)) == 0


def test_exact_recursive_exponential_representation_remains_exact():
    x, z = sp.symbols("x z", positive=True)
    inner = transseries_from_expression(1 / x, x, point=sp.oo, complete=True)
    out = compose_transseries(sp.exp(z), inner, argument=z, terms=4)
    assert out.remainder.kind is RemainderKind.EXACT
    assert sp.simplify(out.truncate() - sp.exp(1 / x)) == 0


def test_asymptotic_integration_records_next_term_remainder():
    from asymptotic import asymptotic_integrate

    x = sp.symbols("x", positive=True)
    source = transseries_from_expression(sp.exp(-(x**2)), x, point=sp.oo, complete=True)
    primitive = asymptotic_integrate(source, terms=4)
    assert primitive.remainder.kind is RemainderKind.BIG_O
    assert (
        sp.simplify(primitive.remainder.scale + sp.Rational(105, 32) * sp.exp(-(x**2)) / x**9) == 0
    )
    residual = sp.simplify(sp.diff(primitive.truncate(), x) - sp.exp(-(x**2)))
    assert residual == -sp.Rational(105, 16) * sp.exp(-(x**2)) / x**8


def test_remainder_replay_rejects_mismatched_context():
    import pytest

    x, y = sp.symbols("x y", positive=True)
    remainder = AsymptoticRemainder.big_o(1 / x, x, sp.oo, exact_expression=1 / x**2)

    with pytest.raises(ValueError, match="different coordinates"):
        remainder.check(context=AsymptoticContext(y, sp.oo))
    with pytest.raises(ValueError, match="different coordinates"):
        remainder.check(context=AsymptoticContext(x, 0))
