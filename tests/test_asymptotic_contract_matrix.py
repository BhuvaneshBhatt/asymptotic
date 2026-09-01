import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from asymptotic import (
    asymptotic_integrate,
    asymptotic_relation,
    multiseries,
)
from asymptotic.general_ops import asymptotic_integral


def test_noun_style_integral_alias_has_identical_semantics():
    x = sp.symbols("x", positive=True)
    left = asymptotic_integral(1 / x**2, x, point=sp.oo, terms=3)
    right = asymptotic_integrate(1 / x**2, x, point=sp.oo, terms=3)
    assert sp.simplify(left.truncate() - right.truncate()) == 0


@given(st.integers(min_value=2, max_value=6))
def test_requesting_more_multiseries_terms_preserves_existing_prefix(terms):
    x = sp.symbols("x", positive=True)
    low = multiseries(sp.exp(1 / x + 1 / x**2), x, terms=terms)
    high = multiseries(sp.exp(1 / x + 1 / x**2), x, terms=terms + 1)
    assert low.terms(terms) == high.terms(terms)


def test_multivariate_relation_rays_only_falsify_not_certify():
    x, y = sp.symbols("x y")
    result = asymptotic_relation(
        x**2 + y**2,
        x**2,
        (x, y),
        (0, 0),
        relation="equivalent",
    )
    assert result.value is False
    assert result.certified is True

    inconclusive = asymptotic_relation(
        x**2 + y**2,
        x**2 + y**2,
        (x, y),
        (0, 0),
        relation="equivalent",
    )
    assert inconclusive.value is None
    assert inconclusive.certified is False
