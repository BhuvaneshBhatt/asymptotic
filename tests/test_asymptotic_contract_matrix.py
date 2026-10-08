import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from asymptotic import (
    integrate,
    multiseries,
    relation,
)


def test_integrate_expression_dispatch_is_available_at_root():
    x = sp.symbols("x", positive=True)
    result = integrate(1 / x**2, x, point=sp.oo, terms=3, return_result=True)
    assert sp.simplify(sp.diff(result.truncate(), x) - 1 / x**2) == 0


@given(st.integers(min_value=2, max_value=6))
def test_requesting_more_multiseries_terms_preserves_existing_prefix(terms):
    x = sp.symbols("x", positive=True)
    low = multiseries(sp.exp(1 / x + 1 / x**2), x, terms=terms, return_result=True)
    high = multiseries(sp.exp(1 / x + 1 / x**2), x, terms=terms + 1, return_result=True)
    assert low.terms(terms) == high.terms(terms)


def test_multivariate_relation_rays_only_falsify_not_certify():
    x, y = sp.symbols("x y")
    result = relation(
        x**2 + y**2, x**2, (x, y), (0, 0), relation="equivalent", return_result=True
    )
    assert result.value is False
    assert result.certified is True

    inconclusive = relation(
        x**2 + y**2,
        x**2 + y**2,
        (x, y),
        (0, 0),
        relation="equivalent",
        return_result=True,
    )
    assert inconclusive.value is None
    assert inconclusive.certified is False
