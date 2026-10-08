import sympy as sp

from asymptotic.discrete_limits import discrete_limit
from asymptotic.local_expansion import local_series
from asymptotic.mellin import mellin
from asymptotic.products import product


def test_erf_infinity_provider_has_vanishing_remainder():
    x = sp.symbols("x", positive=True)
    expansion = local_series(sp.erf(x), x, sp.oo, depth=4)
    assert expansion is not None
    assert expansion.provider == "special-function"
    assert sp.limit(sp.Abs(expansion.order), x, sp.oo) == 0


def test_fresnel_provider_records_oscillatory_endpoint_term():
    x = sp.symbols("x", positive=True)
    expansion = local_series(sp.fresnelc(x), x, sp.oo, depth=4)
    assert expansion is not None
    assert expansion.prefix.has(sp.sin(sp.pi * x**2 / 2))


def test_discrete_limit_uses_continuous_restriction_when_available():
    n = sp.symbols("n", integer=True, positive=True)
    result = discrete_limit((n + 1) / n, n, return_result=True)
    assert result.certified
    assert result.value == 1


def test_product_refuses_unresolved_sign():
    k, n = sp.symbols("k n", integer=True, positive=True)
    a = sp.symbols("a", real=True)
    result = product(a + k / n, k, 1, n, parameter=n)
    assert not result.certified


def test_mellin_exposes_unresolved_nonrational_poles():
    x, s = sp.symbols("x s", positive=True)
    result = mellin(sp.exp(-x), x, s)
    assert not result.certified
    assert result.obligations
