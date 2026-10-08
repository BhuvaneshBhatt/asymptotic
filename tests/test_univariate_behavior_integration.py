import sympy as sp

from asymptotic.algebraic_series import algebraic_series
from asymptotic.generalized_series import generalized_series, singularity_series
from asymptotic.gruntz_normal_form import gruntz_normal_form
from asymptotic.oscillatory_normal_form import oscillatory_cluster_interval
from asymptotic.special_function_infinity import special_function_infinity


def test_infinity_is_normalized_to_positive_local_coordinate():
    x = sp.symbols("x", positive=True)
    n = gruntz_normal_form(sp.exp(x) / x**3, x)
    assert n.transformed == sp.exp(1 / n.local_variable) * n.local_variable**3


def test_generalized_series_records_exact_terms_and_order():
    x = sp.symbols("x")
    s = generalized_series(sp.sin(x), x, terms=6)
    assert s is not None and s.terms[0].exponent == 1
    assert sp.simplify(s.terms[0].coefficient - 1) == 0


def test_singularity_series_composes_outer_series():
    x = sp.symbols("x")
    q = singularity_series(lambda u: sp.erf(u), x**2, x, terms=4)
    assert sp.limit(q / x**2, x, 0) == 2 / sp.sqrt(sp.pi)


def test_algebraic_series_uses_existing_implicit_engine():
    x, y = sp.symbols("x y")
    r = algebraic_series(y - x - x * y, y, x, dependent_limit=0, order=4)
    assert r.expansion is not None and r.singular is False


def test_exact_periodic_cluster_interval():
    x = sp.symbols("x", real=True)
    r = oscillatory_cluster_interval(3 + 2 * sp.sin(x**2), x)
    assert r is not None and r.set == sp.Interval(1, 5) and r.all_intermediate


def test_special_function_infinity_leading_library():
    x, nu = sp.symbols("x nu", positive=True)
    assert special_function_infinity(sp.erfc(x), x) is not None
    assert special_function_infinity(sp.gamma(x), x) is not None
    assert special_function_infinity(sp.besselj(nu, x), x) is not None
    assert special_function_infinity(sp.bessely(nu, x), x) is not None
    assert special_function_infinity(sp.airyai(x), x) is not None
