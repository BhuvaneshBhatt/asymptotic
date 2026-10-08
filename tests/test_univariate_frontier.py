import sympy as sp

from asymptotic.algebraic_series import implicit_map_series
from asymptotic.generalized_series import generalized_series
from asymptotic.limits import LimitStatus, limit
from asymptotic.oscillatory_normal_form import oscillatory_cluster_interval
from asymptotic.special_function_infinity import special_function_infinity


def test_primepi_pnt_certificate():
    x = sp.symbols("x", positive=True)
    r = limit(sp.primepi(x) * sp.log(x) / x, (x,), (sp.oo,), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 1
    assert r.evidence


def test_multifrequency_periodic_image_is_exact_symbolic_image():
    x = sp.symbols("x", real=True)
    r = oscillatory_cluster_interval(sp.sin(x) + sp.cos(7 * x), x)
    assert r is not None and r.all_intermediate
    assert isinstance(r.set, sp.ImageSet)


def test_log_fractional_and_bessely_singular_series():
    x = sp.symbols("x", positive=True)
    for e in (
        sp.log(x),
        sp.sqrt(x) * sp.log(x),
        x**-3 + sp.log(x),
        sp.log(sp.sin(x)),
        sp.bessely(0, x),
    ):
        assert generalized_series(e, x, point=0, terms=5) is not None


def test_regular_vector_implicit_map_series():
    x, y1, y2 = sp.symbols("x y1 y2")
    r = implicit_map_series((y1 - x**3, y2 - y1 - x), (y1, y2), x, order=5)
    assert r is not None
    assert sp.expand(r.series[0] - x**3) == 0
    assert sp.expand(r.series[1] - (x + x**3)) == 0


def test_broadened_special_infinity_catalog():
    x = sp.symbols("x", positive=True)
    exprs = (
        sp.zeta(x),
        sp.primepi(x),
        sp.polygamma(0, x),
        sp.polygamma(1, x),
        sp.airyaiprime(x),
        sp.airybi(x),
        sp.airybiprime(x),
        sp.uppergamma(8, x),
    )
    assert all(special_function_infinity(e, x) is not None for e in exprs)
