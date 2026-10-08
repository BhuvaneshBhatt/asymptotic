import sympy as sp

from asymptotic.limits import LimitStatus, limit


def test_erf_registered_germ_lifts_through_product_inner():
    x, y = sp.symbols("x y", real=True)
    u = x * y
    r = limit(sp.erf(u) / u, (x, y), (0, 0), domain=sp.Ne(u, 0), return_result=True)
    assert r.status is LimitStatus.PROVED
    assert sp.simplify(r.value - 2 / sp.sqrt(sp.pi)) == 0
    assert any(e.method == "local_germ_composition" for e in r.evidence)


def test_equal_first_order_special_germs_cancel():
    x, y = sp.symbols("x y", real=True)
    u = x * y
    r = limit(
        (sp.erfi(u) - sp.erf(u)) / u,
        (x, y),
        (0, 0),
        domain=sp.Ne(u, 0),
        return_result=True,
    )
    assert r.status is LimitStatus.PROVED and r.value == 0


def test_nested_erf_sine_germ_composes():
    x, y = sp.symbols("x y", real=True)
    u = x * y
    r = limit(
        sp.erf(sp.sin(u)) / u, (x, y), (0, 0), domain=sp.Ne(u, 0), return_result=True
    )
    assert r.status is LimitStatus.PROVED
    assert sp.simplify(r.value - 2 / sp.sqrt(sp.pi)) == 0
