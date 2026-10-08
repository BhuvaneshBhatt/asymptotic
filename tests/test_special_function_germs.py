import sympy as sp

from asymptotic.local_germ_algebra import function_germ


def test_erf_and_erfi_linear_germs():
    r = sp.symbols("r", positive=True)
    for f in (sp.erf, sp.erfi):
        g = function_germ(f, radial_variable=r)
        assert g.order.radial == 1
        assert sp.simplify(g.coefficient - 2 / sp.sqrt(sp.pi)) == 0


def test_gamma_pole_germ():
    r = sp.symbols("r", positive=True)
    g = function_germ(sp.gamma, radial_variable=r)
    assert g.order.radial == -1 and g.coefficient == 1


def test_besselj_varying_order_germ_retained():
    r = sp.symbols("r", positive=True)
    nu = sp.symbols("nu")
    g = function_germ(sp.besselj, nu, radial_variable=r)
    assert g.order.radial == nu
    assert sp.simplify(g.coefficient - 1 / (2**nu * sp.gamma(nu + 1))) == 0
