import sympy as sp

from asymptotic.local_germ_algebra import GermOrder, LocalAsymptoticGerm, function_germ


def test_germ_product_quotient_orders():
    r = sp.symbols("r", positive=True)
    f = LocalAsymptoticGerm(GermOrder(2, 1, 0), radial_variable=r)
    g = LocalAsymptoticGerm(GermOrder(-1, 0, 3), radial_variable=r)
    assert (f * g).order == GermOrder(1, 1, 3)
    assert (f / g).order == GermOrder(3, 1, -3)


def test_radial_differentiation_handles_log_resonance():
    r = sp.symbols("r", positive=True)
    f = LocalAsymptoticGerm(GermOrder(0, 2, 0), radial_variable=r)
    d = f.differentiate_radial()
    assert d.order == GermOrder(-1, 1, 0)
    assert d.coefficient == 2


def test_bessel_rule_is_germ_data():
    r = sp.symbols("r", positive=True)
    g = function_germ(sp.besselj, sp.Integer(2), radial_variable=r)
    assert g.order.radial == 2
    assert sp.simplify(g.coefficient - sp.Rational(1, 8)) == 0


def test_real_imag_conjugate_are_germ_operations():
    r, u = sp.symbols("r u", positive=True)
    g = LocalAsymptoticGerm(GermOrder(1), 1 + sp.I * u, r, (u,), 1 + sp.I)
    assert g.conjugate().order == g.order
    assert g.real_part().order == g.order
    assert g.imag_part().order == g.order
