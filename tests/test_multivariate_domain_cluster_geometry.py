import sympy as sp

from asymptotic.domain_cluster_geometry import local_domain_components


def test_wedge_is_one_accumulating_local_component():
    x, y = sp.symbols("x y", real=True)
    g = local_domain_components(sp.And(y > x, y < 2 * x), (x, y), (0, 0))
    assert g.certified and len(g.components) == 1


def test_deleted_algebraic_variety_splits_local_components():
    x, y = sp.symbols("x y", real=True)
    g = local_domain_components(sp.Ne(x, 0), (x, y), (0, 0))
    assert g.certified and len(g.components) == 2


def test_cusp_semialgebraic_geometry_is_certified():
    x, y = sp.symbols("x y", real=True)
    g = local_domain_components(y**2 >= x**3, (x, y), (0, 0))
    assert g.certified and g.components
