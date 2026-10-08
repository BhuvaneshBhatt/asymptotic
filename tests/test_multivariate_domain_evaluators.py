import sympy as sp

from asymptotic.blowup_geometry import weighted_spherical_atlas
from asymptotic.domain_cluster_geometry import transform_domain_to_chart
from asymptotic.multivariate_limits_advanced import multivariate_cluster_set


def test_wedge_is_pulled_to_angular_chart():
    x, y = sp.symbols("x y", real=True)
    atlas = weighted_spherical_atlas((x, y), (0, 0), (1, 1))
    d = transform_domain_to_chart(sp.And(y > x, y < 2 * x), atlas.charts[0])
    u, v = atlas.charts[0].angular_variables
    assert sp.simplify(sp.Equivalent(d, sp.And(v > u, v < 2 * u))) is sp.S.true


def test_weighted_cluster_evaluator_accepts_nonorthant_wedge():
    x, y = sp.symbols("x y", real=True)
    result = multivariate_cluster_set(
        x**2 / (x**2 + y**2), (x, y), (0, 0), domain=sp.And(y > x, y < 2 * x)
    )
    assert result.certified
    assert result.cluster_set is not None


def test_cusp_domain_transforms_under_weighted_blowup():
    x, y = sp.symbols("x y", real=True)
    atlas = weighted_spherical_atlas((x, y), (0, 0), (2, 3))
    d = transform_domain_to_chart(y**2 >= x**3, atlas.charts[0])
    assert d is not sp.S.false
