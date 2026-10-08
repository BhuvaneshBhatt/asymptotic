import sympy as sp

from asymptotic.complex_cluster_geometry import branched_blowup_atlas


def test_nested_log_sqrt_divisors_pull_back_through_common_atlas():
    x, y = sp.symbols("x y", real=True)
    expr = sp.log(sp.sqrt(x + sp.I * y))
    charts, coverage = branched_blowup_atlas(expr, (x, y), (0, 0))
    assert coverage.certified
    assert charts
    assert all(len(c.divisors) >= 2 for c in charts)
    assert all(c.sectors for c in charts)


def test_branch_atlas_intersects_relative_wedge_domain():
    x, y = sp.symbols("x y", real=True)
    charts, coverage = branched_blowup_atlas(
        sp.log(x + sp.I * y), (x, y), (0, 0), domain=sp.And(x >= 0, y >= 0)
    )
    assert coverage.certified
    assert all(c.domain is not sp.S.false for c in charts)
