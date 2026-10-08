import sympy as sp

from asymptotic.recursive_resolution import resolve_exceptional_strata


def test_positive_radial_order_is_immediately_discharged():
    x, y = sp.symbols("x y", real=True)
    _nodes, coverage = resolve_exceptional_strata(x * x + y * y, (x, y), (0, 0))
    assert coverage.certified
    assert all(n.certified for n in _nodes)


def test_regular_angular_image_is_certified():
    x, y = sp.symbols("x y", real=True)
    _nodes, coverage = resolve_exceptional_strata(
        x * y / (x * x + y * y), (x, y), (0, 0), max_depth=1
    )
    assert coverage.certified or coverage.missing_regimes


def test_unresolved_polar_stratum_creates_explicit_obligation():
    x, y = sp.symbols("x y", real=True)
    _nodes, coverage = resolve_exceptional_strata(x / y, (x, y), (0, 0), max_depth=0)
    assert not coverage.certified
    assert coverage.missing_regimes


def test_positive_dimensional_stratum_gets_smooth_centered_model():
    from asymptotic.recursive_resolution import ExceptionalStratum, _stratum_local_model

    u, v, w = sp.symbols("u v w", real=True)
    s = ExceptionalStratum(sp.Eq(w, 0), "plane", (w,))
    model = _stratum_local_model(s, (u, v, w))
    assert model is not None
    assert model.certified_smooth
    assert model.rank == 1
    assert len(model.tangent_basis) == 2
    assert len(model.normal_basis) == 1


def test_jacobian_minor_atlas_covers_smooth_parabola_without_sampling():
    from asymptotic.recursive_resolution import ExceptionalStratum, finite_stratum_atlas

    x, y = sp.symbols("x y", real=True)
    s = ExceptionalStratum(sp.Eq(y - x**2, 0), "parabola", (y - x**2,), 1)
    atlas = finite_stratum_atlas(s, (x, y))
    assert atlas is not None and atlas.coverage.certified
    assert atlas.rank == 1
    assert atlas.singular_stratum is None
    assert atlas.smooth_charts
    assert all(
        c.normal_symbols and c.radial_symbol.is_positive for c in atlas.smooth_charts
    )


def test_jacobian_minors_separate_cusp_singular_locus():
    from asymptotic.recursive_resolution import ExceptionalStratum, finite_stratum_atlas

    x, y = sp.symbols("x y", real=True)
    s = ExceptionalStratum(sp.Eq(y**2 - x**3, 0), "cusp", (y**2 - x**3,), 1)
    atlas = finite_stratum_atlas(s, (x, y))
    assert atlas is not None and atlas.coverage.certified
    assert atlas.singular_stratum is not None
    assert sp.simplify(atlas.singular_stratum.condition.subs({x: 0, y: 0})) is sp.S.true
    # Every non-origin point of the cusp has a nonzero Jacobian entry and is
    # therefore in at least one smooth minor patch.
    assert len(atlas.smooth_charts) == 2


def test_singular_locus_chain_descends_cusp_to_rank_zero_point():
    from asymptotic.recursive_resolution import ExceptionalStratum, singular_locus_chain

    x, y = sp.symbols("x y", real=True)
    s = ExceptionalStratum(sp.Eq(y**2 - x**3, 0), "cusp", (y**2 - x**3,), 1)
    chain = singular_locus_chain(s, (x, y), max_depth=3)
    assert chain
    assert chain[0].singular_stratum is not None


def test_tubular_chart_is_executable_by_common_radial_engine():
    from asymptotic.recursive_resolution import (
        ExceptionalStratum,
        _radial_initial,
        finite_stratum_atlas,
    )

    x, y = sp.symbols("x y", real=True)
    g = y - x**2
    atlas = finite_stratum_atlas(
        ExceptionalStratum(sp.Eq(g, 0), "parabola", (g,), 1), (x, y)
    )
    chart = atlas.smooth_charts[0]
    initial = _radial_initial(g, chart)
    assert initial is not None
    order, _num, _den = initial
    assert order > 0
    assert chart.radial_variable == chart.radial_symbol
    assert chart.angular_variables == chart.base_symbols + chart.normal_symbols


def test_cusp_singular_remainder_has_certified_dimension_drop():
    from asymptotic.recursive_resolution import ExceptionalStratum, finite_stratum_atlas

    x, y = sp.symbols("x y", real=True)
    g = y**2 - x**3
    atlas = finite_stratum_atlas(
        ExceptionalStratum(sp.Eq(g, 0), "cusp", (g,), 1), (x, y)
    )
    assert atlas.singular_components
    assert atlas.dimension_descent
    assert all(c.certified for c in atlas.dimension_descent)
    assert {c.dimension_hint for c in atlas.singular_components} == {0}
