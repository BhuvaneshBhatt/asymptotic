import sympy as sp

from asymptotic.blowup_geometry import projective_blowup_atlas, projective_chart


def test_three_dimensional_projective_atlas_covers_rp2():
    x, y, z = sp.symbols("x y z", real=True)
    atlas = projective_blowup_atlas((x, y, z), (0, 0, 0))
    assert atlas.certified
    assert len(atlas.charts) == 3
    assert len(atlas.transitions) == 6
    assert all(len(c.angular_variables) == 3 for c in atlas.charts)


def test_projective_overlap_transition_is_exact():
    x, y, z = sp.symbols("x y z", real=True)
    atlas = projective_blowup_atlas((x, y, z), (0, 0, 0))
    tr = next(
        t for t in atlas.transitions if t.source_chart == 0 and t.target_chart == 1
    )
    source = atlas.charts[0]
    target = atlas.charts[1]
    assert tr.condition == sp.Ne(source.angular_variables[1], 0)
    subs = dict(tr.substitutions)
    assert (
        sp.simplify(subs[target.angular_variables[0]] - 1 / source.angular_variables[1])
        == 0
    )


def test_projective_chart_no_longer_bivariate_only():
    x, y, z, w = sp.symbols("x y z w", real=True)
    chart = projective_chart((x, y, z, w), (0, 0, 0, 0), 2)
    assert chart.angular_variables[2] == 1
