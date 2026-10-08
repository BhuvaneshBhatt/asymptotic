import sympy as sp

from asymptotic.parameterized_geometry import parameterized_geometry_cells


def test_parameter_cells_detect_projective_pole_collision():
    x, y, a = sp.symbols("x y a", real=True)
    result = parameterized_geometry_cells(
        ((x + a * y) / (x - a * y),), (x, y), (0, 0), parameters=(a,)
    )
    assert result.certified
    assert len(result.cells) >= 2
    assert any(c.topology_predicates for c in result.cells)


def test_parameter_cells_detect_newton_three_variables():
    x, y, z, a = sp.symbols("x y z a", real=True)
    result = parameterized_geometry_cells(
        (x**2 + y**2 + a * z**2,), (x, y, z), (0, 0, 0), parameters=(a,)
    )
    assert result.certified
    assert len(result.cells) >= 2
    assert all(c.projective_atlas.certified for c in result.cells)
