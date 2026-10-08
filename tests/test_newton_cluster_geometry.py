import sympy as sp

from asymptotic.multivariate_limits_advanced import joint_cluster_geometry
from asymptotic.newton_geometry import (
    newton_polyhedral_fan,
    structured_semialgebraic_geometry,
)


def test_newton_polyhedral_fan_has_coverage_faces_and_adjacency():
    x, y = sp.symbols("x y", real=True)
    fan = newton_polyhedral_fan((x**2 + y**4,), (x, y), (0, 0))
    assert fan.coverage_certified
    assert fan.adjacency_certified
    assert {cone.dimension for cone in fan.cones} == {0, 1}
    assert any(cone.representative_weight == (2, 1) for cone in fan.cones)
    maximal = [cone for cone in fan.cones if cone.dimension == 1]
    assert len(maximal) == 2
    assert maximal[1].index in maximal[0].adjacent


def test_structured_geometry_implicitizes_parametric_cluster_curve():
    t = sp.symbols("t", real=True)
    curve = sp.ImageSet(sp.Lambda(t, sp.Tuple(t, 1 - t)), sp.Interval(0, 1))
    geometry = structured_semialgebraic_geometry(curve)
    assert geometry is not None
    assert geometry.dimension == 1
    assert len(geometry.components) == 1
    assert geometry.incidence_certified


def test_joint_cluster_geometry_carries_fan_and_structured_image():
    x, y = sp.symbols("x y", real=True)
    result = joint_cluster_geometry(
        (x**2 / (x**2 + y**2), y**2 / (x**2 + y**2)),
        (x, y),
        (0, 0),
    )
    assert result.certified
    assert result.newton_fan is not None
    assert result.newton_fan.coverage_certified
    assert result.strata
    assert result.strata[0].structured_geometry is not None
    assert result.strata[0].structured_geometry.dimension == 1


def test_three_variable_newton_fan_contains_all_face_dimensions():
    x, y, z = sp.symbols("x y z", real=True)
    fan = newton_polyhedral_fan((x**2 + y**4 + z**6,), (x, y, z), (0, 0, 0))
    assert fan.coverage_certified
    assert {cone.dimension for cone in fan.cones} == {0, 1, 2}
    assert any(cone.representative_weight == (6, 3, 2) for cone in fan.cones)
