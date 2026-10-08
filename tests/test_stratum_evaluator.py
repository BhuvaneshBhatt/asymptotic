import sympy as sp

from asymptotic.local_strata import (
    LocalStratum,
    combine_local_strata,
    frontier_incidence_complex,
)
from asymptotic.multivariate_geometry_extended import phase_geometry
from asymptotic.multivariate_limits_advanced import joint_cluster_geometry
from asymptotic.stratum_evaluator import evaluate_local_stratum


def test_recursive_evaluator_forwards_image_newton_node():
    x = sp.symbols("x", real=True)
    image = LocalStratum(
        "semialgebraic_stratum",
        sp.And(x >= 0, x <= 1),
        1,
        payload=None,
    )
    # Generic structural nodes do not invent a set from a bare constraint.
    root = combine_local_strata(image, kind="newton_cluster_fiber")
    result = evaluate_local_stratum(root)
    assert result.limiting_image is None
    assert not result.certified


def test_recursive_evaluator_preserves_phase_fiber():
    x, y = sp.symbols("x y", real=True)
    phase = phase_geometry((sp.sin(1 / (x**2 + y**2)),), (x, y), (0, 0))
    result = evaluate_local_stratum(phase)
    assert result.certified
    assert isinstance(result.limiting_image, sp.ConditionSet)
    assert result.stratum.stratum_kind == "phase_torus"


def test_recursive_frontier_is_propagated_from_cluster_image():
    x, y = sp.symbols("x y", real=True)
    result = joint_cluster_geometry(
        (x**2 / (x**2 + y**2), y**2 / (x**2 + y**2)),
        (x, y),
        (0, 0),
    )
    assert result.certified
    evaluations = tuple(evaluate_local_stratum(s) for s in result.strata)
    assert evaluations
    assert any(e.limiting_image is not None for e in evaluations)
    # Structured limiting images carry their frontier through the recursive tree.
    assert any(e.frontier_complex is not None for e in evaluations)


def test_frontier_complex_segment_has_dimension_drop_and_incidence():
    t = sp.symbols("t", real=True)
    fc = frontier_incidence_complex(sp.And(t >= 0, t <= 1), (t,))
    assert fc is not None and fc.certified
    assert {s.dimension for s in fc.strata} == {0, 1}
    assert fc.incidence
