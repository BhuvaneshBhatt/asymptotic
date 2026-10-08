import sympy as sp

from asymptotic.projective_clusters import parameterized_projective_cluster_strata


def test_symbolic_mobius_topology_splits_on_determinant():
    x, y, a = sp.symbols("x y a", real=True)
    strata = parameterized_projective_cluster_strata(x + a * y, x + y, (x, y))
    assert len(strata) == 2
    at_one = [s for s in strata if sp.simplify(s.condition.subs(a, 1)) is sp.S.true]
    generic = [s for s in strata if sp.simplify(s.condition.subs(a, 2)) is sp.S.true]
    assert len(at_one) == len(generic) == 1
    assert at_one[0].decomposition.cluster_set == sp.FiniteSet(1)
    assert generic[0].decomposition.cluster_set == sp.S.Reals


def test_two_parameter_mobius_discriminant_cells_are_exact():
    x, y, a, b = sp.symbols("x y a b", real=True)
    strata = parameterized_projective_cluster_strata(a * x + y, x + b * y, (x, y))
    assert len(strata) == 2
    assert any(
        s.condition.has(sp.Eq(a * b - 1, 0)) or s.condition.has(sp.Eq(1 - a * b, 0))
        for s in strata
    )
