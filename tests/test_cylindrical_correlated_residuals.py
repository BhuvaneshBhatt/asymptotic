import sympy as sp

from asymptotic.limits import LimitStatus, limit


def test_cylindrical_lift_keeps_passive_domain_independent():
    x, y = sp.symbols("x y", real=True)
    r = limit(sp.atan(1 / sp.Abs(x)), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == sp.pi / 2
    assert r.reduced_variables == (x,)
    assert any(e.method == "variable_subset_reduction" for e in r.evidence)
    restricted = limit(x, (x, y), (0, 0), domain=y > 0, return_result=True)
    assert restricted.reduced_variables == (x, y)
    assert not any(e.method == "variable_subset_reduction" for e in restricted.evidence)


def test_cylindrical_non_singleton_univariate_cluster_certifies_dne():
    x, y = sp.symbols("x y", real=True)
    r = limit((1 + x**2) * sp.sin(1 / x), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert {e.value for e in r.evidence} == {-1, 1}
    for evidence in r.evidence:
        sequence = dict(evidence.substitutions)
        index = next(iter(sequence[x].free_symbols))
        assert sp.limit(sequence[x], index, sp.oo) == 0
        assert sequence[x].is_positive is True
        along = ((1 + x**2) * sp.sin(1 / x)).subs(sequence, simultaneous=True)
        assert sp.limit(along, index, sp.oo) == evidence.value


def test_correlated_directional_acos_image_certifies_dne():
    x, y = sp.symbols("x y", real=True)
    rho = sp.sqrt(x * x + y * y)
    r = limit(rho / sp.acos(x / rho), (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST
    assert any(e.method == "cluster_set_semantics" for e in r.evidence)
