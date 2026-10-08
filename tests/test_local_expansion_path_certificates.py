import sympy as sp

from asymptotic import limit
from asymptotic._limit_paths import _expansion_path_conflict
from asymptotic.local_expansion import adaptive_path_limit, local_series

x, y = sp.symbols("x y", real=True)
t = sp.symbols("t", positive=True)


def test_whole_germ_cancels_singular_elementary_terms():
    expr = (sp.cosh(y) - 1) / x**2 - y**2 / (2 * x**2)
    along = expr.subs({x: t, y: t}, simultaneous=True)
    certified = adaptive_path_limit(along, t)
    assert certified is not None
    assert certified[0] == 0


def test_elementary_singular_cancellation_can_disprove_joint_limit():
    expr = (sp.exp(y) - 1 - y) / x**2
    result = limit(expr, (x, y), (0, 0), return_result=True)
    assert result.status.name == "DOES_NOT_EXIST"
    witnesses = _expansion_path_conflict(expr, (x, y), (0, 0))
    assert witnesses is not None
    assert len({item.value for item in witnesses}) == 2
    for item in witnesses:
        assert item.method == "local_expansion_path"
        path = dict(item.substitutions)
        parameter = next(iter(set().union(*(v.free_symbols for v in path.values()))))
        along = expr.subs(path, simultaneous=True)
        assert sp.limit(along, parameter, 0, dir="+") == item.value
        assert (x**2).subs(path, simultaneous=True).is_positive is True


def test_complete_cancellation_is_not_a_negative_witness():
    expr = (sp.sin(x) - x) / x + (sp.cos(y) - 1)
    assert _expansion_path_conflict(expr, (x, y), (0, 0)) is None


def test_expansion_records_vanishing_remainder_order():
    expansion = local_series(sp.exp(t) - 1 - t, t, depth=4)
    assert expansion is not None
    assert expansion.order is not None
    assert sp.limit(sp.Abs(expansion.order), t, 0, dir="+") == 0
