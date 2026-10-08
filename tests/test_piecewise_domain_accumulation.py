import sympy as sp

from asymptotic.domain_cluster_geometry import local_domain_accumulates
from asymptotic.limits import LimitStatus, limit


def test_target_only_piece_does_not_affect_limit():
    x, y = sp.symbols("x y", real=True)
    r2 = x * x + y * y
    f = sp.Piecewise((r2, sp.Ne(r2, 0)), (sp.Integer(1), True))
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0


def test_two_accumulating_halfplanes_with_different_values_prove_dne():
    x, y = sp.symbols("x y", real=True)
    f = sp.Piecewise((sp.Integer(1), x >= 0), (sp.Integer(2), True))
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_piecewise_priority_removes_shadowed_branch():
    x, y = sp.symbols("x y", real=True)
    f = sp.Piecewise((x * x + y * y, True), (sp.Integer(7), x > 0), evaluate=False)
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 0


def test_ambient_domain_can_remove_conflicting_piece():
    x, y = sp.symbols("x y", real=True)
    f = sp.Piecewise((sp.Integer(1), x >= 0), (sp.Integer(2), True))
    r = limit(f, (x, y), (0, 0), domain=(x >= 0), return_result=True)
    assert r.status is LimitStatus.PROVED and r.value == 1


def test_punctured_accumulation_ignores_target_membership():
    x, y = sp.symbols("x y", real=True)
    assert local_domain_accumulates(sp.Eq(x * x + y * y, 0), (x, y), (0, 0)) is False
    assert local_domain_accumulates(sp.Ne(x * x + y * y, 0), (x, y), (0, 0)) is True


def test_lower_dimensional_piece_can_other_coordinate():
    x, y = sp.symbols("x y", real=True)
    f = sp.Piecewise((1, sp.Eq(y, 0)), (2, True))
    r = limit(f, (x, y), (0, 0), return_result=True)
    assert r.status is LimitStatus.DOES_NOT_EXIST


def test_disconnected_relative_domain_preserves_conflicting_components():
    x, y = sp.symbols("x y", real=True)
    r = limit(
        sp.sign(x), (x, y), (0, 0), domain=sp.Or(x > 0, x < 0), return_result=True
    )
    assert r.status is LimitStatus.DOES_NOT_EXIST
